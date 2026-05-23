# Health Management System Design

## 1. Overview

This document describes the design of the health management system in Relax, which is responsible for monitoring service health status and triggering recovery actions for unhealthy services. The system supports both **in-place restart** (single service recovery) and **global restart** (full Controller re-initialization from zero).

### 1.1 Design Goals

- **Single Responsibility**: Each component has a single, well-defined purpose
- **Decoupling**: Internal implementation details are hidden from consumers
- **Clarity**: Names and interfaces accurately reflect functionality
- **Simplicity**: No over-engineering; focus on essential abstractions
- **Testability**: Components can be tested independently
- **Fault Tolerance**: Automatic detection and recovery from service failures

## 2. Architecture

### 2.1 Component Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Controller (Consumer)                        │
│  uses: HealthManager                                            │
│  owns: restart_serve() → in-place restart / global restart      │
│  args: --use-health-check, --max-global-restart                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│          HealthManager (Coordinator Layer)                      │
│  Owns and coordinates internal components                       │
│  Public Interface:                                              │
│    • start(on_unhealthy: Callable)                              │
│    • stop(timeout: float)                                       │
│    • mark_healthy / mark_unhealthy / update_heartbeat           │
│    • report_error / get_service_health / get_all_health         │
│    • get_current_step / increment_restart_count                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
            ┌────────┴──────────┐
            │                   │
┌───────────▼──────┐  ┌────────▼────────────────┐
│  HealthStatus    │  │  HealthChecker          │
│  (Remote Actor)  │  │  (Background Thread)    │
│                  │  │                         │
│ Stores per-svc   │  │ Periodic checking       │
│ health state:    │  │ Unhealthy detection     │
│ heartbeat, error │  │ Stale heartbeat detect  │
│ step, restart_ct │  │ Callback invocation     │
└──────────────────┘  └─────────────────────────┘
```

### 2.2 Data Flow

```
Service (Actor/Rollout/Critic/ActorFwd/...)
    │
    ├── healthy.update_heartbeat.remote(role, step)     ← periodic heartbeat
    ├── healthy.report_error.remote(role, error_msg)    ← on exception
    │
    ▼
HealthStatus (Ray Actor)
    │
    ├── get_unhealthy_services()  →  [roles with healthy=False]
    ├── get_stale_services()      →  [roles with heartbeat timeout]
    │
    ▼
HealthChecker (Background Thread, check every 1s)
    │
    ├── detects unhealthy → on_unhealthy(role) callback
    ├── detects stale     → mark_unhealthy + on_unhealthy(role)
    │
    ▼
Controller.restart_serve(role)
    │
    ├── actor / rollout / actor_fwd / restart_count >= 3
    │       → _global_restart()     (full teardown + re-init)
    │
    └── other roles
            → Service.restart()     (in-place restart)
```

### 2.3 Component Responsibilities

| Component              | Responsibility                                                           | Implementation           |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------ |
| **ServiceHealthState** | Per-service health data (healthy, error, heartbeat, step, restart_count) | Python dataclass         |
| **HealthStatus**       | Store and query all service health state                                 | Ray remote actor         |
| **HealthChecker**      | Periodically check health, detect stale heartbeat, trigger callbacks     | Background daemon thread |
| **HealthManager**      | Coordinate both components, expose unified interface                     | Composite wrapper        |
| **Controller**         | Decide restart strategy (in-place vs global), execute restart            | Main orchestrator        |

## 3. Component Specifications

### 3.1 ServiceHealthState (Dataclass)

```python
@dataclass
class ServiceHealthState:
    healthy: bool = True
    error: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    current_step: int = 0
    task_running: bool = False
    restart_count: int = 0
```

### 3.2 HealthStatus (Remote Actor)

**Interface**:

```python
@ray.remote
class HealthStatus:
    HEARTBEAT_TIMEOUT = 120.0

    def mark_healthy(self, role: str) -> None
    def mark_unhealthy(self, role: str, error: Optional[str] = None) -> None
    def update_heartbeat(self, role: str, step: int = 0) -> None
    def set_task_status(self, role: str, running: bool) -> None
    def report_error(self, role: str, error: str) -> None
    def get_service_health(self, role: str) -> Dict
    def get_all_health(self) -> Dict[str, Dict]
    def get_unhealthy_services(self) -> List[str]
    def get_stale_services(self) -> List[str]
    def get_current_step(self, role: str) -> int
    def increment_restart_count(self, role: str) -> int
```

**Key behaviors**:

- `update_heartbeat()` also sets `healthy=True` — a heartbeat implies the service is alive.
- `report_error()` sets `healthy=False` and `task_running=False` — stops heartbeat stale detection.
- `get_stale_services()` returns roles where `task_running=True` and heartbeat exceeds `HEARTBEAT_TIMEOUT` (120s).

### 3.3 HealthChecker (Background Thread)

Two detection mechanisms:

1. **Error-based**: Services call `report_error()` → `get_unhealthy_services()` picks them up.
2. **Heartbeat-based**: Services stop sending heartbeats → `get_stale_services()` detects timeout.

**Thread Safety**: After `on_unhealthy()` callback, immediately checks `_stop_event` — global restart sets this to prevent the old checker from touching stale Ray actor handles.

### 3.4 HealthManager (Coordinator)

Top-level interface proxying all health operations to the underlying `HealthStatus` actor:

```python
class HealthManager:
    def start(self, on_unhealthy: Callable[[str], None]) -> None
    def stop(self, timeout: float = 2.0) -> None
    def is_running(self) -> bool
    def mark_healthy(self, role: str) -> None
    def mark_unhealthy(self, role: str, error: Optional[str] = None) -> None
    def update_heartbeat(self, role: str, step: int = 0) -> None
    def report_error(self, role: str, error: str) -> None
    def get_service_health(self, role: str) -> Dict
    def get_all_health(self) -> Dict[str, Dict]
    def get_current_step(self, role: str) -> int
    def increment_restart_count(self, role: str) -> int

    @property
    def status(self) -> HealthStatus  # exposed for service registration
```

## 4. Recovery Strategies

### 4.1 Two-Tier Recovery

| Condition                                     | Strategy         | Action                                       |
| --------------------------------------------- | ---------------- | -------------------------------------------- |
| Actor / Rollout / ActorFwd fails              | Global restart   | Full Controller teardown + re-init from zero |
| Any role restart_count ≥ 3                    | Global restart   | System is unstable, need clean slate         |
| Other roles fail (e.g., Critic, Advantages)   | In-place restart | Service.restart() — reuse PG, restore step   |
| Global restart count > `--max-global-restart` | Terminate        | `os._exit(1)` — system is unrecoverable      |

### 4.2 Global Restart Flow

```
Phase 1 — Teardown:
  1.1  Stop HealthManager (prevent further callbacks)
  1.2  Delete all Ray Serve deployments (services)
  1.3  Delete metrics deployment
  1.4  Delete DCS coordinator deployment
  1.5  Kill data system (storage units + controller)
  1.6  serve.shutdown() + ray.shutdown()
  1.7  Wait 5s for process cleanup
  1.8  ray.init() + serve.start()

Phase 2 — Re-initialize:
  2.1  self.__init__(config, runtime_env)
  2.2  Signal main thread via restart_done_event
  2.3  Main thread re-runs training_loop()
```

### 4.3 In-Place Restart Flow (Service.restart)

```
1. Save current step from running deployment (via HTTP)
2. Stop heartbeat thread
3. Gracefully stop old deployment (HTTP POST /stop_service)
4. serve.delete(role)
5. Wait 3s for resource release
6. Validate / rebuild placement group
7. Redeploy with same cls, config, PG
8. Restore step (HTTP POST /set_step)
9. Re-run service task
```

## 5. Heartbeat & Error Reporting

### 5.1 How Services Report Health

Each service reports health through the shared `HealthStatus` Ray actor handle:

```python
# In Actor._background_run():
try:
    ray.get(self.actor_model.train_fully_async(self.step))
    self.healthy.update_heartbeat.remote("actor", self.step + 1)
except Exception as e:
    error_msg = f"Actor training failed at step {self.step}: {e}"
    self.healthy.report_error.remote("actor", error_msg)
```

### 5.2 Service Heartbeat Thread

Each `Service` wrapper runs a background heartbeat thread (every 10s):

```python
def _heartbeat_loop():
    while not self._stop_heartbeat.is_set():
        self.healthy.update_heartbeat.remote(self.role, 0)
        time.sleep(10)
```

### 5.3 Detection Timing

| Detection Type  | Latency                            | Source                         |
| --------------- | ---------------------------------- | ------------------------------ |
| Error-based     | ~1s (HealthChecker check_interval) | Service calls `report_error()` |
| Heartbeat stale | ~120s (HEARTBEAT_TIMEOUT)          | No heartbeat for 120s          |

## 6. CLI Configuration

| Argument               | Type | Default | Description                            |
| ---------------------- | ---- | ------- | -------------------------------------- |
| `--use-health-check`   | flag | False   | Enable the global health check system  |
| `--max-global-restart` | int  | 3       | Max global restarts before termination |

## 7. Thread Coordination in Global Restart

```
Main Thread                              HealthChecker Thread
    │                                         │
    ├── run(run_all_services())               │
    │   (blocked on ray.get)                  │
    │                                         ├── detects unhealthy
    │                                         ├── on_unhealthy(role)
    │                                         │   └── Controller.restart_serve()
    │                                         │       └── _global_restart()
    │                                         │           ├── Phase 1: teardown
    │   ← exception from ray.get             │           │   (ray.shutdown kills tasks)
    │                                         │           │
    ├── catches exception                     │           │
    ├── sees self._restarting == True         │           │
    ├── _restart_done_event.wait() ──────►   │           │
    │   (blocked)                             │           ├── Phase 2: __init__()
    │                                         │           ├── restart_done_event.set()
    │   ◄──────────────────────────────────────────────── │
    ├── _restart_done_event received          │
    ├── continue → re-run run_all_services()  │     (thread exits)
```

**Key invariant**: The old HealthChecker's `_stop_event` is set before `ray.shutdown()` to ensure the old checker thread exits cleanly.

## 8. Design Decisions

### 8.1 Why `report_error()` Instead of Health Probes?

Services run long-blocking operations (train steps, NCCL collectives) — probes would timeout during normal operation. Services self-report errors via `report_error()`, supplemented by heartbeat stale detection for crash scenarios.

### 8.2 Why Global Restart for Actor/Rollout/ActorFwd?

These services have deep cross-dependencies (shared NCCL groups, weight sync channels, checkpoint engine connections). Recovering one without the others is unreliable.

### 8.3 Why `os._exit(1)` for Max Restart Exceeded?

`raise` only exits the main thread; Ray Serve and daemon threads keep the process alive. `os._exit(1)` is the only reliable way to terminate the entire process.

## 9. Future Extensions

### 9.1 GPU Memory Cleanup

Dispatch `nvidia-smi` + `kill` to all cluster nodes via Ray remote tasks before `ray.shutdown()`, with a local-only fallback after shutdown.

### 9.2 Per-Service Recovery Policies

Allow configuring per-role recovery strategies via arguments.

### 9.3 Adaptive Health Check

Adaptive backoff based on failure frequency.

______________________________________________________________________
