# Metrics Service 使用指南

## 概述

新的 Metrics Service 将 metrics 上报逻辑（包括 tensorboard、wandb、clearml）重构为一个独立的服务，类似 rollout service 的设计。该服务只在 step end 时记录一次，支持批量上报。

## 架构

```
┌─────────────────┐    HTTP/REST     ┌─────────────────┐
│    Client Code  │ ───────────────> │ Metrics Service │
│  (训练/推理代码)│                  │  (Ray Serve)    │
└─────────────────┘    JSON API      └────────┬────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │  Metrics Buffer   │
                                    └─────────┬─────────┘
                                              │
                               ┌──────────────┼──────────────┐
                               ▼              ▼              ▼
                        ┌──────────┐   ┌──────────┐   ┌──────────┐
                        │ Tensor-  │   │   WandB  │   │ ClearML  │
                        │  Board   │   │          │   │          │
                        └──────────┘   └──────────┘   └──────────┘
```

## 代码结构

```
relax/metrics/
├── __init__.py          # 包入口，导出 MetricsClient, get_metrics_client
├── service.py           # MetricsService (Ray Serve 部署) + MetricsBuffer
└── client.py            # MetricsClient (HTTP 客户端) + get_metrics_client

relax/utils/
├── metrics_service_adapter.py  # MetricsServiceAdapter (向后兼容适配器)
└── tracking_utils.py           # 集成入口 (init_tracking, log, flush_metrics)
```

## 主要特性

1. **独立服务**：使用 Ray Serve 部署，与主应用解耦
2. **批量上报**：只在 step end 时记录一次，减少网络开销
3. **向后兼容**：保持与现有 `tracking_utils.log()` 相同的接口
4. **多后端支持**：同时支持 TensorBoard、WandB、ClearML
5. **异步处理**：metrics 收集和上报分离

## 快速开始

### 1. 部署 Metrics Service

```python
import ray
from ray import serve
from relax.metrics.service import MetricsService
from relax.utils.misc import create_namespace

# 创建配置
args_dict = {
    'use_wandb': True,
    'use_tensorboard': True,
    'use_clearml': False,
    'tb_project_name': 'my-project',
    'tb_experiment_name': 'experiment-1',
    'wandb_project': 'my-project',
    'wandb_team': 'my-team',
    'use_metrics_service': True,
}
args = create_namespace(args_dict)

# 部署服务
ray.init()
deployment = MetricsService.bind(
    healthy=None,
    pg=None,
    config=args,
    role="metrics"
)
serve.run(deployment, name="metrics", route_prefix="/metrics")
```

### 2. 在现有代码中使用（向后兼容）

只需在配置中添加 `use_metrics_service=True`，现有代码无需修改：

```python
# 现有代码保持不变
from relax.utils import tracking_utils

# 初始化（添加 use_metrics_service=True 到 args）
tracking_utils.init_tracking(args)

# 记录 metrics（与之前完全一样）
metrics = {
    "step": current_step,
    "train/loss": loss_value,
    "train/accuracy": accuracy_value,
}
tracking_utils.log(args, metrics, "step")

# 在 step end 时调用 flush（新增）
tracking_utils.flush_metrics(args, step)
```

### 3. 直接使用 Metrics Client

```python
from relax.metrics.client import MetricsClient
from relax.utils.utils import get_serve_url

# 获取服务 URL（自动从 Ray Serve 获取）
service_url = get_serve_url(route_prefix="/metrics")
client = MetricsClient(service_url)

# 记录单个 metric
client.log_metric(step=1, metric_name="train/loss", metric_value=0.5)

# 批量记录 metrics
batch_metrics = {
    "train/loss": 0.5,
    "train/accuracy": 0.92,
    "perf/throughput": 150,
}
client.log_metrics_batch(step=1, metrics=batch_metrics)

# 在 step end 时上报
result = client.report_step(step=1)
print(f"Report result: {result}")
```

## 配置选项

### 必需配置

```python
args.use_metrics_service = True  # 启用 metrics service
# 服务 URL 会自动通过 get_serve_url() 获取，无需手动配置
```

### 后端配置（与之前相同）

```python
# TensorBoard
args.use_tensorboard = True
args.tb_project_name = "my-project"
args.tb_experiment_name = "experiment-1"

# WandB
args.use_wandb = True
args.wandb_project = "my-project"
args.wandb_team = "my-team"
args.wandb_group = "my-group"

# ClearML
args.use_clearml = True
# ClearML 会自动从环境变量读取配置
```

## 迁移指南

### 从旧系统迁移

1. **无痛迁移**：如果希望保持现有代码不变，只需：

   - 在配置中添加 `use_metrics_service=True`
   - 在适当位置（如 step end）添加 `tracking_utils.flush_metrics(args, step)`

2. **逐步迁移**：可以同时运行新旧系统，通过配置控制：

   - 设置 `use_metrics_service=False` 使用旧系统
   - 设置 `use_metrics_service=True` 使用新系统

### 代码示例对比

**之前**：

```python
# 每次需要记录时直接调用
tracking_utils.log(args, metrics, "step")
```

**之后（批量模式）**：

```python
# 在训练循环中
for step in range(total_steps):
    # ... 训练代码 ...

    # 记录 metrics（缓冲，不立即发送）
    metrics = {
        "step": step,
        "train/loss": loss,
        "train/accuracy": accuracy,
    }
    tracking_utils.log(args, metrics, "step")

    # 在 step end 时上报所有缓冲的 metrics
    tracking_utils.flush_metrics(args, step)
```

## API 参考

### Metrics Service HTTP API

- `POST /metrics/log_metric` - 记录单个 metric
- `POST /metrics/log_metrics_batch` - 批量记录 metrics
- `POST /metrics/report_step` - 上报指定 step 的所有 metrics
- `GET /metrics/health` - 健康检查
- `GET /metrics/query_metrics` - 获取已记录的 metrics
- `POST /metrics/clear_metrics` - 清除 metrics

### Python Client API

```python
class MetricsClient:
    def __init__(self, service_url: str = "http://localhost:8000/metrics")
    def log_metric(step, metric_name, metric_value, tags=None, immediate=False)
    def log_metrics_batch(step, metrics, tags=None, immediate=False)
    def report_step(step)
    def health_check()
    def clear_buffer(step=None)
    def get_buffered_metrics_count(step=None)
```

### 向后兼容 Adapter

```python
class MetricsServiceAdapter:
    def __init__(args)  # 服务 URL 自动通过 get_serve_url() 获取
    def log(metrics, step_key="step")  # 与 tracking_utils.log 相同接口
    def flush()
    def direct_log(step, metrics)
```

## 性能考虑

1. **网络延迟**：Metrics Service 是独立服务，会有网络往返开销
2. **批量优势**：只在 step end 上报一次，减少总请求数
3. **缓冲机制**：Client 端缓冲 metrics，减少网络调用
4. **异步处理**：Service 内部异步处理上报，不阻塞客户端

## 故障排除

### 常见问题

1. **服务不可达**：检查 Ray Serve 是否正确部署，以及网络连接
2. **Metrics 未上报**：确保调用了 `flush_metrics()` 或 `report_step()`
3. **后端配置错误**：检查 TensorBoard/WandB/ClearML 配置

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查服务健康
from relax.metrics.client import MetricsClient
from relax.utils.utils import get_serve_url

service_url = get_serve_url(route_prefix="/metrics")
client = MetricsClient(service_url)
health = client.health_check()
print(f"Service health: {health}")
```

## 示例

完整的示例请参考 `relax/entrypoints/deploy_metrics_service.py`。

运行示例：

```bash
python relax/entrypoints/deploy_metrics_service.py
```

## 总结

新的 Metrics Service 提供了：

1. **更好的架构**：服务化设计，与主应用解耦
2. **性能优化**：批量上报，减少网络开销
3. **易于维护**：集中管理所有 metrics 上报逻辑
4. **向后兼容**：现有代码无需修改即可迁移
5. **扩展性**：易于添加新的 metrics 后端

## TimelineTrace 接入

1. 参考 https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw 文档中 Trace Event 的定义
2. 修改 `relax/utils/timer.py` 中 Timer 的实现，每次记录完整的 Event Record 而不是耗时，Event 中按文档记录 tid、pid 等，注意，args 中存入步数和当前的调用栈，方便追溯位置。你必须保证所有 timer 调用的地方不被 break（log_dict 函数返回原本 name -> 耗时的结构），新增一个函数 `log_record` 返回 record 本身，在调用 tracking_utils.log 时候放到 metrics 中一并传给 metrics server。
3. 添加 TimelineTrace adapter 实现，并且接入 `relax/metrics/service.py` 和 `relax/metrics/client.py`，当 report_step 时，所有 record 事件单独进入 timeline trace adapter（或者改成 adapter 对不同类型 metrics 的监听逻辑）。生成一个完整的 timeline event record json，然后每个 step 都 dump 一次，覆盖原来的文件，写到 --timeline-dump-dir 中。
4. 新增 `relax/utils/arguments.py` 中对应的参数，只需要 `--timeline-dump-dir` 即可，为空代表关闭，为一个目录代表开启
