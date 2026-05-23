# Security and Reliability Checklist (Relax Project)

## Input/Output Safety

### Command Injection

```python
# Bad
os.system(f"rm -rf {user_input}")
subprocess.call(f"ls {path}", shell=True)

# Good: list form, no shell=True
subprocess.run(["rm", "-rf", user_input], check=True)
```

### Path Traversal

```python
# Bad: unsanitized path (checkpoint/model paths from config)
with open(f"/data/{filename}") as f:
    return f.read()

# Good: validate path stays within allowed directory
safe_path = os.path.realpath(os.path.join("/data", filename))
if not safe_path.startswith("/data/"):
    raise ValueError("Invalid path")
```

### Unsafe Deserialization

- `pickle.load()` on untrusted data → arbitrary code execution
- `yaml.load()` without `Loader=yaml.SafeLoader`
- `torch.load()` without `weights_only=True` (PyTorch ≥2.6 default changed)

### Secret Leakage

- API keys / tokens hardcoded in source
- Secrets logged or included in error messages
- Environment variables exposed in logs

______________________________________________________________________

## Race Conditions

### Shared State

```python
# Bad: non-atomic increment
counter = 0
def increment():
    global counter
    counter += 1  # Not atomic!

# Good: use lock
with lock:
    counter += 1
```

### Check-Then-Act (TOCTOU)

```python
# Bad
if os.path.exists(filepath):
    with open(filepath) as f:  # may be deleted between check and open
        data = f.read()

# Good: handle exception
try:
    with open(filepath) as f:
        data = f.read()
except FileNotFoundError:
    data = None
```

### Async Race Conditions

- Shared state between coroutines without `asyncio.Lock()`
- Multiple `await` calls without synchronization

______________________________________________________________________

## Runtime Risks

### Resource Exhaustion

```python
# Bad: no timeout
response = requests.get(url)

# Good
response = requests.get(url, timeout=30)
```

- Unbounded loops or recursion without depth limits
- Unbounded collections growing without limit

### Blocking Operations

- Sync I/O in async context (blocking the event loop)
- Heavy computation without yielding

______________________________________________________________________

## Distributed System Risks (Relax-Specific)

### Ray Actor Safety

- Actors sharing state without proper synchronization
- Missing error handling for actor failures
- Resource leaks in actor lifecycle

### Process Group Operations

- Collective ops (all_reduce, broadcast) not called by all ranks → hang
- Mismatched tensor shapes across ranks
- Deadlocks from incorrect operation ordering

### Checkpoint Safety

```python
# Bad: non-atomic write
torch.save(model.state_dict(), checkpoint_path)

# Good: atomic write via temp file + rename
temp_path = checkpoint_path + ".tmp"
torch.save(model.state_dict(), temp_path)
os.rename(temp_path, checkpoint_path)
```

- Race between checkpoint save and model update
- Missing validation of loaded checkpoints
