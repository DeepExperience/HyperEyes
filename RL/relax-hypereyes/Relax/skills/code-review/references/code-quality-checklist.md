# Code Quality Checklist (Relax Project)

## Error Handling

### Anti-patterns to Flag

```python
# Bad: Swallowed exception
try:
    risky_operation()
except Exception:
    pass

# Bad: Bare except — catches KeyboardInterrupt, SystemExit
try:
    operation()
except:
    handle_error()

# Good: Catch specific, preserve chain
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Operation failed: {e}")
    raise OperationError("Failed") from e
```

- Overly broad `except Exception` hiding real bugs
- Missing error handling around I/O, Ray remote calls, distributed ops
- Unhandled coroutine exceptions (`asyncio.create_task` without awaiting)

______________________________________________________________________

## Resource Management

```python
# Bad: Resource leak on exception
f = open("file.txt")
data = f.read()
f.close()

# Good: Context manager
with open("file.txt") as f:
    data = f.read()
```

Key resources in Relax: file handles, locks (`threading.Lock`, `asyncio.Lock`), GPU memory, temporary files, network sockets.

______________________________________________________________________

## Performance

### Hot Path Issues

- Regex compiled inside loops — compile once outside
- Redundant computation — same calculation repeated without caching
- String concatenation in loops — use `"".join()`

### Memory

- Unbounded collections growing without limit
- Large objects held past useful lifetime
- Loading entire large files — use streaming/iteration

### Caching

- `@lru_cache` / `@functools.cache` for pure expensive functions
- Cache without TTL → stale data

______________________________________________________________________

## Boundary Conditions

### None / Empty Handling

```python
# Bad: Truthy check when 0, "", [] are valid values
if value:
    process(value)

# Good: Explicit None check
if value is not None:
    process(value)
```

- Division by zero: `total / count` → `total / max(count, 1)`
- Empty collection access: `items[0]` without length check
- Off-by-one in slicing / ranges

______________________________________________________________________

## Type Safety

- Missing type hints on public functions
- Excessive `Any` usage
- Missing `Optional[]` for nullable values
- `isinstance` cascades (often a design smell — prefer polymorphism)

______________________________________________________________________

## Code Readability

- **Magic numbers/strings** → extract named constants
- **Complex nested conditionals** → extract to named booleans
- **Deep nesting** → use early returns / `continue`
- **Long functions** (>50 lines) → split by responsibility
- Public functions missing docstrings
