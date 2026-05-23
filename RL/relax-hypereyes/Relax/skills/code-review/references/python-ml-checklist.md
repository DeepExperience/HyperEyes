# ML/PyTorch Checklist (Relax Project)

## Tensor Operations

### Shape / Dtype / Device Mismatches

```python
# Bad: implicit assumptions
def compute_loss(logits, labels):
    return F.cross_entropy(logits, labels)

# Good: validate shapes
assert logits.dim() == 2 and labels.dim() == 1
assert logits.size(0) == labels.size(0)
```

- `squeeze()` without specifying dim — removes ALL size-1 dims
- `view()` vs `reshape()` — view requires contiguous memory
- Creating tensors without matching `dtype`/`device` of existing tensors

______________________________________________________________________

## Gradient Issues

### Key Anti-patterns

- **Missing `.detach()`** when storing tensors for later use (buffers, logging, target networks)
- **In-place ops** on `requires_grad` tensors (`x.add_(1)`)
- **Using `.data`** (deprecated, breaks autograd)
- **Missing `@torch.no_grad()`** during inference
- **`loss.item()` vs `loss`** — accumulating `loss` (not `.item()`) holds the computation graph

______________________________________________________________________

## Memory Management

```python
# Bad: holds computation graphs
losses = []
for batch in dataloader:
    loss = compute_loss(model(batch))
    losses.append(loss)  # graph retained!

# Good: detach to scalar
losses.append(loss.item())
```

- Large tensors not explicitly `del`-ed after use
- GPU memory fragmentation — `torch.cuda.empty_cache()` when needed
- Consider gradient checkpointing for memory-intensive models

______________________________________________________________________

## Distributed Training

### Collective Operation Ordering

All ranks **must** call collectives (all_reduce, all_gather, broadcast) in the same order.

```python
# Bad: conditional collective → hang
if rank == 0:
    dist.broadcast(tensor, src=0)

# Good: all ranks participate
dist.broadcast(tensor, src=0)
```

### Process Group Usage

```python
# Bad: implicit default group
dist.all_reduce(tensor)

# Good: explicit group
dist.all_reduce(tensor, group=self.data_parallel_group)
```

- Verify tensor shapes are identical across ranks before collectives
- Check `param.grad is not None` before gradient all-reduce

______________________________________________________________________

## Numerical Stability

- Division by zero → add `eps` or `clamp(min=eps)`
- `torch.log(prob)` with zero prob → use `log_softmax` or `clamp(min=1e-8)`
- Gradient clipping: `clip_grad_norm_` to prevent explosion
- Check for NaN/Inf gradients in training loop
- Mixed precision: use `GradScaler` properly

______________________________________________________________________

## Relax-Specific Patterns

### RolloutBatch

- Check for required keys before access
- Handle optional keys (`batch.get("values")` for non-PPO)
- Be deliberate about in-place mutation of batch dicts

### Loss Scaling (Megatron)

Loss must account for: `num_microbatches`, `global_batch_size`, `data_parallel_world_size`.

### Context Parallelism

- Use `all_gather_with_cp` for proper gathering across CP ranks
- Verify offset computation via `get_logits_and_tokens_offset_with_cp`

______________________________________________________________________

## Review Questions

| Area | Question |
|------|----------|
| Shapes | "What are the expected shapes here?" |
| Gradients | "Should this be detached?" |
| Distributed | "Is this collective called by all ranks?" |
| Memory | "Could this accumulate tensors?" |
| Numerical | "Could this overflow / divide by zero?" |
