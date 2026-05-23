# Update Weights Pipeline Optimization (Colocate Mode)

## Overview

In Colocate mode, after Actor training completes, Megatron-format weights must be converted to HF format and transferred to SGLang inference engines via CUDA IPC. Weights are sent in multiple **chunks**, and in the original implementation chunks were processed strictly in sequence — each chunk had to wait for its IPC transfer to complete via `ray.get` before the next could begin, resulting in high `update_weights_time`.

This optimization introduces **inter-chunk pipelining** (deferring `ray.get` so that IPC transfer overlaps with the next chunk's HF conversion) and **removes unnecessary barriers**, reducing the cost by roughly half.

______________________________________________________________________

## Background

### Scenario

When training large-scale MoE models on a multi-node cluster, the `perf/update_weights_time` metric was roughly twice the expected cost.

### Weight Update Call Chain

```
MegatronTrainRayActor.update_weights()
  └─ UpdateWeightFromTensor.update_weights()
       ├─ rank 0: pause_generation + flush_cache
       ├─ dist.barrier(gloo)                          ← required
       ├─ weights_getter()                            ← fetch Megatron local weights
       │
       ├─ for chunk in get_hf_weight_chunks():        ← multiple chunks
       │     ├─ _send_hf_params(chunk)                  HF conversion + serialize + Gloo gather
       │     └─ ray.get(refs)                         ← blocking wait!
       │
       └─ rank 0: continue_generation
```

The core bottleneck is the `ray.get(refs)` call in the chunk loop — each chunk must wait for IPC transfer to complete before starting the next, causing all chunks to run strictly in series and accounting for the majority of total cost.

______________________________________________________________________

## Optimization

### Inter-Chunk Pipelining

**Core idea**: defer `ray.get` until the *next* chunk has been prepared and sent, so that the current chunk's IPC transfer overlaps with the next chunk's HF conversion.

**Before** (strictly serial):

```
Chunk 0: [HF conv][serialize+gather][IPC xfer][wait]
Chunk 1:                                            [HF conv][serialize+gather][IPC xfer][wait]
Chunk 2:                                                                                      [HF conv]...
```

**After** (pipelined overlap):

```
Chunk 0: [HF conv][serialize+gather][IPC xfer]
Chunk 1:                        [HF conv][serialize+gather][wait prev][IPC xfer]
Chunk 2:                                               [HF conv][serialize+gather][wait prev][IPC xfer]
                                                                                                  [wait last]
```

Optimized code:

```python
prev_refs: list[ObjectRef] = []
prev_long_lived_tensors = None
for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
    if prev_refs:
        ray.get(prev_refs)
    del prev_long_lived_tensors
    prev_refs = refs
    prev_long_lived_tensors = long_lived_tensors
if prev_refs:
    ray.get(prev_refs)
del prev_long_lived_tensors
```


______________________________________________________________________

## Expected Benefits

- **Chunk loop time** — significantly reduced through pipelined overlap
- **Total update_weights_time** — reduced by roughly half overall

### Resource Impact

| Resource | Impact |
|----------|--------|
| GPU memory | Slight peak increase (one extra chunk of `long_lived_tensors`) |
| CPU memory | No change |
| Network bandwidth | No change (same total transfer, only timing overlap) |

### Applicability

This optimization benefits most in the following scenarios:

- **MoE / large-parameter models** — more chunks means greater pipelining gains
- **Colocate mode** — CUDA IPC path allows true parallelism between IPC transfer and HF conversion
- **Multi-node deployments** — cross-node engines use NCCL broadcast with higher latency, making pipelining overlap even more effective


______________________________________________________________________

## Next Steps

- [Distributed Checkpoint](./distributed-checkpoint.md) — the DCS weight synchronization mechanism used in non-colocated deployments
