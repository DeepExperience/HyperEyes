# Performance Tuning

A practical guide to maximizing training throughput in Relax. All parameters mentioned here are documented in the [Configuration Reference](./configuration.md).

---

## Profiling Training Performance

Before tuning, identify the bottleneck. Relax integrates PyTorch Profiler to generate TensorBoard-compatible traces.

### Enabling the Profiler

The profiler is controlled by `--profile-step-start` and `--profile-step-end` (Megatron native parameters) together with `--profile-target`:

```bash
python3 relax/entrypoints/train.py \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --use-tensorboard \
    --tb-project-name /path/to/tb_logs \
    # ... other args
```

You can specify multiple targets: `--profile-target train_overall train_actor train_log_probs`.

### Profiler Detail Flags

Three flags control what additional information the profiler records:

| Flag | Effect |
|---|---|
| `--profile-with-stack` | Record Python call stack in each trace event. Useful for identifying which code path triggers an expensive operation |
| `--profile-with-memory` | Track CUDA memory allocations/deallocations in the trace. Helps find memory spikes |
| `--profile-with-flops` | Estimate FLOPs for each operator. Useful for calculating hardware utilization (MFU) |

Example with all detail flags:

```bash
python3 relax/entrypoints/train.py \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --profile-with-stack \
    --profile-with-memory \
    --profile-with-flops \
    --use-tensorboard \
    --tb-project-name /path/to/tb_logs \
    # ... other args
```

::: warning
Enabling `--profile-with-stack` and `--profile-with-memory` adds overhead. Use them for diagnostic runs, not for production training.
:::

View the trace in TensorBoard:

```bash
tensorboard --logdir /path/to/tb_logs
```

---

## Dynamic Batching

Dynamic batching packs variable-length samples so each micro-batch approaches a target token count, improving GPU utilization compared to fixed-size micro-batches. It also serves as an effective OOM prevention mechanism — with a fixed `--micro-batch-size`, a batch of unusually long sequences can exceed GPU memory, while dynamic batching caps the total tokens per micro-batch to `--max-tokens-per-gpu`, keeping memory usage predictable.

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216
```

When using Context Parallelism (CP), set `--max-tokens-per-gpu` to approximately `max_response_len / cp_size`.

If computing log probs is a separate bottleneck, you can set a different token budget for that phase:

```bash
--log-probs-max-tokens-per-gpu 12288
```

::: tip
If you experience OOM during training, switching from fixed `--micro-batch-size` to `--use-dynamic-batch-size` with a conservative `--max-tokens-per-gpu` is often the first step. See [OOM Troubleshooting](./oom-troubleshooting.md) for more details.
:::

---

## Parallelism Configuration

### Tensor and Sequence Parallelism

For models that fit on a single node, Tensor Parallelism (TP) + Sequence Parallelism (SP) is the most common setup:

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel
```

Larger models (30B+) typically use TP=2 or TP=4 with SP enabled.

### MoE Expert Parallelism

For Mixture-of-Experts models (e.g., Qwen3-30B-A3B), distribute experts across GPUs:

```bash
--expert-model-parallel-size 2 \
--expert-tensor-parallel-size 1
```

### Context Parallelism

For long-context training, split the sequence across GPUs:

```bash
--context-parallel-size 2
```

---

## Activation Recomputation

Recomputation trades compute for memory. For most RL training workloads, enabling recomputation is recommended:

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

This configuration recomputes activations uniformly across layers. Adjust `--recompute-num-layers` based on your memory/compute tradeoff. See [Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM) for details on `selective` granularity and `block` method.

---

## Multimodal Processing Parallelism

When training multimodal models (e.g., Qwen3-VL), the HuggingFace processor for image/video data can become a CPU bottleneck due to Python's GIL. The `--mm-processor-pool-size` parameter creates a `ProcessPoolExecutor` to bypass GIL contention:

```bash
--mm-processor-pool-size 4
```

| Value | Behavior |
|---|---|
| `0` (default) | Uses `ThreadPoolExecutor` — subject to GIL contention |
| `> 0` | Creates a `ProcessPoolExecutor` with the specified number of workers for true CPU parallelism |

::: tip
Start with a pool size equal to the number of CPU cores available per GPU. For example, on a node with 64 CPUs and 8 GPUs, try `--mm-processor-pool-size 8`.
:::

---

## SGLang Inference Engine Tuning

### Memory Allocation

Control how much GPU memory SGLang reserves for KV cache:

```bash
--sglang-mem-fraction-static 0.8
```

Typical values range from 0.6 to 0.85. In colocate mode, a higher value (0.8) leaves less room for training but improves inference throughput. In fully async mode with separate GPUs, you can push this higher.

### Inference TP Size

Set the number of GPUs per inference engine instance:

```bash
--rollout-num-gpus-per-engine 1
```

For large models, increase this to match the model's minimum TP requirement. Using TP=1 for inference when possible gives the best per-query throughput.

---

## Partial Rollout

In long response scenarios (e.g., code generation, chain-of-thought reasoning), waiting for all samples to fully complete generation can leave training GPUs idle for extended periods. Partial Rollout avoids this by allowing incomplete samples to be interrupted and recycled back to the data buffer, so training can proceed with the samples that have finished:

```bash
--partial-rollout
```

### Controlling Abort Frequency

By default, a sample can be aborted an unlimited number of times. Set `--partial-rollout-max-aborted-count` to guarantee that a sample eventually completes generation after being aborted a certain number of times:

```bash
--partial-rollout \
--partial-rollout-max-aborted-count 3
```

### On-Policy Masking

When a sample is recycled and continues generation in a later rollout, its earlier tokens were generated under a previous policy version. Use `--mask-offpolicy-in-partial-rollout` to mask those off-policy tokens so only on-policy generated tokens participate in training:

```bash
--partial-rollout \
--mask-offpolicy-in-partial-rollout
```

::: tip
Partial Rollout is most effective for workloads where `max_response_len` is large (e.g., 8K+) and response lengths vary significantly. For short, uniform-length responses, the overhead of recycling may not be worthwhile.
:::

---

## Data Loading Optimization

### Streaming Dataset

For very large datasets that don't fit in memory, use streaming mode:

```bash
--use-streaming-dataset \
--streaming-buffer-size 10000
```

### Data Balancing

Distribute token counts evenly across data parallel ranks to reduce idle time:

```bash
--balance-data
```

::: warning
`--balance-data` is only available in colocate mode. It is not supported in fully async mode (`--fully-async`) because the TransferQueue data consumption path is incompatible with data balancing. Enabling both will raise an error at startup.
:::

::: warning
With `--balance-data`, different responses for the same prompt may be assigned to different training steps.
:::

---

## Weight Update Pipeline

For MoE models with large parameter counts, chunk the weight update to avoid memory spikes:

```bash
--update-weight-buffer-size 536870912  # 512 MB
```

Reduce this value if you observe memory pressure during weight synchronization.

---

## Fully Async Training

For maximum throughput, use the fully async training pipeline with dedicated GPU clusters for training and rollout:

```bash
--fully-async \
--max-staleness 1 \
--num-data-storage-units 1
```

### Staleness Tuning

The `--max-staleness` parameter controls how many versions behind the rollout data can be relative to the current training model. It directly affects the tradeoff between throughput and data freshness:

| Value | Behavior |
|---|---|
| `1` (default) | Training consumes only data from the current or previous version. Lower throughput but fresher data |
| `2-3` | Allows moderately stale data. Good balance for most workloads |
| Higher | Higher throughput by reducing training idle time, but data may be generated under an older policy |

::: tip
Start with `--max-staleness 1` and increase if you observe the training process frequently waiting for fresh rollout data. Monitor training loss stability — if loss becomes unstable with higher staleness, reduce the value.
:::

See [Fully Async Training](./fully-async-training.md) for the complete setup guide.

---

## Recommended Configurations

### Qwen3-4B on 8 GPUs (Colocate)

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216 \
--sglang-mem-fraction-static 0.8 \
--colocate
```

### Qwen3-30B-A3B on 16 GPUs (Fully Async)

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--expert-model-parallel-size 2 \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--optimizer-cpu-offload \
--sglang-mem-fraction-static 0.6 \
--fully-async
```

---

## Next Steps

- [OOM Troubleshooting](./oom-troubleshooting.md) — when tuning causes memory issues
- [Configuration Reference](./configuration.md) — full parameter list
- [Debugging Guide](./debugging.md) — isolating training and inference issues
