# Debugging Guide

A practical guide to debugging accuracy issues and isolating training/inference components in Relax.

## Accuracy Debugging

### 1. Check Rollout Responses

Search for `Finish rollout` in the logs to locate the current rollout ID and the generated responses. First, determine whether the responses are coherent and sensible.

- **If garbled output appears from the very first step**, this usually indicates a checkpoint loading error or model conversion issue. A thorough method is to save all parameters inside the model's SGLang `load_weights` implementation and compare them against the loaded checkpoint. If all parameter updates are correct but the issue persists, it may be caused by special buffers in SGLang being released during the `release_memory_occupation` phase. If you are testing with a pretrain model, try switching to the instruct version of the same architecture to check whether the garbled output is specific to pretrain models.

- **If the first step produces normal responses but later steps degrade**, the training has diverged. You need to carefully inspect each step's reward computation, hyperparameter settings, and other factors.

### 2. Verify log_probs and ref_log_probs

Check the rollout stats printed at the first step. Verify that `log_probs` and `ref_log_probs` are exactly equal (i.e., KL = 0 at step 1) and the values are small.

- **If they are not exactly equal**, this is typically caused by non-deterministic kernels in Transformer Engine. For example, in some versions of TE, Megatron requires `--attention-backend flash` to force Flash Attention usage and avoid numerical instability of fused attention under Context Parallelism (CP).

- **If the values are large** (e.g., > 1), there are generally two possibilities:
  - Very large values usually indicate a training configuration error.
  - Values only slightly higher than the SFT loss baseline (e.g., logprob around 0.8 for an instruct model) may mean the data doesn't match the training chat template, or doesn't match the cold-start distribution.

### 3. Verify KL and grad_norm at Step 1

With one-step-per-rollout (`num_steps_per_rollout == 1`), check whether KL is 0 and `grad_norm` is small at step 1.

Issues at this stage are typically Megatron / Transformer Engine related bugs. For example:

- MoE models require `--moe-permute-fusion` to be enabled.

## Isolated Debugging

Relax supports running the training and inference components independently, which allows:

- Debugging the inference pipeline with minimal GPU resources.
- Debugging the training pipeline with fixed inputs, eliminating rollout randomness.

### Available Debug Flags

The following CLI arguments enable isolated debugging:

| Flag | Description |
|---|---|
| `--debug-rollout-only` | Only initialize SGLang (skip Megatron). Use for inference debugging. |
| `--debug-train-only` | Only initialize Megatron (skip SGLang). Use for training debugging. |
| `--save-debug-rollout-data <path>` | Save rollout results to the specified path for later replay. |
| `--load-debug-rollout-data <path>` | Load rollout data from the specified path. Automatically sets `--debug-train-only`. |
| `--dump-details <dir>` | Dump all training details (automatically enables rollout data saving). |

### Workflow 1: Debug Inference Only

Use `--debug-rollout-only` to skip Megatron initialization entirely. Only SGLang engines will be started, allowing you to test inference with fewer GPUs.

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    # ... other rollout args
```

You can combine this with `--save-debug-rollout-data` to capture rollout results:

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... other rollout args
```

### Workflow 2: Debug Training Only

Use `--load-debug-rollout-data` to load pre-saved rollout data and run only the training pipeline. This automatically sets `debug_train_only=True`, so SGLang will not be initialized.

```bash
python3 relax/entrypoints/train.py \
    --load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... other training args
```

This approach is especially useful for:

- Tuning parallelism configurations (TP, PP, EP, CP) without waiting for rollout.
- Reproducing and fixing training-specific issues with deterministic inputs.
- Iterating quickly on loss computation or optimizer changes.

### Workflow 3: Full Detail Dump

Use `--dump-details` to save all training details for post-hoc analysis. When set, it automatically enables:

- `--save-debug-rollout-data` at `<dir>/rollout_data/{rollout_id}.pt`
- `--save-debug-train-data` at `<dir>/train_data/{rollout_id}_{rank}.pt`

```bash
python3 relax/entrypoints/train.py \
    --dump-details /path/to/dump/dir \
    # ... other args
```

::: tip
`--dump-details` is also useful for collecting data for bug reports — it captures everything needed to reproduce an issue.
:::
