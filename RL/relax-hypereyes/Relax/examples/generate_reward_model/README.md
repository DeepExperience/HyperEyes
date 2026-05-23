# Generative Reward Model (GenRM) Examples

Training examples that use a **Generative Reward Model** (GenRM) — an LLM-as-judge approach — to score rollout responses, replacing traditional trained reward models.

## Overview

GenRM leverages a pre-trained LLM (e.g. Qwen3-VL-30B-A3B-Instruct) to evaluate whether model responses are consistent with ground-truth labels. Instead of training a separate reward model, GenRM performs inference-time evaluation via an SGLang engine deployed as an independent Ray Serve service.

Key benefits:

- **Zero reward-model training** — uses an off-the-shelf LLM directly
- **Strong generalization** — leverages LLM reasoning for evaluation on unseen tasks
- **Flexible criteria** — evaluation behavior is controlled via prompt templates

Both scripts in this directory train **Qwen3-4B** with **GRPO** on the `dapo-math-17k` dataset, using GenRM (`--rm-type dapo-genrm`) for reward scoring and AIME-2024 for evaluation.

## Scripts

| Script                            | Mode            | Description                                        |
| :-------------------------------- | :-------------- | :------------------------------------------------- |
| `run-qwen3-4B-8xgpu-colocated.sh` | Colocate (sync) | Actor & Rollout share GPUs; GenRM on separate GPUs |
| `run-qwen3-4B-8xgpu-async.sh`     | Fully Async     | Independent GPU pools per role; maximum throughput |

### Resource Layout

**Colocate mode** (`--colocate`):

```
Actor (training):  8 GPU (colocated with rollout)
Rollout:           4 GPU (time-shared with actor)
GenRM:             4 GPU
```

**Async mode** (`--fully-async`):

```
Actor (training):  2 GPU
Rollout:           3 GPU
Reference:         1 GPU
Actor Forward:     1 GPU
GenRM:             1 GPU
```

## Quick Start

### Prerequisites

1. **Model weights** — Download Qwen3-4B (policy) and Qwen3-VL-30B-A3B-Instruct (GenRM judge):

   ```bash
   # Place under exps/ (or set EXP_DIR / MODEL_DIR)
   exps/Qwen3-4B/
   exps/Qwen3-VL-30B-A3B-Instruct/
   ```

2. **Dataset** — `dapo-math-17k` and `aime-2024` for evaluation:

   ```bash
   exps/dapo-math-17k/dapo-math-17k.jsonl
   exps/aime-2024/aime-2024.jsonl
   ```

3. **Ray cluster** — A running Ray cluster reachable at `http://127.0.0.1:8265`.

### Run Training

```bash
# Colocate mode (memory-efficient, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh

# Fully async mode (higher throughput, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-async.sh
```

## Key Parameters

| Parameter                     | Default                      | Description                            |
| :---------------------------- | :--------------------------- | :------------------------------------- |
| `--rm-type dapo-genrm`        | —                            | Use DAPO-GenRM reward function         |
| `--genrm-model-path`          | —                            | Path to the GenRM judge model          |
| `--genrm-num-gpus-per-engine` | 1 or 4                       | GPUs allocated per GenRM SGLang engine |
| `--genrm-engine-config`       | `{"max_context_len": 10240}` | SGLang engine configuration            |
| `--genrm-sampling-config`     | `{"temperature": 0.1, ...}`  | Sampling params for the judge          |
| `--max-staleness`             | 0 (coloc) / 2 (async)        | Max data staleness for async training  |

## File Structure

```
examples/generate_reward_model/
├── README.md                            # This document
├── run-qwen3-4B-8xgpu-colocated.sh     # Colocate mode launch script
└── run-qwen3-4B-8xgpu-async.sh         # Fully async mode launch script
```

## Further Reading

- [GenRM Example](../../docs/en/examples/generative-reward-model.md) — Full GenRM architecture, configuration, and script walkthrough
- [Fully Async Training](../../docs/en/guide/fully-async-training.md) — How async mode works in Relax
- [Configuration Guide](../../docs/en/guide/configuration.md) — Complete parameter reference
