# Generative Reward Model (GenRM) Example

Training examples that use a **Generative Reward Model** (GenRM) — an LLM-as-judge approach — to score rollout responses, replacing traditional trained reward models.

## Overview

GenRM (Generative Reward Model) leverages a pre-trained LLM (e.g., Qwen3-VL-30B-A3B-Instruct) to evaluate whether model responses are consistent with ground-truth labels. Instead of training a separate reward model, GenRM performs inference-time evaluation via an [SGLang](https://github.com/sgl-project/sglang) engine deployed as an independent Ray Serve service.

Key benefits:

- **Zero reward-model training** — uses an off-the-shelf LLM directly, no additional reward model training required
- **Strong generalization** — leverages LLM reasoning capabilities for effective evaluation on unseen tasks
- **Flexible criteria** — evaluation behavior is controlled via prompt templates

Both scripts in this example train **Qwen3-4B** with **GRPO** on the `dapo-math-17k` dataset, using GenRM (`--rm-type dapo-genrm`) for reward scoring and AIME-2024 for evaluation.

## Architecture

In the recommended **colocate mode**, all 8 GPUs are owned by the Actor (training). During the inference phase, the Actor offloads its weights and the GPUs are time-shared: 4 GPUs run the Rollout engine, and the other 4 GPUs run the GenRM engine. Once inference is complete, GenRM and Rollout offload their weights back, and all 8 GPUs are reclaimed for training. This means the GenRM GPUs are not wasted — they directly accelerate training when not evaluating.

```
                 8-GPU Colocate Mode

 ┌──────────── Placement Group (8 GPU) ────────────┐
 │                                                 │
 │  Inference phase:                               │
 │  ┌───────────────────┐   ┌───────────────────┐  │
 │  │  Rollout  (4 GPU) │──►│  GenRM   (4 GPU)  │  │
 │  │  SGLang Engine    │◄──│  SGLang Engine    │  │
 │  └───────────────────┘   └───────────────────┘  │
 │                  Score: 0 / 1                   │
 │                                                 │
 │  Training phase (offload inference weights):    │
 │  ┌─────────────────────────────────────────┐    │
 │  │         Actor  (8 GPU)                  │    │
 │  │         Megatron Training               │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘
```

All components live in the same placement group. During the inference phase, 4 GPUs run Rollout and 4 GPUs run GenRM. Rollout generates candidate responses and sends them with the ground-truth label to GenRM over HTTP; GenRM returns a binary score (1 = consistent, 0 = inconsistent). After reward computation, inference weights are offloaded and all 8 GPUs are reclaimed by the Actor for training.

## Scripts

| Script                             | Mode                    | Description                                                                    |
| :--------------------------------- | :---------------------- | :----------------------------------------------------------------------------- |
| `run-qwen3-4B-8xgpu-colocated.sh` | Colocate (recommended)  | All 8 GPUs for training; rollout & GenRM time-share via offload                |
| `run-qwen3-4B-8xgpu-async.sh`     | Fully Async             | Independent GPU pools per role; rollout & training fully overlapped             |

### Resource Layout

**Colocate mode** (`--colocate`, recommended):

```
Actor (training):  8 GPU  (all GPUs participate in training)
Rollout:           4 GPU  (time-shared with actor via offload)
GenRM:             4 GPU  (time-shared with actor via offload)
```

In this mode, the GenRM GPUs are not idle during training — they are offloaded back to the Actor for gradient computation, effectively giving training the full 8-GPU parallelism.

**Async mode** (`--fully-async`):

```
Actor (training):  2 GPU  (dedicated)
Rollout:           3 GPU  (dedicated)
Reference:         1 GPU
Actor Forward:     1 GPU
GenRM:             1 GPU  (dedicated)
```

## Quick Start

### Prerequisites

1. **Model weights** — Download Qwen3-4B (policy model) and Qwen3-VL-30B-A3B-Instruct (GenRM judge model):

   ```bash
   # Place under exps/ (or set EXP_DIR / MODEL_DIR)
   exps/Qwen3-4B/
   exps/Qwen3-VL-30B-A3B-Instruct/
   ```

2. **Dataset** — Prepare `dapo-math-17k` for training and `aime-2024` for evaluation:

   ```bash
   exps/dapo-math-17k/dapo-math-17k.jsonl
   exps/aime-2024/aime-2024.jsonl
   ```

3. **Ray cluster** — A running Ray cluster reachable at `http://127.0.0.1:8265`.

### Run Training

```bash
# Colocate mode (recommended, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh

# Fully async mode (8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-async.sh
```

### Verify Service Health

Once the training job is running, check that the GenRM service is healthy:

```bash
curl http://localhost:8000/genrm/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "genrm"
}
```

## Configuration

### GenRM-Specific CLI Arguments

| Argument                      | Type   | Default | Description                                                                         |
| :---------------------------- | :----- | :------ | :---------------------------------------------------------------------------------- |
| `--genrm-model-path`          | `str`  | `None`  | GenRM model path. Setting this enables GenRM                                        |
| `--genrm-num-gpus`            | `int`  | `1`     | Total number of GPUs for GenRM                                                      |
| `--genrm-num-gpus-per-engine` | `int`  | `1`     | Number of GPUs per GenRM engine instance                                            |
| `--genrm-engine-config`       | `JSON` | `None`  | JSON dict for engine initialization (e.g., `max_context_len`, `dp_size`, `pp_size`) |
| `--genrm-sampling-config`     | `JSON` | `None`  | JSON dict for sampling parameters                                                   |

### Engine Config Keys

| Key                | Type  | Default | Description               |
| :----------------- | :---- | :------ | :------------------------ |
| `max_context_len`  | `int` | `8192`  | Maximum context length    |
| `dp_size`          | `int` | `1`     | Data parallelism size     |
| `pp_size`          | `int` | `1`     | Pipeline parallelism size |
| `max_total_tokens` | `int` | `8192`  | Maximum total tokens      |

### Sampling Config Keys

| Key                | Type    | Default | Description                  |
| :----------------- | :------ | :------ | :--------------------------- |
| `temperature`      | `float` | `0.2`   | Sampling temperature         |
| `top_p`            | `float` | `1.0`   | Nucleus sampling probability |
| `top_k`            | `int`   | `-1`    | Top-k sampling (-1 disables) |
| `max_response_len` | `int`   | `1024`  | Maximum response length      |

### Resource Allocation

GenRM is included in the `--resource` JSON as a `"genrm"` role. The format is `[num_groups, num_gpus_per_group]`.

**Colocated mode** (recommended):

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 4 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}' \
    --colocate \
    --rm-type dapo-genrm
```

::: warning
In colocated mode, total inference GPUs (rollout + genRM) must not exceed actor GPUs. For example, on an 8-GPU machine: `--resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}'` uses all 8 GPUs for inference, which are shared with training via offload.
:::

**Fully-Async mode**:

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 1 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 2], "rollout": [1, 3], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
    --fully-async \
    --rm-type dapo-genrm
```

## Script Walkthrough

Both scripts share the same structure. Here is a breakdown of the key configuration groups:

### Reward Configuration

The critical setting that enables GenRM is `--rm-type dapo-genrm`, which routes reward computation through `async_compute_score_genrm()` in `relax/engine/rewards/dapo_genrm.py`. The core implementation is straightforward:

```python
DAPO_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. ...
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""

def _format_messages(question, ground_truth, predict_str):
    # Extract answer after "Answer:" marker, or take last 300 chars
    if "Answer:" in predict_str:
        predict_str = predict_str.split("Answer:")[-1]
    else:
        predict_str = predict_str[-300:]
    prompt = DAPO_GENRM_PROMPT_TEMPLATE.format(
        question=question, ground_truth=ground_truth, predict_str=predict_str,
    )
    return [{"role": "user", "content": prompt}]

async def async_compute_score_genrm(args, sample) -> dict:
    genrm_client = get_genrm_client()          # singleton HTTP client
    question = sample.metadata.get("question", "")
    ground_truth = sample.metadata.get("label", "")
    messages = _format_messages(question, ground_truth, sample.response)

    response = await genrm_client.generate(messages)  # call GenRM service
    prediction = response.strip()

    # Strict equality: only exact "1" yields a positive score
    score = 1.0 if prediction == "1" else 0.0
    return {"score": score, "acc": int(score), "pred": prediction}
```

```bash
ROLLOUT_ARGS=(
   --rm-type dapo-genrm        # Use GenRM for reward scoring
   --reward-key score           # Key for reward in output dict
   --n-samples-per-prompt 8     # Generate 8 responses per prompt
   --rollout-max-response-len 8192
   --rollout-temperature 1
)
```

### Training Configuration

Both scripts use GRPO with the following hyperparameters:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis                    # Truncated importance sampling
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)
```

### GenRM Service Configuration

The GenRM model and engine are configured at the `ray job submit` level:

```bash
--genrm-model-path ${MODEL_DIR}/Qwen3-VL-30B-A3B-Instruct/ \
--genrm-num-gpus-per-engine 1 \
--genrm-engine-config '{"max_context_len": 10240}' \
--genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}'
```

::: tip
A low temperature (e.g., 0.1) is recommended for GenRM to produce deterministic evaluation results. Higher temperatures introduce evaluation variance.
:::

## Usage Examples

### Call GenRM API Directly

```bash
curl -X POST http://localhost:8000/genrm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Evaluate the answer consistency..."}
    ]
  }'
```

Response:

```json
{
  "response": "1"
}
```

### Use GenRMClient in Python

```python
from relax.utils.genrm_client import get_genrm_client

# Get singleton client (avoids per-request client creation overhead)
client = get_genrm_client()

# Async generate
response = await client.generate(
    messages=[{"role": "user", "content": "Evaluate..."}],
    sampling_params={"temperature": 0.2},
)
print(response)  # "1" or "0"
```

## Best Practices

1. **Prefer colocate mode**: In colocate mode, GenRM GPUs are offloaded back to training when not evaluating, so all 8 GPUs participate in gradient computation. This gives better GPU utilization than async mode, where GenRM GPUs sit idle during training
2. **Set appropriate context length**: `max_context_len` in engine config should accommodate your longest prompt + response combination
3. **Use low sampling temperature**: A temperature of 0.1 produces deterministic evaluations; increase only if evaluation diversity is desired
4. **Monitor health**: Periodically check the `/health` endpoint to ensure GenRM engines are running properly
5. **Match GPU allocation to model size**: For large GenRM models (e.g., 30B), allocate more GPUs per engine via `--genrm-num-gpus-per-engine`

## Troubleshooting

### GenRM Not Enabled

Ensure `--genrm-model-path` is set. GenRM is only activated when this argument is not `None`.

### Resource Allocation Error in Colocated Mode

In colocated mode with GenRM, total inference GPUs (rollout + genRM) must not exceed actor GPUs:

```
rollout_num_gpus + genrm_num_gpus <= actor_total_gpus
```

Adjust the `rollout` and/or `genrm` GPU allocation in `--resource` accordingly.

### Engine Initialization Timeout

If GenRM engines fail to initialize:

1. Check that the model path is accessible from all nodes
2. Verify sufficient GPU memory is available
3. Review Ray logs for SGLang engine startup errors

### GenRM Always Returns 0

The DAPO-GenRM reward function uses strict equality to parse responses — only an exact `"1"` string yields a positive score. If the GenRM model outputs anything else (e.g., `"1."`, `"Yes"`, or multi-line text), the score will be 0. Verify that the GenRM model and prompt template produce clean `"1"` / `"0"` outputs.

## File Structure

```
examples/generate_reward_model/
├── README.md                            # Example overview
├── run-qwen3-4B-8xgpu-colocated.sh     # Colocate mode launch script
└── run-qwen3-4B-8xgpu-async.sh         # Fully async mode launch script
```

## Further Reading

- [Architecture](/en/guide/architecture) — Understand the overall Relax architecture
- [Fully Async Training](/en/guide/fully-async-training) — How async mode works in Relax
- [Configuration](/en/guide/configuration) — Complete configuration reference
- [GenRM API](/en/api/genrm) — HTTP API reference for the GenRM service
