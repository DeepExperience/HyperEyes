# verl → Relax: Detailed Migration Mapping

## Table of contents

1. [Import mapping](#import-mapping)
2. [Reward function transformation](#reward-function-transformation)
3. [Tool / environment transformation](#tool--environment-transformation)
4. [Data field mapping](#data-field-mapping)
5. [Hydra config → CLI args transformation](#hydra-config--cli-args-transformation)
6. [Launch script transformation](#launch-script-transformation)
7. [Example: GSM8K single-turn GRPO](#example-gsm8k-single-turn-grpo)
8. [Example: Multi-turn tool-calling agent](#example-multi-turn-tool-calling-agent)
9. [LLM-as-Judge reward migration](#llm-as-judge-reward-migration)

______________________________________________________________________

## Import mapping

| verl Import | Relax Replacement |
| --- | --- |
| `from verl import DataProto` | `from relax.utils.types import Sample` |
| `from verl.utils.reward_score import default_compute_score` | Copy scoring logic directly into `reward_<algo>.py` |
| `from verl.utils.reward_score.gsm8k import compute_score` | Copy function body; remove verl dependency |
| `from verl.utils.reward_score.math_reward import compute_score` | Copy function body; remove verl dependency |
| `from verl.utils.reward_score.math_dapo import compute_score` | Copy function body; remove verl dependency |
| `from verl.tools.base_tool import BaseTool` | `from examples.<algo>.base_env import BaseInteractionEnv` |
| `from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse` | Remove; use plain dicts in Relax env |
| `from verl.workers.reward_manager.naive import NaiveRewardManager` | Remove; use `--custom-rm-path` |
| `from verl.workers.reward_manager.abstract import AbstractRewardManager` | Remove; not needed in Relax |
| `from verl.trainer.main_ppo import main` | `relax/entrypoints/train.py` (CLI) |
| `from verl.utils.dataset.rl_dataset import RLHFDataset` | Parquet files + CLI args |
| `from verl.utils.import_utils import load_extern_object` | Not needed; Relax loads via module path |
| `from omegaconf import DictConfig, OmegaConf` | Not needed; use `argparse.Namespace` |

______________________________________________________________________

## Reward function transformation

### Pattern A: Single data source reward

A typical verl reward function handles one specific dataset:

**Before (verl):**

```python
# verl/utils/reward_score/gsm8k.py
import re

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):
    """Extract answer from #### pattern and compare with ground truth."""
    # Extract answer
    if "####" in solution_str:
        extracted = solution_str.split("####")[-1].strip()
        if extracted == ground_truth:
            return score
        return format_score
    return 0.0
```

Registered via Hydra config or used through `default_compute_score`:
```python
# In NaiveRewardManager.__call__:
score = self.compute_score(
    data_source=data_source,
    solution_str=response_str,
    ground_truth=ground_truth,
    extra_info=extra_info,
)
```

**After (Relax):**

```python
# examples/gsm8k_grpo/reward_gsm8k.py
import re
from relax.utils.types import Sample


def compute_score(predict_str: str, ground_truth: str) -> dict:
    """Extract answer from #### pattern and compare with ground truth."""
    if "####" in predict_str:
        extracted = predict_str.split("####")[-1].strip()
        if extracted == ground_truth:
            return {"score": 1.0, "format_correct": True, "answer_correct": True}
        return {"score": 0.1, "format_correct": True, "answer_correct": False}
    return {"score": 0.0, "format_correct": False, "answer_correct": False}


async def reward_func(args, sample: Sample, **kwargs):
    """Entry point called by Relax engine."""
    ground_truth = sample.label
    return compute_score(sample.response, ground_truth)
```

### Pattern B: Multi-source dispatch reward

verl's `default_compute_score` dispatches to different functions based on `data_source`:

**Before (verl):**

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math"]:
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps"]:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    return float(res) if not isinstance(res, dict) else res
```

**After (Relax):**

In Relax, each recipe lives in its own directory with its own reward. You don't need multi-source dispatch:

```python
# examples/math_grpo/reward_math.py
from relax.utils.types import Sample

# Copy the specific math scoring logic directly
def compute_score(predict_str: str, ground_truth: str) -> dict:
    """Math-specific scoring logic (copied from verl/utils/reward_score/math_dapo.py)."""
    # ... extracted scoring logic ...
    return {"score": score, "is_correct": is_correct}


async def reward_func(args, sample: Sample, **kwargs):
    ground_truth = sample.label
    return compute_score(sample.response, ground_truth)
```

If you truly need multi-source dispatch (mixed datasets), use `sample.metadata`:

```python
async def reward_func(args, sample: Sample, **kwargs):
    data_source = sample.metadata.get("data_source", "default")
    ground_truth = sample.label

    if data_source == "openai/gsm8k":
        return compute_score_gsm8k(sample.response, ground_truth)
    elif data_source in ["math_dapo", "math"]:
        return compute_score_math(sample.response, ground_truth)
    else:
        return {"score": 0.0}
```

### Pattern C: Custom reward function with reward_kwargs

verl passes extra configuration via `custom_reward_function.reward_kwargs`:

**Before (verl):**

```yaml
custom_reward_function:
  path: /path/to/my_reward.py
  name: compute_score
  reward_kwargs:
    format_score: 0.1
    use_strict: true
```

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  format_score=0.1, use_strict=True):
    ...
```

**After (Relax):**

Use `--custom-config-path` to pass a YAML config. Relax loads the YAML and sets all key-value pairs as attributes on `args` via `setattr(args, k, v)`:

```yaml
# examples/<algo>/<algo>_config.yaml
format_score: 0.1
use_strict: true
```

```bash
--custom-config-path examples/<algo>/<algo>_config.yaml
```

```python
# examples/<algo>/reward_<algo>.py
from relax.utils.types import Sample

def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1, use_strict: bool = True) -> dict:
    ...
    return {"score": score}

async def reward_func(args, sample: Sample, **kwargs):
    format_score = getattr(args, "format_score", 0.1)
    use_strict = getattr(args, "use_strict", True)
    return compute_score(sample.response, sample.label, format_score=format_score, use_strict=use_strict)
```

### Pattern D: Group/batch reward (verl BatchRewardManager → Relax --group-rm)

verl's `BatchRewardManager` or `DAPORewardManager` processes rewards for a group of samples simultaneously (e.g., for majority voting, group-level normalization).

**Before (verl):**

```python
# verl: BatchRewardManager processes all samples for a prompt together
class DAPORewardManager(NaiveRewardManager):
    def __call__(self, data: DataProto, ...):
        # Processes all n samples for each prompt as a group
        for prompt_group in grouped_by_prompt:
            scores = [compute_score(...) for sample in prompt_group]
            # Group-level normalization, majority voting, etc.
```

**After (Relax):**

Add `--group-rm` to CLI and change reward function to accept `samples: list[Sample]`:

```bash
--group-rm
--custom-rm-path examples.<algo>.reward_<algo>.reward_func
```

```python
# examples/<algo>/reward_<algo>.py
from relax.utils.types import Sample

async def reward_func(args, samples: list[Sample], **kwargs):
    """Batch reward: receives all n samples for one prompt as a list."""
    scores = []
    for sample in samples:
        score = compute_score(sample.response, sample.label)
        scores.append(score)

    # Group-level normalization, majority voting, etc.
    correct_count = sum(1 for s in scores if s["score"] > 0)
    for s in scores:
        s["majority_ratio"] = correct_count / len(scores)

    return scores  # list[dict], one per sample
```

______________________________________________________________________

## Tool / environment transformation

### Interface mapping

| verl (`BaseTool`) | Relax (`BaseInteractionEnv`) |
| --- | --- |
| `__init__(config, tool_schema)` | `__init__(*, max_turns, image=None, ...)` |
| `async create(instance_id) → (id, ToolResponse)` | `reset() → (obs_dict, info)` |
| `async execute(instance_id, parameters) → (ToolResponse, reward, metrics)` | `step(response_text) → (obs_dict, done, info)` |
| `async calc_reward(instance_id) → float` | Handled in separate `reward_func` |
| `async release(instance_id)` | `close()` |
| Schema in YAML tool config | Tool parsing in `step()` method |
| `ToolResponse(text=..., image=..., video=...)` | `{"obs_str": text, "role": "user", "multi_modal_data": {...}}` |
| Returns step reward in `execute()` | Step reward via env info dict or separate reward |

### Tool schema conversion

**verl YAML:**

```yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating reward"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer"
          required: ["answer"]
```

**Relax equivalent:**

```python
# In env_<algo>.py - tool parsing is done in step()
import json
import re

class MyAgentEnv(BaseInteractionEnv):
    def step(self, response_text: str):
        # Parse tool calls from model output
        tool_calls = self._parse_tool_calls(response_text)
        
        for tool_call in tool_calls:
            if tool_call["name"] == "calc_gsm8k_reward":
                answer = tool_call["arguments"].get("answer", "")
                result = self._check_answer(answer)
                obs = {"obs_str": f"Result: {result}", "role": "tool"}
                return obs, True, {"answer": answer}
        
        # No valid tool call
        return {"obs_str": "Please use the tool to submit your answer.", "role": "user"}, False, {}
    
    def _parse_tool_calls(self, text):
        """Parse function call from model output."""
        # Match common function call patterns
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match in matches:
            try:
                call = json.loads(match)
                calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls
```

### Factory function

**verl:** Tool instances are managed internally by the rollout worker:

```python
# Automatic lifecycle: create → execute (multiple turns) → calc_reward → release
```

**Relax:** You must provide a `build_env()` factory:

```python
def build_env(sample: Sample = None, args=None, **_) -> MyAgentEnv:
    """Factory function, required by Relax rollout."""
    max_turns = getattr(args, 'max_turns', 5)
    image = None
    if sample and sample.multimodal_inputs:
        images = sample.multimodal_inputs.get("images") or sample.multimodal_inputs.get("image")
        if images:
            image = images[0]
    return MyAgentEnv(max_turns=max_turns, image=image)
```

### Config YAML

**verl multi-turn config (in main Hydra config):**

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      max_assistant_turns: 5
      tool_config_path: "path/to/tool_config.yaml"
```

**Relax custom config YAML:**

```yaml
# examples/<algo>/<algo>_config.yaml
max_turns: 5
rollout_interaction_env_path: examples.<algo>.env_<algo>
```

______________________________________________________________________

## Data field mapping

### verl `DataProto` → Relax `Sample` field mapping

| verl Access Pattern | Relax Equivalent | Description |
| --- | --- | --- |
| `data.batch["prompts"]` (tensor) | `sample.prompt` (str/list) | Prompt tokens vs text |
| `data.batch["responses"]` (tensor) | `sample.response` (str) | Response tokens vs text |
| `data.batch["attention_mask"]` | `sample.loss_mask` | Attention/loss masking |
| `data.non_tensor_batch["reward_model"]["ground_truth"]` | `sample.label` | Ground truth string |
| `data.non_tensor_batch["data_source"]` | `sample.metadata["data_source"]` | Dataset identifier |
| `data.non_tensor_batch["extra_info"]` | `sample.metadata` | Extra metadata dict |
| `data.non_tensor_batch["multi_modal_data"]` | `sample.multimodal_inputs` | Multimodal data |
| `data.non_tensor_batch.get("__num_turns__")` | (via env info in rollout) | Multi-turn count |
| `data.non_tensor_batch.get("reward_scores")` | (via reward_func return) | Step rewards |
| `data.meta_info` | `args` (Namespace) | Config/meta information |
| `len(data)` | N/A (per-sample in Relax) | Batch size |

### Parquet column mapping

| verl Parquet Column | Relax CLI Arg | Relax `Sample` Field | Notes |
| --- | --- | --- | --- |
| `prompt` | `--input-key prompt` | `sample.prompt` | Direct mapping |
| `reward_model` (dict with `ground_truth`) | `--label-key label` | `sample.label` | **Preprocess**: extract `ground_truth` into a flat `label` column (see below) |
| `data_source` | N/A (route by example dir) | `sample.metadata["data_source"]` | Use `--metadata-key` if needed |
| `extra_info` | `--metadata-key extra_info` | `sample.metadata` | Direct mapping |
| `images` | `--multimodal-keys '{"image":"images"}'` | `sample.multimodal_inputs["images"]` | Direct mapping |
| `videos` | `--multimodal-keys '{"video":"videos"}'` | `sample.multimodal_inputs["videos"]` | Direct mapping |

**Parquet preprocessing for verl data:**

verl stores ground truth as `reward_model: {"style": "rule", "ground_truth": "72"}`, but Relax's `--label-key` reads the column value as-is into `sample.label`. Preprocess to extract `ground_truth` into a flat column:

```python
import pandas as pd

df = pd.read_parquet("verl_data/train.parquet")
# Extract ground_truth from reward_model dict
df["label"] = df["reward_model"].apply(
    lambda x: x.get("ground_truth", "") if isinstance(x, dict) else str(x)
)
# Optionally, also extract data_source into metadata-friendly format
if "extra_info" not in df.columns:
    df["extra_info"] = df.apply(
        lambda row: {"data_source": row.get("data_source", "")}, axis=1
    )
df.to_parquet("relax_data/train.parquet", index=False)
```

______________________________________________________________________

## Hydra config → CLI args transformation

### Data config

```yaml
# verl (Hydra)                           # Relax (CLI)
data:
  train_files: /path/train.parquet        # --prompt-data "['/path/train.parquet']"
  val_files: /path/test.parquet           # --eval-prompt-data gsm8k /path/test.parquet
  train_batch_size: 1024                  # --global-batch-size 1024
  max_prompt_length: 512                  # --rollout-max-prompt-len 512
  max_response_length: 1024              # --rollout-max-response-len 1024
  filter_overlong_prompts: True           # (default behavior in Relax)
  return_raw_chat: True                   # --apply-chat-template
  shuffle: True                           # --rollout-shuffle
```

### Actor / model config

```yaml
# verl (Hydra)                                    # Relax (CLI)
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-8B                           # --hf-checkpoint Qwen/Qwen3-8B
                                                   # --ref-load Qwen/Qwen3-8B
    use_remove_padding: True                       # (handled by Megatron backend)
    enable_gradient_checkpointing: True            # --recompute-granularity full
    trust_remote_code: True                        # (default in Relax)
  actor:
    optim:
      lr: 1e-6                                     # --lr 1e-6
    use_kl_loss: True                              # --use-kl-loss --kl-loss-coef 0.001
    kl_loss_coef: 0.001                            # --kl-loss-coef 0.001
    kl_loss_type: low_var_kl                       # --kl-loss-type low_var_kl
    entropy_coeff: 0                               # --entropy-coef 0
    ppo_mini_batch_size: 256                       # --global-batch-size 256 (note: verl's is global total)
    ppo_micro_batch_size_per_gpu: 32               # --micro-batch-size 32 (per-GPU)
    fsdp_config:
      param_offload: False                         # (N/A — Relax uses Megatron backend, not FSDP)
      optimizer_offload: False                     # --optimizer-cpu-offload (Megatron flag; for large models)
    strategy: fsdp                                 # (Relax uses Megatron by default)
```

### Rollout config

```yaml
# verl (Hydra)                                    # Relax (CLI)
actor_rollout_ref:
  rollout:
    name: sglang                                   # (SGLang is default in Relax)
    gpu_memory_utilization: 0.6                    # --sglang-mem-fraction-static 0.6
    n: 5                                           # --n-samples-per-prompt 5
    temperature: 1.0                               # --rollout-temperature 1.0
    tensor_model_parallel_size: 2                  # --tensor-model-parallel-size 2
    mode: async                                    # (async is default in Relax)
    multi_turn:
      enable: True                                 # --custom-generate-function-path examples.<algo>.rollout.generate
      max_assistant_turns: 5                       # max_turns: 5 (in <algo>_config.yaml)
      tool_config_path: path/to/tool.yaml          # --custom-config-path examples/<algo>/<algo>_config.yaml
```

### Algorithm config

```yaml
# verl (Hydra)                                    # Relax (CLI)
algorithm:
  adv_estimator: grpo                              # --advantage-estimator grpo
  gamma: 1.0                                       # --gamma 1.0
  lam: 1.0                                         # --lambd 1.0
  use_kl_in_reward: False                          # --kl-coef 0 (reward shaping; 0 = disabled)
  kl_penalty: kl                                   # --kl-loss-type k1 (verl "kl" ≈ Relax "k1"; valid: k1/k2/k3/low_var_kl)
  kl_ctrl:
    type: fixed                                    # (Relax uses fixed by default)
    kl_coef: 0.001                                 # --kl-coef 0.001 (reward shaping, NOT --kl-loss-coef)
```

### Trainer config

```yaml
# verl (Hydra)                                    # Relax (CLI)
trainer:
  total_epochs: 15                                 # --num-epoch 15 (preferred) or --num-rollout N
  n_gpus_per_node: 8                               # --resource '{"actor": [1, 8], "rollout": [1, 8]}'
  nnodes: 1                                        #   ↑ format: [ignored, total_gpus]; total_gpus = n_gpus_per_node × nnodes
  save_freq: 20                                    # --save-interval 20
  test_freq: 5                                     # --eval-interval 5
  project_name: my_project                         # --tb-project-name my_project
  experiment_name: my_exp                          # --tb-experiment-name my_exp
  logger: ["console","wandb"]                      # --use-wandb (or --use-clearml)
  critic_warmup: 0                                 # (not needed for GRPO)
  val_before_train: True                           # (default in Relax)
```

### Reward config

```yaml
# verl (Hydra)                                    # Relax (CLI)
# Option 1: Built-in reward functions
reward:
  reward_manager:
    name: naive                                    # --rm-type <type> (deepscaler, dapo, etc.)

# Option 2: Custom reward function
custom_reward_function:
  path: /path/to/my_reward.py                     # --custom-rm-path examples.<algo>.reward.reward_func
  name: compute_score                              # (function name is part of the module path)
  reward_kwargs:                                   # --custom-config-path <yaml> (keys become args attrs)
    key: value
```

______________________________________________________________________

## Launch script transformation

### Environment setup

```bash
# verl
# No specific env setup needed; Hydra handles config

# Relax
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/<model>.sh"
```

### Model checkpoint

```bash
# verl
# Model path is specified in Hydra config:
# actor_rollout_ref.model.path=Qwen/Qwen3-8B

# Relax
CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-8B
    --ref-load ${MODEL_DIR}/Qwen3-8B
    --save ${SAVE_DIR}/Qwen3-8B-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
    # --load ${SAVE_DIR}/resume-checkpoint   # for resuming
)
```

### Launch command

```bash
# verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=... \
    ...

# Relax (single-node)
# local.sh already calls `ray start --head`, so just run python3 directly.
python3 relax/entrypoints/train.py \
    "${PERF_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log

# Relax (multi-node: submit via Ray dashboard when HEAD_IP is a remote node)
# ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HEAD_IP}:8265" \
#     -- python3 relax/entrypoints/train.py ...
```

______________________________________________________________________

## Example: GSM8K single-turn GRPO

Migrating the standard verl GSM8K GRPO example (single-turn, rule-based reward).

### Source (verl)

- Script: `examples/grpo_trainer/run_qwen3-8b.sh`
- Reward: `verl/utils/reward_score/gsm8k.py` (via `default_compute_score`)
- Config: inline Hydra args

### Target structure (Relax)

```
examples/gsm8k_grpo/
├── __init__.py
├── reward_gsm8k.py          # compute_score + reward_func
└── run_gsm8k_grpo.sh        # Launch script
```

### reward_gsm8k.py

```python
"""GSM8K rule-based reward function for Relax.

Ported from verl/utils/reward_score/gsm8k.py.
"""

import re

from relax.utils.types import Sample


def _extract_answer(text: str) -> str | None:
    """Extract numeric answer after #### marker."""
    if "####" not in text:
        return None
    answer = text.split("####")[-1].strip()
    # Clean up: remove commas, dollar signs, etc.
    answer = answer.replace(",", "").replace("$", "").strip()
    return answer


def compute_score(predict_str: str, ground_truth: str) -> dict:
    """GSM8K scoring: check if extracted answer matches ground truth.

    Returns:
        dict with "score" key: 1.0 if correct, 0.1 if format ok but wrong, 0.0 otherwise.
    """
    extracted = _extract_answer(predict_str)
    if extracted is None:
        return {"score": 0.0, "format_correct": False, "answer_correct": False}

    # Normalize ground truth
    gt = ground_truth.replace(",", "").replace("$", "").strip()

    if extracted == gt:
        return {"score": 1.0, "format_correct": True, "answer_correct": True}
    return {"score": 0.1, "format_correct": True, "answer_correct": False}


async def reward_func(args, sample: Sample, **kwargs):
    """Entry point called by Relax engine.

    Expects sample.label to be a plain ground truth string.
    Preprocess verl parquet to extract ground_truth into a flat 'label' column.
    """
    ground_truth = sample.label or ""
    return compute_score(sample.response, ground_truth)
```

### run_gsm8k_grpo.sh

```bash
#!/bin/bash
set -ex
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
# NOTE: Create scripts/models/qwen3-8B.sh with MODEL_ARGS for your model arch.
# See scripts/models/qwen3-4B.sh as a template.
source "${MODEL_CONFIG_DIR}/qwen3-8B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/gsm8k-grpo}"
EXP_NAME="qwen3-8b-gsm8k-grpo-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    exit 1
fi
mkdir -p ${SAVE_DIR}

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-8B
    --ref-load ${MODEL_DIR}/Qwen3-8B
    --save ${SAVE_DIR}/Qwen3-8B-GSM8K-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
)

PROMPT_SET="['${DATA_DIR}/gsm8k/train.parquet']"
TEST_FILES="${DATA_DIR}/gsm8k/test.parquet"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --custom-rm-path examples.gsm8k_grpo.reward_gsm8k.reward_func
    --reward-key score
    --num-rollout 200
    --rollout-batch-size 32
    --n-samples-per-prompt 5
    --rollout-max-response-len 1024
    --rollout-max-prompt-len 512
    --rollout-temperature 1
    --global-batch-size 256
    --apply-chat-template
    --rollout-shuffle
)

EVAL_ARGS=(
    --eval-interval 100
    --eval-prompt-data gsm8k ${TEST_FILES}
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0
    --eps-clip 0.2
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
)

SGLANG_ARGS=(
    --sglang-mem-fraction-static 0.6
)

PERF_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --max-staleness 1
    --colocate
    --tensor-model-parallel-size 2
    --sequence-parallel
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
)

LOG_ARGS=(
    --use-clearml
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

mkdir -p logs

# local.sh already starts Ray via `ray start --head`, so python3 works directly.
python3 relax/entrypoints/train.py \
    "${PERF_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
```

______________________________________________________________________

## Example: Multi-turn tool-calling agent

Migrating a verl multi-turn tool-calling recipe (e.g., search agent or calculator agent).

### Source (verl)

- Script: `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh`
- Config: `examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`
- Tool: `verl/tools/gsm8k_tool.py` + `config/tool_config/gsm8k_tool_config.yaml`
- Reward: `verl/utils/reward_score/gsm8k.py`

### Target structure (Relax)

```
examples/gsm8k_multiturn/
├── __init__.py
├── base_env.py                 # Copy from examples/deepeyes/base_env.py
├── env_gsm8k.py                # Gsm8kToolEnv(BaseInteractionEnv) + build_env()
├── reward_gsm8k.py             # compute_score + reward_func
├── rollout.py                  # Copy from examples/deepeyes/rollout.py
├── gsm8k_multiturn_config.yaml # max_turns, env module path
└── run_gsm8k_multiturn.sh      # Launch script
```

### env_gsm8k.py

```python
"""GSM8K multi-turn tool environment for Relax.

Ported from verl/tools/gsm8k_tool.py.
"""

import json
import re

from examples.gsm8k_multiturn.base_env import BaseInteractionEnv
from relax.utils.types import Sample


class Gsm8kToolEnv(BaseInteractionEnv):
    """Environment that provides a calculator tool for GSM8K problems."""

    def __init__(self, *, max_turns=5):
        self.max_turns = max_turns
        self.turn = 0
        self.ground_truth = None

    def reset(self):
        self.turn = 0
        return {}, {}

    def step(self, response_text: str):
        self.turn += 1
        done = self.turn >= self.max_turns

        # Parse tool calls from model output
        tool_calls = self._parse_tool_calls(response_text)

        if not tool_calls:
            # No tool call found - either model is done or needs guidance
            if self._has_final_answer(response_text):
                done = True
                return {}, done, {"response": response_text}
            obs = {
                "obs_str": "Please use the calculator tool to solve the problem step by step.",
                "role": "user",
            }
            return obs, done, {}

        # Execute the first tool call
        tool_call = tool_calls[0]
        result = self._execute_calculator(tool_call)
        obs = {
            "obs_str": f"<tool_response>{result}</tool_response>",
            "role": "tool",
        }
        return obs, done, {"tool_result": result}

    def close(self):
        pass

    def _parse_tool_calls(self, text):
        """Parse function calls from model output."""
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match in matches:
            try:
                calls.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        return calls

    def _execute_calculator(self, tool_call):
        """Execute calculator operation."""
        params = tool_call.get("arguments", tool_call.get("parameters", {}))
        expression = params.get("expression", params.get("answer", ""))
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception:
            return f"Error: could not evaluate '{expression}'"

    def _has_final_answer(self, text):
        """Check if response contains a final answer."""
        return "####" in text or "<answer>" in text


def build_env(sample: Sample = None, args=None, **_) -> Gsm8kToolEnv:
    """Factory function, required by Relax rollout."""
    max_turns = getattr(args, 'max_turns', 5)
    env = Gsm8kToolEnv(max_turns=max_turns)
    if sample:
        env.ground_truth = sample.label
    return env
```

### gsm8k_multiturn_config.yaml

```yaml
max_turns: 5
rollout_interaction_env_path: examples.gsm8k_multiturn.env_gsm8k
```

### rollout.py

```python
"""Multi-turn rollout for GSM8K tool-calling.

Copied from examples/deepeyes/rollout.py with updated DEFAULT_ENV_MODULE.
"""

DEFAULT_ENV_MODULE = "examples.gsm8k_multiturn.env_gsm8k"

# ... rest of rollout logic copied from examples/deepeyes/rollout.py ...
```

### run_gsm8k_multiturn.sh changes vs single-turn

The key differences in the launch script:

```bash
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --custom-rm-path examples.gsm8k_multiturn.reward_gsm8k.reward_func
    --reward-key score
    --custom-generate-function-path examples.gsm8k_multiturn.rollout.generate     # NEW: multi-turn rollout
    --custom-config-path examples/gsm8k_multiturn/gsm8k_multiturn_config.yaml     # NEW: env config
    --num-rollout 200
    --rollout-batch-size 32
    --n-samples-per-prompt 16
    --rollout-max-response-len 1024
    --rollout-max-prompt-len 1024
    --rollout-temperature 1
    --global-batch-size 256
    --apply-chat-template
)
```

______________________________________________________________________

## verl built-in reward → Relax reward type mapping

Some verl reward functions have direct Relax equivalents via `--rm-type`:

| verl `data_source` | Relax `--rm-type` | Notes |
| --- | --- | --- |
| `openai/gsm8k` | custom (`--custom-rm-path`) | GSM8K has no built-in `--rm-type`; use custom reward |
| `math_dapo`, `math` | `dapo` | DAPO math reward |
| `lighteval/MATH` | `math` | Math verification |
| N/A | `deepscaler` | DeepScaler scoring (different from GSM8K) |
| N/A | `f1` | F1 score |
| N/A | `multiple_choice` | Multiple choice |
| N/A | `remote_rm` | HTTP reward model |

For reward functions not listed above, always use `--custom-rm-path` with a custom implementation. The `--rm-type` built-in types have their own scoring logic in `relax/engine/rewards/` — only use them when the scoring behavior matches exactly.

______________________________________________________________________

## LLM-as-Judge reward migration

verl recipes that use LLM-based reward scoring (e.g., OpenAI API calls, local judge models) can be migrated to Relax's async reward function pattern.

### Before (verl)

```python
# verl: synchronous LLM judge call
import openai

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    response = client.chat.completions.create(
        model="judge-model",
        messages=[
            {"role": "system", "content": "Rate the answer 1-10."},
            {"role": "user", "content": f"Q: {ground_truth}\nA: {solution_str}"},
        ],
        max_tokens=64,
    )
    score = float(response.choices[0].message.content.strip()) / 10.0
    return {"score": score}
```

### After (Relax)

Use `httpx.AsyncClient` for async HTTP calls (preferred over sync OpenAI SDK):

```python
# examples/<algo>/reward_<algo>.py
import httpx
from relax.utils.types import Sample

_client = httpx.AsyncClient(timeout=60)

async def reward_func(args, sample: Sample, **kwargs):
    """LLM-as-Judge reward using async HTTP."""
    judge_url = getattr(args, "judge_url", "http://localhost:8000/v1/chat/completions")
    judge_model = getattr(args, "judge_model", "judge-model")

    resp = await _client.post(judge_url, json={
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "Rate the following answer 1-10."},
            {"role": "user", "content": f"Question: {sample.prompt}\nAnswer: {sample.response}"},
        ],
        "max_tokens": 64,
        "temperature": 0,
    })
    result = resp.json()
    score_text = result["choices"][0]["message"]["content"]
    score = float(score_text.strip()) / 10.0
    return {"score": score}
```

Pass judge config via `--custom-config-path`:

```yaml
# examples/<algo>/<algo>_config.yaml
judge_url: "http://localhost:8000/v1/chat/completions"
judge_model: "Qwen/Qwen3-8B"
```

```bash
--custom-config-path examples/<algo>/<algo>_config.yaml
--custom-rm-path examples.<algo>.reward_<algo>.reward_func
--reward-key score
```

**Key differences from verl:**
- Use `httpx.AsyncClient` instead of synchronous `openai.OpenAI` — Relax reward functions are `async`
- Pass judge URL/model via `--custom-config-path` YAML (sets attrs on `args`) instead of hardcoding
- Module-level `_client` for connection pooling (created once, reused across calls)
