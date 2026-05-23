---
name: verl-to-relax
description: Migrate RL training recipes from verl to Relax framework.
  Use when user wants to port reward functions, tool environments, training
  scripts, or any recipe code from the verl (volcengine/verl) codebase to
  Relax. Handles reward, rollout, tool/env, dataset, and launch script
  conversion. Supports both colocate (default) and fully async deployment modes.
---

# verl → Relax Recipe Migration

This skill guides migration of RL training recipes (reward functions, tool environments, multi-turn rollouts, training scripts) from the verl framework to Relax.

For detailed import/code mapping tables and transformation templates, see `references/migration_mapping.md`.

______________________________________________________________________

## Migration overview

A verl recipe typically consists of:

| verl Component | verl Location | Relax Equivalent | Relax Location |
| --- | --- | --- | --- |
| Reward function (`compute_score`) | `verl/utils/reward_score/<dataset>.py` or custom file | Async `reward_func(args, sample)` | `examples/<algo>/reward_<algo>.py` via `--custom-rm-path` |
| Tool class (`BaseTool`) | `verl/tools/<tool>.py` | `BaseInteractionEnv` subclass | `examples/<algo>/env_<algo>.py` |
| Multi-turn config YAML | `examples/sglang_multiturn/config/` | Custom config YAML | `examples/<algo>/<algo>_config.yaml` |
| Training launch script | `examples/<recipe>/run_*.sh` | Shell script (`python3 relax/entrypoints/train.py`) | `examples/<algo>/run_<algo>.sh` |
| Dataset class | `verl/utils/dataset/rl_dataset.py` or custom | Parquet + CLI args | `--prompt-data`, `--input-key`, etc. |
| Hydra YAML config | `verl/trainer/config/ppo_trainer.yaml` | CLI argparse flags | `relax/entrypoints/train.py` args |
| RewardManager | `verl/workers/reward_manager/naive.py` | `RewardExecutor` + custom-rm-path | `relax/engine/rewards/` |

> **Reward 两层机制说明**：Relax 的 reward 系统分为两层。
> 1. **内置 reward**（`relax/engine/rewards/`）：通过 `--rm-type deepscaler|math|dapo|...` 直接使用，无需写 Python 代码。如果 verl 的 `compute_score` 恰好等价于某个内置类型（如简单数学答案校验），可直接使用 `--rm-type` 而不必迁移代码。
> 2. **自定义 reward**（`--custom-rm-path`）：当 `--custom-rm-path` 被设置时，`RewardExecutor` 会优先加载用户函数，跳过内置分发。verl 的 `compute_score` 通常包含算法特定的打分逻辑，属于自定义范畴，因此迁移目标是 `examples/<algo>/reward_<algo>.py`，通过 `--custom-rm-path examples.<algo>.reward_<algo>.reward_func` 注册。

The algorithm code lives under `examples/<algo>/` in Relax — not inside the framework core.

______________________________________________________________________

## Core architecture differences

### 1. Configuration paradigm

| Aspect | verl | Relax |
| --- | --- | --- |
| Config system | Hydra (YAML-based, `@hydra.main`) | CLI argparse + optional YAML for custom configs |
| Config override | `key.subkey=value` (dot notation) | `--key-subkey value` (dash notation) |
| Entry point | `python3 -m verl.trainer.main_ppo` | `python3 relax/entrypoints/train.py` (after `scripts/entrypoint/local.sh` starts Ray) |
| Config composition | `defaults` list in YAML | `source scripts/models/<model>.sh` |

### 2. Data protocol

| Aspect | verl | Relax |
| --- | --- | --- |
| Core data type | `DataProto` (TensorDict + non_tensor_batch) | `Sample` dataclass |
| Tensor data | `data.batch["prompts"]`, `data.batch["responses"]` | `sample.tokens`, `sample.rollout_tokens` |
| Text data | Decoded from token IDs in RewardManager | `sample.prompt`, `sample.response` (strings) |
| Ground truth | `data.non_tensor_batch["reward_model"]["ground_truth"]` | `sample.label` (via `--label-key label`; preprocess verl data to extract `ground_truth` into a flat `label` column) |
| Data source | `data.non_tensor_batch["data_source"]` | `sample.metadata["data_source"]` (via `--metadata-key`) |
| Extra info | `data.non_tensor_batch["extra_info"]` | `sample.metadata` |
| Multimodal | `data.non_tensor_batch["multi_modal_data"]` | `sample.multimodal_inputs` |

### 3. Reward system

| Aspect | verl | Relax |
| --- | --- | --- |
| Reward entry | `compute_score(data_source, solution_str, ground_truth, extra_info)` | `async def reward_func(args, sample, **kwargs)` (single-sample) or `async def reward_func(args, samples, **kwargs)` (batch, with `--group-rm`) |
| Return type | `float` or `dict` with `"score"` key | `float` or `dict` with `"score"` key (single); `list[float]` or `list[dict]` (batch). When returning dict, add `--reward-key score` |
| Registration | `custom_reward_function.path` + `custom_reward_function.name` in Hydra | `--custom-rm-path module.path.reward_func` |
| Batch mode | `BatchRewardManager` / `DAPORewardManager` | `--group-rm` flag → `reward_func(args, samples: list[Sample])` |
| Manager class | `NaiveRewardManager` / `BatchRewardManager` / `DAPORewardManager` | `RewardExecutor` (built-in) |
| Execution | Synchronous, in main process or ThreadPool | Async, Ray remote workers for CPU-intensive |

### 4. Rollout / multi-turn

| Aspect | verl | Relax |
| --- | --- | --- |
| Multi-turn config | `actor_rollout_ref.rollout.multi_turn.enable=True` | `--custom-generate-function-path` |
| Tool definition | `BaseTool` class + YAML tool schema | `BaseInteractionEnv` subclass + `build_env()` factory |
| Tool registry | YAML `tools` list with `class_name` | Python module path in config YAML |
| Turn control | `max_assistant_turns` in rollout config | `max_turns` in custom config YAML |

______________________________________________________________________

## Workflow

### Step 0: Create the target directory

```bash
mkdir -p examples/<algo>
touch examples/<algo>/__init__.py
```

### Step 1: Migrate reward function

**This is the most critical step.** verl and Relax have different reward function interfaces.

#### verl pattern (function-based, synchronous, routed by data_source)

```python
# verl: standalone function, dispatched by data_source string
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Called by NaiveRewardManager for each sample.
    Args:
        data_source: str - dataset identifier (e.g. "openai/gsm8k")
        solution_str: str - model's decoded response text
        ground_truth: str - ground truth answer
        extra_info: dict - additional metadata
    Returns:
        float or dict with "score" key
    """
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math"]:
        return math_dapo.compute_score(solution_str, ground_truth)
    ...
```

Registered via Hydra config:
```yaml
custom_reward_function:
  path: /path/to/my_reward.py
  name: compute_score
  reward_kwargs:
    key1: value1
```

#### Relax pattern (function-based, async, per-sample)

```python
# Relax: async function, operates on Sample dataclass
from relax.utils.types import Sample

def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """Synchronous single-sample scoring. Must return dict with 'score' key."""
    ...
    return {"score": final_score, "acc": ..., ...}

async def reward_func(args, sample: Sample, **kwargs):
    """Entry point called by Relax engine. Wraps compute_score."""
    ground_truth = sample.label
    return compute_score(sample.response, ground_truth, extra_info=sample.metadata)
```

Registered via CLI:
```bash
--custom-rm-path examples.<algo>.reward_<algo>.reward_func
```

**Key conversion rules:**

1. **Remove `data_source` dispatch** — verl routes rewards by `data_source` string; in Relax, each example has its own reward module, so the dispatch is unnecessary. Extract the specific scoring logic for your dataset.
2. **Wrap in async `reward_func`** — Add `async def reward_func(args, sample: Sample, **kwargs)` as entry point. For batch/group reward, use `async def reward_func(args, samples: list[Sample], **kwargs)` and add `--group-rm` to CLI.
3. **Map data fields** — `solution_str` → `sample.response`, `ground_truth` → `sample.label` (preprocess verl parquet to extract `ground_truth` into a flat `label` column; see Step 4), `extra_info` → `sample.metadata`.
4. **Return dict with `"score"`** — Both frameworks support returning a dict; ensure the `"score"` key is present (batch mode returns `list[dict]`). When returning dict, add `--reward-key score` to CLI so Relax can extract the float value via `sample.reward[args.reward_key]`. Alternatively, return a plain `float` (no `--reward-key` needed).
5. **Remove verl imports** — Replace `from verl.utils.reward_score import ...` with direct imports of the scoring logic, or copy the relevant scoring functions.
6. **Handle `reward_kwargs`** — In verl, extra kwargs are passed via `custom_reward_function.reward_kwargs`; in Relax, create a YAML file and pass via `--custom-config-path path/to/config.yaml`. All keys are set as `args` attributes via `setattr(args, k, v)`, accessible as `args.key1` in your reward function.

### Step 2: Migrate tool environment (if multi-turn/agentic)

Only needed for **multi-turn** or **tool-calling** recipes. Skip for pure single-turn reward-only recipes.

#### verl pattern (`BaseTool`)

```python
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

class MyTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id=None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory."""
        return str(uuid4()), ToolResponse()

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute tool and return (response, step_reward, metrics)."""
        result = do_something(parameters)
        return ToolResponse(text=result), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward based on tool state."""
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        """Release tool instance."""
        pass
```

Registered via YAML:
```yaml
tools:
  - class_name: "verl.tools.my_tool.MyTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "my_tool"
        description: "Tool description"
        parameters: {...}
```

#### Relax pattern (`BaseInteractionEnv`)

```python
from examples.<algo>.base_env import BaseInteractionEnv
from relax.utils.types import Sample

class MyAgentEnv(BaseInteractionEnv):
    def __init__(self, *, max_turns, image=None):
        self.max_turns = max_turns
        self.image = image
        self.turn = 0

    def reset(self):
        """Return (observation, info). No arguments — sample data passed via build_env()."""
        self.turn = 0
        return {"obs_str": "Initial prompt", "role": "user"}, {}

    def step(self, response_text: str):
        """Parse tool calls from response, execute, return (obs_dict, done, info)."""
        self.turn += 1
        tool_result = self._execute_tool(response_text)
        done = self.turn >= self.max_turns
        obs = {
            "obs_str": f"<tool_response>{tool_result}</tool_response>",
            "role": "user",
        }
        return obs, done, {"tool_result": tool_result}

    def close(self):
        pass

def build_env(sample: Sample = None, args=None, **_) -> MyAgentEnv:
    """Factory function, required by Relax rollout."""
    max_turns = args.max_turns if args else 5
    image = None
    if sample and sample.multimodal_inputs:
        images = sample.multimodal_inputs.get("images") or sample.multimodal_inputs.get("image")
        if images:
            image = images[0]
    return MyAgentEnv(max_turns=max_turns, image=image)
```

**Key conversion rules:**

1. **`BaseTool` → `BaseInteractionEnv`** — verl tools are stateless async services with `create/execute/calc_reward/release`; Relax envs are stateful objects with `reset()/step()/close()`.
2. **Tool schema** — verl uses OpenAI function tool schema in YAML; Relax handles tool parsing in the env's `step()` method.
3. **Step reward** — verl returns `(ToolResponse, step_reward, metrics)` from `execute`; in Relax, step reward is handled separately (in the reward function or env info dict).
4. **Instance management** — verl uses `instance_id` for lifecycle management; Relax instantiates one env per sample via `build_env()`.
5. **Observation format** — verl returns `ToolResponse(text=...)` objects; Relax returns dicts `{"obs_str": text, "role": "user", "multi_modal_data": {...}}`.
6. **Copy `base_env.py`** — from `examples/deepeyes/base_env.py` or import `BaseInteractionEnv` from there.
7. **Create config YAML** — with `max_turns` and `rollout_interaction_env_path`.

### Step 3: Migrate rollout (if multi-turn/agentic)

For multi-turn/agentic recipes, the multi-turn rollout logic lives in a `generate()` function.

**verl approach:** Multi-turn is handled internally by the rollout worker with `multi_turn.enable=True` in config. Tools are registered via YAML and executed automatically.

**Relax approach:** Multi-turn is handled by a custom `generate()` function specified via `--custom-generate-function-path`.

**Recommendation:** Copy `examples/deepeyes/rollout.py` into your example directory and update `DEFAULT_ENV_MODULE` to point to your env module:

```python
DEFAULT_ENV_MODULE = "examples.<algo>.env_<algo>"
```

Then configure in the launch script:

```bash
--custom-generate-function-path examples.<algo>.rollout.generate
```

And in the config YAML (loaded via `--custom-config-path`):

```yaml
max_turns: 5
rollout_interaction_env_path: examples.<algo>.env_<algo>
```

This ensures each example is **self-contained** — no cross-example dependencies.

Only modify the rollout further if your algorithm has custom turn logic (e.g., parallel tool execution, custom stopping conditions, special token budget management).

### Step 4: Migrate dataset handling

#### verl pattern

verl uses Parquet files with specific columns, loaded by a dataset class:

```python
# Data columns in parquet:
# - "prompt": chat messages (list of dicts or string)
# - "reward_model.ground_truth": ground truth for reward computation
# - "data_source": dataset identifier for reward routing
# - "extra_info": additional metadata dict
# - "images": (optional) image data for multimodal

# Hydra config:
data:
  train_files: /path/to/train.parquet
  val_files: /path/to/test.parquet
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024
```

#### Relax pattern

Relax also uses Parquet files but specifies column mapping via CLI:

```bash
ROLLOUT_ARGS=(
    --prompt-data "['/path/to/train.parquet']"
    --input-key prompt              # column containing chat messages
    --label-key label               # column containing ground truth (plain string)
    --metadata-key extra_info       # column containing metadata
    --multimodal-keys '{"image":"images"}'  # multimodal column mapping
    --apply-chat-template           # apply chat template to prompts
)
```

**Data preprocessing for verl parquet:**

verl parquet files are **not directly compatible** with Relax. You must write a conversion script (typically `scripts/tools/process_<algo>.py`) and **mention it in the run script header** so users know to run it first. Key transformations:

1. **`reward_model` → `label`**: verl stores ground truth in a `reward_model` dict column (e.g., `{"style": "rule", "ground_truth": "72"}`), but Relax expects `sample.label` to be a plain string. Extract it into a flat `label` column.
2. **Image data**: If the dataset is multimodal, preserve the image column (e.g., extract raw bytes from `preprocessed_images`). Then set `--multimodal-keys '{"image":"<column_name>"}'` in the launch script.
3. **`extra_info`**: Preserve the `extra_info` column if it exists; map via `--metadata-key extra_info`.

```python
# scripts/tools/process_<algo>.py — conversion script template
import pandas as pd

def convert_row(row):
    result = {
        "prompt": row["prompt"],  # keep chat format as-is
        "label": row["reward_model"]["ground_truth"],
    }
    # Preserve images for multimodal datasets
    if "preprocessed_images" in row:
        result["image"] = [img["bytes"] for img in row["preprocessed_images"]]
    # Preserve metadata
    if "extra_info" in row:
        result["extra_info"] = row["extra_info"]
    return result

df = pd.read_parquet("verl_data/train.parquet")
df_out = pd.DataFrame([convert_row(row) for _, row in df.iterrows()])
df_out.to_parquet("relax_data/train.parquet", index=False)
```

Then add a **data conversion reminder** in the run script header:

```bash
# Prerequisites:
#   1. Convert data:  python3 scripts/tools/process_<algo>.py \
#                       --input-dir /path/to/verl/data.parquet \
#                       --output-dir /path/to/relax/data.parquet
#   2. Set env vars:  MODEL_DIR=/path/to/models  DATA_DIR=/path/to/data
#   3. Run:           bash examples/<algo>/run_<algo>.sh
```

**Key conversion rules:**

1. `data.train_files` → `--prompt-data "[...]"` (wrap in JSON list)
2. `data.val_files` → `--eval-prompt-data <name> <files...>`
3. `data.train_batch_size` → `--global-batch-size`
4. `data.max_prompt_length` → `--rollout-max-prompt-len`
5. `data.max_response_length` → `--rollout-max-response-len`
6. Column mapping: use `--input-key`, `--label-key`, `--metadata-key`, `--multimodal-keys`
7. **Preprocess verl parquet** — extract `reward_model["ground_truth"]` into a flat `label` column; `--label-key` reads the column value as-is into `sample.label`, so it should be a plain string, not a dict.
8. If verl uses a custom dataset class (`data.custom_cls`), extract the data preprocessing logic and apply it offline to the Parquet files before loading in Relax.

### Step 5: Migrate training launch script

#### verl pattern

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
```

#### Relax pattern

Relax run scripts rely on a two-layer environment setup:

| Variable | Set by | Purpose |
| --- | --- | --- |
| `MODEL_CONFIG_DIR` | Entrypoint (`local.sh` or external) | Path to `scripts/models/`, contains model architecture configs |
| `MODEL_ARGS` | Model config shell (e.g. `qwen3-8B.sh`) | Architecture flags (hidden size, layers, TP/PP defaults) |
| `MODEL_DIR` | User | Directory containing HF model checkpoints |
| `DATA_DIR` | User | Directory containing preprocessed Parquet data |
| `SAVE_DIR` | User (optional) | Checkpoint save directory |

The generated script should **always** support both Colocate (sync) and Fully Async modes via a `MODE` parameter, defaulting to **sync** (colocate). This way the user can switch between modes without rewriting the script:

```bash
#!/bin/bash
# Usage: bash examples/<algo>/run_<algo>.sh [sync|async]

set -ex
set -o pipefail

MODE=${1:-${MODE:-"sync"}}    # Arg $1 > env $MODE > default "sync"

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint.
# local.sh sets MODEL_CONFIG_DIR, RUNTIME_ENV_JSON, PYTHONPATH, and starts Ray.
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/<model>.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/<algo>}"
EXP_NAME="<model>-<algo>-fully-${MODE}-${TIMESTAMP}"

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/<Model>
    --ref-load ${MODEL_DIR}/<Model>
    # --load ${MODEL_DIR}/<Model>_mcore/     # for resuming
    # --save ${MODEL_DIR}/<Model>_mcore/
    # --save-interval 4
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --metadata-key extra_info
    --multimodal-keys '{"image":"image"}'   # if multimodal; omit for text-only
    --reward-key score
    --apply-chat-template
    --custom-rm-path examples.<algo>.reward_<algo>.reward_func
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 1024
    --rollout-max-prompt-len 512
    --rollout-temperature 1
    # global-batch-size MUST equal rollout-batch-size × n-samples-per-prompt
    --global-batch-size 256
    --rollout-shuffle
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 4     # TP × PP must divide actor GPU count
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size 1
    --max-tokens-per-gpu 9216          # dynamic batch memory cap
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.8
)

LOG_ARGS=(
    --use-tensorboard
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

EVAL_ARGS=(
    --eval-interval 100
    --eval-prompt-data <name> ${TEST_FILES}
)

#=============================================================================
# Launch: fully async or colocate (sync)
#=============================================================================
mkdir -p logs

if [ "${MODE}" = "async" ]; then
    # Fully Async: actor/rollout/reference/actor_fwd/advantages on separate GPUs.
    # 8 GPU example: actor=4, rollout=2, reference=1, actor_fwd=1, advantages=CPU
    python3 relax/entrypoints/train.py \
        --resource '{"actor": [1, 4], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
        --max-staleness 3 \
        --num-data-storage-units 1 \
        --num-iters-per-train-update 8 \
        --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}' \
        --fully-async \
        --use-health-check \
        "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${LOG_ARGS[@]}" \
        "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
        2>&1 | tee logs/${EXP_NAME}.log
else
    # Colocate (sync): actor and rollout share the same GPUs.
    python3 relax/entrypoints/train.py \
        --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
        --max-staleness 1 \
        --num-data-storage-units 1 \
        --colocate \
        --use-health-check \
        --balance-data \
        "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${LOG_ARGS[@]}" \
        "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
        2>&1 | tee logs/${EXP_NAME}.log
fi
```

#### Colocate vs Fully Async: key differences

The script template above supports both modes. Here is what changes between them:

| Aspect | Colocate / sync (default) | Fully Async |
| --- | --- | --- |
| Resource | `--resource '{"actor": [1, N], "rollout": [1, N]}'` | `--resource '{"actor": [1, A], "rollout": [1, R], "reference": [1, Ref], "actor_fwd": [1, AF], "advantages": [1, 0]}'` |
| Mode flag | `--colocate` | `--fully-async` |
| Staleness | `--max-staleness 1` (strict on-policy) | `--max-staleness 3` (recommended 2-3) |
| Training iters | Not needed | `--num-iters-per-train-update 8` (train 8 epochs per rollout batch) |
| Ref/ActorFwd config | Not needed (computed inside Actor) | `--ref-actor-config '{...}'` (separate lightweight services) |
| TP constraint | `TP × PP` divides total GPU count | `TP × PP` divides **Actor** GPU count |

> verl does not have a direct equivalent of fully async mode. verl uses a colocated architecture. No verl config maps to `--fully-async`.

#### GPU resource allocation for fully async

```
8 GPU example:
├── Actor (training):     4 GPU  (TP=4, PP=1, DP=1)
├── Rollout (inference):  2 GPU  (SGLang engines)
├── Reference (forward):  1 GPU  (TP=1 via ref-actor-config)
├── ActorFwd (forward):   1 GPU  (TP=1 via ref-actor-config)
└── Advantages (compute): 0 GPU  (CPU only)

16 GPU large model example:
├── Actor (training):     8 GPU  (TP=4, PP=2, DP=1)
├── Rollout (inference):  4 GPU  (SGLang engines)
├── Reference (forward):  2 GPU  (TP=2 via ref-actor-config)
├── ActorFwd (forward):   2 GPU  (TP=2 via ref-actor-config)
└── Advantages (compute): 0 GPU  (CPU only)
```

`--ref-actor-config` overrides parallelism for Reference and ActorFwd (typically single GPU, TP=1):

```bash
--ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}'
```

#### Key conversion rules

1. `python3 -m verl.trainer.main_ppo` → `python3 relax/entrypoints/train.py` (both sync and async; `local.sh` already starts Ray via `ray start --head`, so no `ray job submit` needed for single-node; do NOT add `--runtime-env-json`)
2. Hydra dot-notation `key.subkey=value` → argparse `--key-subkey value`
3. `actor_rollout_ref.model.path` → `--hf-checkpoint` + `--ref-load`
4. `actor_rollout_ref.rollout.n` → `--n-samples-per-prompt`
5. `data.train_batch_size` → `--global-batch-size`
6. `trainer.n_gpus_per_node` / `trainer.nnodes` → `--resource` (for sync: `{"actor": [1, N], "rollout": [1, N]}` where `N = n_gpus_per_node × nnodes`; for async: split across roles)
7. `trainer.save_freq` → `--save-interval`
8. `trainer.test_freq` → `--eval-interval`
9. `trainer.total_epochs` → `--num-epoch N` (preferred; maps directly to verl's epoch concept) or `--num-rollout` (rollout batch count)
10. `algorithm.adv_estimator=grpo` → `--advantage-estimator grpo`
11. Model config: use `source "${MODEL_CONFIG_DIR}/<model>.sh"` instead of inline TP/PP settings
12. Checkpoint: verl auto-handles model loading from HF path; Relax uses `--hf-checkpoint` for initial, `--load` for resume

#### Notes on specific parameters

**On-policy constraint (critical):**

`--global-batch-size` MUST equal `--rollout-batch-size × --n-samples-per-prompt` to ensure on-policy training. If this constraint is violated, training degrades to off-policy. For example: `--rollout-batch-size 32 --n-samples-per-prompt 8` → `--global-batch-size 256`. In fully async mode, each batch is additionally trained for `--num-iters-per-train-update` iterations, improving data utilization.

**Default optimizer options:**

Always include these three optimizer flags unless there is a specific reason not to:

```bash
--optimizer-cpu-offload              # offload optimizer state to CPU memory
--overlap-cpu-optimizer-d2h-h2d      # overlap D2H/H2D transfers with computation
--use-precision-aware-optimizer      # mixed-precision optimizer for memory efficiency
```

**Multimodal datasets:**

If the verl dataset contains image data, you MUST:
1. Preserve image columns in the data conversion script (extract raw bytes from `preprocessed_images` or keep `images` as-is)
2. Add `--multimodal-keys '{"<relax_key>":"<column_name>"}'` to `ROLLOUT_ARGS` (e.g., `--multimodal-keys '{"image":"image"}'`)

**Fully async specific:**

- `--num-iters-per-train-update` — training epochs per rollout batch; higher values (4-8) improve data efficiency. Especially important in async mode where rollout data generation is continuous.
- `--max-staleness` — controls how far Rollout can run ahead of Actor. Value of 3 means up to 2 unconsumed rollout batches in TransferQueue. Recommended 2-3 for production.
- `--max-tokens-per-gpu` — dynamic batching memory limit; recommended 9216 for 9B models.
- `TP × PP` must divide Actor GPU count — e.g., Actor 4 GPUs with PP=1 → TP can be 1/2/4; Actor 8 GPUs with TP=4, PP=2 → DP=1. Constraint: `Actor_GPUs = TP × PP × DP`.

______________________________________________________________________

## Argument mapping quick reference

| verl Argument (Hydra) | Relax Argument (CLI) |
| --- | --- |
| `algorithm.adv_estimator=grpo` | `--advantage-estimator grpo` |
| `data.train_files=path` | `--prompt-data "[path]"` |
| `data.val_files=path` | `--eval-prompt-data name path` |
| `data.train_batch_size=N` | `--global-batch-size N` |
| `data.max_prompt_length=N` | `--rollout-max-prompt-len N` |
| `data.max_response_length=N` | `--rollout-max-response-len N` |
| `actor_rollout_ref.model.path=P` | `--hf-checkpoint P` + `--ref-load P` |
| `actor_rollout_ref.actor.optim.lr=V` | `--lr V` |
| `actor_rollout_ref.actor.use_kl_loss=True` | `--use-kl-loss --kl-loss-coef 0.001` |
| `actor_rollout_ref.actor.kl_loss_coef=V` | `--kl-loss-coef V` (loss-based KL penalty) |
| `actor_rollout_ref.actor.kl_loss_type=T` | `--kl-loss-type T` (valid: `k1`/`k2`/`k3`/`low_var_kl`; verl `kl` ≈ Relax `k1`) |
| `algorithm.use_kl_in_reward=True` | `--kl-coef V` (reward-shaping KL; note: only one of `--kl-coef` / `--kl-loss-coef` can be non-zero) |
| `algorithm.kl_ctrl.kl_coef=V` | `--kl-coef V` |
| `algorithm.gamma=V` | `--gamma V` |
| `algorithm.lam=V` | `--lambd V` |
| `actor_rollout_ref.actor.entropy_coeff=V` | `--entropy-coef V` |
| `actor_rollout_ref.actor.ppo_mini_batch_size=N` | `--global-batch-size N` (note: verl's is global total, not per-GPU; Relax `--micro-batch-size` is per-GPU gradient accumulation) |
| `actor_rollout_ref.rollout.name=sglang` | (Relax uses SGLang by default) |
| `actor_rollout_ref.rollout.gpu_memory_utilization=V` | `--sglang-mem-fraction-static V` |
| `actor_rollout_ref.rollout.n=N` | `--n-samples-per-prompt N` |
| `actor_rollout_ref.rollout.temperature=V` | `--rollout-temperature V` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size=N` | `--tensor-model-parallel-size N` |
| `actor_rollout_ref.rollout.multi_turn.enable=True` | `--custom-generate-function-path` (specify rollout module) |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=N` | `max_turns: N` in custom config YAML |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path=P` | `--custom-config-path` (env config, not tool YAML) |
| `actor_rollout_ref.ref.fsdp_config.param_offload=True` | (handled automatically in Relax) |
| `trainer.n_gpus_per_node=N` (single-node) | `--resource '{"actor": [1, N], "rollout": [1, N]}'` (second element = total GPUs) |
| `trainer.nnodes=M, n_gpus_per_node=N` | `--resource '{"actor": [1, N*M], "rollout": [1, N*M]}'` (first element is ignored; second = total GPUs = nnodes × n_gpus_per_node) |
| `trainer.save_freq=N` | `--save-interval N` |
| `trainer.test_freq=N` | `--eval-interval N` |
| `trainer.total_epochs=N` | `--num-epoch N` (preferred) or `--num-rollout` (rollout batch count) |
| `trainer.project_name=S` | `--tb-project-name S` |
| `trainer.experiment_name=S` | `--tb-experiment-name S` |
| `trainer.logger=["console","wandb"]` | `--use-wandb` / `--use-clearml` |
| `custom_reward_function.path=P` + `name=N` | `--custom-rm-path module.path.function_name` |

### Fully Async specific parameters (no verl equivalent)

These parameters are Relax-only and have no verl counterpart. They are used in the async branch of the `MODE` switch in Step 5:

| Relax Argument | Default | Description |
| --- | --- | --- |
| `--fully-async` | `false` | Enable fully async training pipeline |
| `--colocate` | `false` | Enable colocate (sync) mode (default migration target) |
| `--max-staleness N` | `1` | Max rollout-ahead steps (1=on-policy, 2-3 recommended for async) |
| `--num-iters-per-train-update N` | `1` | Training epochs per rollout batch (4-8 for async) |
| `--num-data-storage-units N` | `1` | TransferQueue storage actor count |
| `--ref-actor-config '{...}'` | — | JSON config overrides for Reference/ActorFwd services |
| `--use-health-check` | `false` | Enable fault-tolerance health monitoring |
| `--balance-data` | `false` | Balance data across DP ranks (colocate only) |
| `--max-tokens-per-gpu N` | — | Dynamic batching memory cap per GPU |
| `--clip-grad V` | — | Gradient clipping norm |

______________________________________________________________________

## Recipe type decision tree

```
Is the verl recipe single-turn (no tools/multi-turn)?
├── YES → Migrate: Step 1 (reward) + Step 4 (data) + Step 5 (launch script)
│   └── Target structure:
│       examples/<algo>/
│       ├── __init__.py
│       ├── reward_<algo>.py
│       └── run_<algo>.sh           # MODE=${1:-"sync"}, supports both sync & async
│
└── NO (multi-turn / tool-calling)
    ├── Does it use verl BaseTool? → Step 2 (migrate tool → BaseInteractionEnv)
    ├── Does it use custom multi-turn logic? → Step 3 (migrate rollout)
    └── All recipes → Step 1 + Step 4 + Step 5
    └── Target structure:
        examples/<algo>/
        ├── __init__.py
        ├── base_env.py             # Copy from examples/deepeyes/base_env.py
        ├── env_<algo>.py           # BaseInteractionEnv subclass + build_env()
        ├── reward_<algo>.py        # compute_score + reward_func
        ├── rollout.py              # Copy from examples/deepeyes/rollout.py, update DEFAULT_ENV_MODULE
        ├── <algo>_config.yaml      # max_turns, rollout_interaction_env_path
        └── run_<algo>.sh           # MODE=${1:-"sync"}, supports both sync & async

Note: Step 5 always generates a dual-mode script (MODE=${1:-"sync"}).
The user switches to fully async by passing "async" — no separate migration step needed.
```

______________________________________________________________________

## Important rules

- **ALWAYS** create a new `examples/<algo>/` directory; never modify `relax/engine/rewards/`
- **ALWAYS** provide an `async def reward_func(args, sample: Sample, **kwargs)` entry point (single-sample mode, default) or `async def reward_func(args, samples: list[Sample], **kwargs)` (batch mode, with `--group-rm`)
- **ALWAYS** return a dict with a `"score"` key from the reward function (or `list[dict]` in batch mode); add `--reward-key score` to CLI when returning dict
- **ALWAYS** use `Sample` dataclass fields (`sample.response`, `sample.label`, `sample.metadata`)
- **ALWAYS** write a data conversion script (`scripts/tools/process_<algo>.py`) and reference it in the run script header; verl parquet is NOT directly compatible with Relax; When doing so, You need to tell the user that they need to perform data conversion in advance.
- **ALWAYS** preserve image columns in multimodal datasets; add `--multimodal-keys` to the launch script
- **ALWAYS** ensure `--global-batch-size = --rollout-batch-size × --n-samples-per-prompt` for on-policy training
- **ALWAYS** include `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` in optimizer config
- **NEVER** use verl imports (`from verl...`) in Relax code
- **NEVER** use Hydra config syntax in Relax launch scripts
- **NEVER** modify Relax core code (`relax/`) for recipe migration — keep everything in `examples/`
- **NEVER** add `--runtime-env-json` — this is handled by the entrypoint layer
- **ALWAYS** copy scoring logic from verl's `reward_score/` rather than importing it
- **PREFER** `--custom-rm-path` for reward registration over modifying `RewardExecutor`
- **PREFER** `--custom-config-path` YAML for passing extra reward config (replaces verl's `reward_kwargs`)
- **ALWAYS** generate dual-mode scripts with `MODE=${1:-${MODE:-"sync"}}` — supports positional arg, env var, and default; colocate (sync) by default
- **ALWAYS** use `python3 relax/entrypoints/train.py` directly for both modes — `local.sh` already starts Ray via `ray start --head`, no `ray job submit` needed for single-node
- **ALWAYS** ensure `TP × PP` divides Actor GPU count in fully async mode (constraint: `Actor_GPUs = TP × PP × DP`)
- **ALWAYS** add `--ref-actor-config` for Reference/ActorFwd in the async branch of the launch script

## LLM-as-Judge reward migration

If the verl recipe uses an LLM-based reward (e.g., via OpenAI API), migrate as follows:

**verl** typically calls OpenAI API synchronously or via a custom reward function. **Relax** reward functions are `async`, making it natural to use `httpx.AsyncClient` or `openai.AsyncOpenAI`:

```python
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
            {"role": "system", "content": "Rate the following answer..."},
            {"role": "user", "content": f"Question: {sample.prompt}\nAnswer: {sample.response}"},
        ],
        "max_tokens": 64,
        "temperature": 0,
    })
    result = resp.json()
    score_text = result["choices"][0]["message"]["content"]
    score = float(score_text.strip()) / 10.0  # normalize
    return {"score": score}
```

Pass `judge_url` and `judge_model` via `--custom-config-path`:
```yaml
# examples/<algo>/<algo>_config.yaml
judge_url: "http://localhost:8000/v1/chat/completions"
judge_model: "Qwen/Qwen3-8B"
```

## References

- `references/migration_mapping.md` - Detailed import mapping table, data field mapping, and code transformation patterns
- verl documentation: https://verl.readthedocs.io/en/latest/
- verl reward function guide: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
