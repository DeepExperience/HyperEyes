---
name: redaccel-to-relax
description: Migrate RL training algorithms from RedAccel to Relax framework.
  Use when user wants to port reward functions, agent environments, training
  scripts, or any algorithm code from the RedAccel (redaccelrl) codebase to
  Relax. Handles reward, environment, rollout, and launch script conversion.
---

# RedAccel → Relax Algorithm Migration

This skill guides migration of RL training algorithms (reward functions, agent environments, training scripts) from the RedAccel framework to Relax.

For mapping tables and code templates, see `references/migration_mapping.md`.

______________________________________________________________________

## Migration overview

A RedAccel algorithm typically consists of:

| RedAccel Component | RedAccel Location | Relax Equivalent | Relax Location |
| --- | --- | --- | --- |
| Reward class (GRPO/Group) | `aipet_rl/reward_*.py` | Async `reward_func(args, sample)` | `examples/<algo>/reward_<algo>.py` |
| Agent / Tool env | `aipet_rl/agent/*.py` (`ToolBase`) | `BaseInteractionEnv` subclass | `examples/<algo>/env_<algo>.py` |
| LLM-as-Judge helpers | `aipet_rl/llm_judge.py` | Inlined or standalone module | `examples/<algo>/llm_judge.py` |
| Training launch script | `exps/*.sh` (`redaccel-cli`) | Shell script (`ray job submit`) | `examples/<algo>/run_<algo>.sh` |
| Search / external tools | `aipet_rl/agent/unified_search_async.py` | Same module, imported from example dir | `examples/<algo>/search_tools.py` |

The algorithm code lives under `examples/<algo>/` in Relax — not inside the framework core.

______________________________________________________________________

## Workflow

### Step 0: Create the target directory

```bash
mkdir -p examples/<algo>
touch examples/<algo>/__init__.py
```

### Step 1: Migrate reward function

**This is the most critical step.** RedAccel and Relax have fundamentally different reward interfaces.

#### RedAccel pattern (class-based, synchronous)

```python
# RedAccel: registry-decorated class, __call__ returns List[dict]
from redaccel.verl.rewards.group.base import GroupRewards, group_rewards_registry
# or
from redaccel.verl.rewards.std.base import GRPORewards, rewards_registry

@group_rewards_registry.register()  # or @rewards_registry.register()
class MyRewardClass(GroupRewards):  # or GRPORewards
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_step = 0

    def __call__(self, prompts, completions, solutions, **kwargs) -> List[dict]:
        # Access: kwargs.get("extra_info")
        # Returns: [(score, info_dict), ...]
```

#### Relax pattern (function-based, async)

```python
# Relax: standalone async function, operates on Sample dataclass
from relax.utils.types import Sample

def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """Synchronous single-sample scoring. Must return dict with 'score' key."""
    ...
    return {"score": final_score, "acc": ..., ...}

async def reward_func(args, sample: Sample, **kwargs):
    """Entry point called by Relax engine. Wraps compute_score."""
    question = sample.metadata.get("question")
    ground_truth = sample.metadata.get("answer")  # or sample.label
    return compute_score(sample.response, ground_truth, extra_info={"question": question})
```

**Key conversion rules:**

1. Remove `@group_rewards_registry.register()` / `@rewards_registry.register()` decorators
2. Remove class inheritance, extract `__call__` body into `compute_score()` function
3. Add `async def reward_func(args, sample: Sample, **kwargs)` entry point
4. Map `completions[i]` → `sample.response`, `solutions[i]` → `sample.label`, `kwargs["extra_info"]` → `sample.metadata`
5. Replace `from redaccel.verl.rewards.utils import ...` with direct OpenAI/httpx calls or Relax utilities
6. If the reward uses `local_step` for staged training, pass it via `sample.metadata` or `args`

### Step 2: Migrate agent environment (if applicable)

Only needed for **agentic** algorithms (e.g., tool-calling agents). Skip for pure chat/completion rewards.

#### RedAccel pattern (`ToolBase`)

```python
from redaccel.verl.agent.tool_envs import ToolBase

class VideoSearchTools(ToolBase):
    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs): ...
    def execute(self, action_string, **kwargs) -> tuple: ...
```

#### Relax pattern (`BaseInteractionEnv`)

```python
from examples.<algo>.base_env import BaseInteractionEnv
from relax.utils.types import Sample

class MyAgentEnv(BaseInteractionEnv):
    def __init__(self, *, max_turns, image=None): ...
    def reset(self):
        """Return (observation, info). No arguments — sample is passed via build_env()."""
    def step(self, response_text: str):
        """Return (observation, done: bool, info: dict)."""
    def close(self): ...

def build_env(sample: Sample = None, args=None, **_) -> MyAgentEnv:
    """Factory function, required by Relax rollout."""
```

**Key conversion rules:**

1. `reset(raw_prompt, multi_modal_data, ...)` → `__init__` + `reset()` (no args)
2. `execute(action_string)` → `step(response_text)` returning `(obs_dict, done, info)`
3. Observation format: return `{"obs_str": text, "role": "user", "multi_modal_data": {"image": [img]}}` instead of raw chatml messages
4. The `build_env(sample, args)` factory receives the `Sample` object and extracts images from `sample.multimodal_inputs`
5. Copy `base_env.py` from deepeyes example (or import `BaseInteractionEnv` from there)
6. Create `<algo>_config.yaml` with `max_turns` and `rollout_interaction_env_path`

### Step 3: Migrate rollout (if agentic)

For agentic algorithms, the multi-turn rollout logic lives in a `generate()` function.

**Recommendation:** Copy `examples/deepeyes/rollout.py` and update `DEFAULT_ENV_MODULE` to point to your env module. The rollout is already generic and handles multi-turn conversation, multimodal data, and budget management.

```python
DEFAULT_ENV_MODULE = "examples.<algo>.env_<algo>"
```

Only modify the rollout if your algorithm has custom turn logic (e.g., parallel tool execution, custom stopping).

### Step 4: Migrate LLM-as-Judge helpers

RedAccel uses `redaccel.verl.rewards.utils.call_gemini_flash`. In Relax, use direct OpenAI-compatible API calls:

```python
import os
from openai import OpenAI

def _get_judge_client():
    api_key = os.environ.get("DEEPEYES_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("DEEPEYES_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "gpt-4o")
    return client, model

def call_judge(prompt: str, timeout: int = 120) -> str:
    client, model = _get_judge_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()
```

### Step 5: Migrate training launch script

#### RedAccel pattern

```bash
source ${DIR}/env.sh "$@"
export PYTHONPATH=$DIR/../:$PYTHONPATH
redaccel-cli rlx setup <(echo '...')

cmd=(
    trainer.plugin_dir=$DIR/../aipet_rl
    reward_model.reward_name=VideoAgentChatScoreV7
    ...
)
redaccel-cli train "${cmd[@]}"
```

#### Relax pattern

```bash
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH=/root/Megatron-LM/:${PROJECT_ROOT}:$PYTHONPATH

source "${SCRIPT_DIR}/../../scripts/models/<model_config>.sh"

ROLLOUT_ARGS=(
    --custom-rm-path examples.<algo>.reward_<algo>.reward_func
    --custom-generate-function-path examples.<algo>.rollout.generate   # if agentic
    --custom-config-path examples/<algo>/<algo>_config.yaml            # if agentic
    --prompt-data "${PROMPT_SET}"
    ...
)

ray job submit --address="http://127.0.0.1:8265" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" "${ROLLOUT_ARGS[@]}" ...
```

**Key conversion rules:**

1. `redaccel-cli train` → `ray job submit -- python3 relax/entrypoints/train.py`
2. `trainer.plugin_dir` + `reward_model.reward_name` → `--custom-rm-path examples.<algo>.reward.reward_func`
3. `redaccel-cli rlx setup` → remove (Relax handles this internally)
4. Hydra-style `key=value` args → argparse `--key value` args
5. Model config: use `source scripts/models/<model>.sh` instead of inline TP/PP/EP settings
6. Megatron checkpoint loading: `--hf-checkpoint` / `--load` / `--ref-load` instead of `DIST_CKPT_PATH`

______________________________________________________________________

## Argument mapping quick reference

| RedAccel Argument | Relax Argument |
| --- | --- |
| `reward_model.reward_name=ClassName` | `--custom-rm-path module.path.reward_func` |
| `trainer.plugin_dir=...` | N/A (use module path in `--custom-rm-path`) |
| `actor_rollout_ref.rollout.response_length=N` | `--rollout-max-response-len N` |
| `data.train_files=[...]` | `--prompt-data "[...]"` |
| `data.val_files=[...]` | `--eval-prompt-data name files...` |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size=N` | `--tensor-model-parallel-size N` |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=N` | `--pipeline-model-parallel-size N` |
| `algorithm.kl_penalty=kl` | `--kl-loss-coef 0.01 --kl-loss-type low_var_kl` |
| `BATCH_SIZE=N` | `--global-batch-size N` |
| `MICRO_BATCH_SIZE=N` | `--micro-batch-size N` |

______________________________________________________________________

## Important rules

- **ALWAYS** create a new `examples/<algo>/` directory; never modify `relax/engine/rewards/`
- **ALWAYS** provide an `async def reward_func(args, sample, **kwargs)` entry point
- **ALWAYS** return a dict with a `"score"` key from the reward function
- **ALWAYS** use `Sample` dataclass fields (`sample.response`, `sample.label`, `sample.metadata`)
- **NEVER** use RedAccel imports (`from redaccel.verl...`) in Relax code
- **NEVER** use `redaccel-cli` in Relax launch scripts
- **NEVER** modify Relax core code (`relax/`) for algorithm migration — keep everything in `examples/`

## References

- `references/migration_mapping.md` - Detailed import mapping table and code transformation patterns
