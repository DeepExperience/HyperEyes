# RedAccel → Relax: Detailed Migration Mapping

## Table of contents

1. [Import mapping](#import-mapping)
2. [Reward function transformation](#reward-function-transformation)
3. [Agent environment transformation](#agent-environment-transformation)
4. [LLM-as-Judge replacement](#llm-as-judge-replacement)
5. [Data field mapping](#data-field-mapping)
6. [Launch script transformation](#launch-script-transformation)
7. [Example: Chat reward (sly_chat)](#example-chat-reward-sly_chat)
8. [Example: Agent reward (sly_agent)](#example-agent-reward-sly_agent)

______________________________________________________________________

## Import mapping

| RedAccel Import | Relax Replacement |
| --- | --- |
| `from redaccel.verl.rewards.group.base import GroupRewards, group_rewards_registry` | Remove. Use standalone functions |
| `from redaccel.verl.rewards.std.base import GRPORewards, rewards_registry` | Remove. Use standalone functions |
| `from redaccel.verl.rewards.utils import get_judge_model_api_clients` | `from openai import OpenAI` |
| `from redaccel.verl.rewards.utils import call_gemini_flash` | Direct OpenAI/httpx API call (see [LLM-as-Judge replacement](#llm-as-judge-replacement)) |
| `from redaccel.verl.agent.tool_envs import ToolBase` | `from examples.<algo>.base_env import BaseInteractionEnv` |
| `from redaccel.verl.agent.utils import build_obs_text` | Use `_build_obs_text()` method on env class |

______________________________________________________________________

## Reward function transformation

### Pattern A: Standard (per-sample) reward — `GRPORewards`

RedAccel `GRPORewards.__call__` processes samples in a loop and returns `List[dict]`. In Relax, each sample is processed individually by `reward_func`.

**Before (RedAccel):**

```python
@rewards_registry.register()
class VideoAgentChatScoreV7(GRPORewards):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_step = 0

    def __call__(self, prompts, completions, solutions, **kwargs):
        self.local_step += 1
        res = []
        for completion, solution in zip(completions, solutions):
            res.append(compute_score(
                completion, solution,
                extra_info=kwargs.get("extra_info"),
                local_step=self.local_step
            ))
        return res
```

**After (Relax):**

```python
from relax.utils.types import Sample

def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> dict:
    # Same scoring logic, returns dict with "score" key
    ...
    return {"score": final_score, "acc": acc, ...}

async def reward_func(args, sample: Sample, **kwargs):
    question = sample.metadata.get("question")
    ground_truth = sample.label
    return compute_score(sample.response, ground_truth, extra_info={"question": question})
```

### Pattern B: Group reward — `GroupRewards`

RedAccel `GroupRewards.__call__` receives all samples in a group for cross-comparison (e.g., tool_count bonuses). Relax processes samples individually.

**Strategy:** Move group-level logic (like `compute_group_bonus`) into the individual reward or defer it to post-processing. If group comparison is essential:

1. Compute per-sample stats in `reward_func` and return them in the dict
2. Use `batched_async_rm` by providing a custom batch-level `reward_func`:

```python
async def reward_func(args, samples: list[Sample], **kwargs):
    """Batch-mode reward: receives all samples for group comparison."""
    stats = []
    for sample in samples:
        stat = await _compute_single_stat(sample)
        stats.append(stat)

    bonuses = compute_group_bonus(stats)
    results = []
    for stat, bonus in zip(stats, bonuses):
        score = stat["acc_reward"] + stat["density_reward"] + stat["penalty"] + bonus
        results.append(score)
    return results
```

Register it via `--custom-rm-path examples.<algo>.reward.reward_func`. The engine detects batch-mode automatically when `custom_rm_path` is set and calls it with `list[Sample]` via `batched_async_rm`.

### `local_step` handling

RedAccel uses class instance state `self.local_step` for staged training. In Relax, since reward is stateless/function-based, use one of:

1. **Environment variable**: `os.environ.get("TRAINING_STEP", "0")`
2. **Sample metadata**: pass step through `sample.metadata["step"]` from the controller
3. **Custom config**: define in the YAML config file and access via `args.step`

______________________________________________________________________

## Agent environment transformation

### Interface mapping

| RedAccel (`ToolBase`) | Relax (`BaseInteractionEnv`) |
| --- | --- |
| `__init__(name, desc, params, **kwargs)` | `__init__(*, max_turns, image=None, ...)` |
| `reset(raw_prompt, multi_modal_data, origin_multi_modal_data)` | `__init__` receives sample data; `reset()` takes no args |
| `execute(action_string) → (obs, reward, done, info)` | `step(response_text) → (obs_dict, done, info)` |
| Returns chatml list `[{"role":"user","content":...}]` | Returns obs dict `{"obs_str": text, "role": "user", "multi_modal_data": {...}}` |
| Inline reward (`reward` in return tuple) | No inline reward (reward is separate) |

### Observation format

**RedAccel** returns either a list of messages or a dict with `chat` + `multi_modal_data`:

```python
# RedAccel observation
obs = {
    "chat": [{"role": "user", "content": f"<tool_response>{result}</tool_response>"}],
    "multi_modal_data": {"image": [img]}
}
# or just
obs = [{"role": "user", "content": "..."}]
```

**Relax** uses a standardized obs dict:

```python
# Relax observation
obs = {
    "obs_str": "<tool_response>search result text</tool_response>",
    "role": "user",                                    # or "tool"
    "multi_modal_data": {"image": [pil_image]},        # optional
}
```

The `format_observation()` method on `BaseInteractionEnv` converts this dict into the message format expected by the chat template.

### Factory function

Relax requires a `build_env(sample, args)` factory:

```python
def build_env(sample: Sample = None, args=None, **_) -> MyEnv:
    max_turns = args.max_turns
    image = None
    if sample and sample.multimodal_inputs:
        images = sample.multimodal_inputs.get("images") or sample.multimodal_inputs.get("image")
        if images:
            image = images[0]
    return MyEnv(max_turns=max_turns, image=image)
```

### Config YAML

```yaml
max_turns: 5
rollout_interaction_env_path: examples.<algo>.env_<algo>
```

______________________________________________________________________

## LLM-as-Judge replacement

RedAccel provides `call_gemini_flash` from `redaccel.verl.rewards.utils`. In Relax, use direct API calls:

**Option A: OpenAI SDK (recommended)**

```python
import os
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["JUDGE_API_KEY"],
            base_url=os.environ.get("JUDGE_BASE_URL"),
        )
    return _client

def call_judge(prompt: str, model: str = None, timeout: int = 120) -> str:
    client = _get_client()
    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"Judge call failed after 3 attempts: {e}")
                return ""
    return ""
```

**Option B: httpx async (for high concurrency)**

```python
import httpx

async def call_judge_async(prompt: str, timeout: int = 120) -> str:
    url = os.environ["JUDGE_BASE_URL"] + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['JUDGE_API_KEY']}"}
    payload = {
        "model": os.environ.get("JUDGE_MODEL", "gpt-4o"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        return resp.json()["choices"][0]["message"]["content"].strip()
```

______________________________________________________________________

## Data field mapping

### `Sample` dataclass fields

| Access Pattern | Description |
| --- | --- |
| `sample.prompt` | Formatted prompt string (after chat template) |
| `sample.response` | Model-generated completion text |
| `sample.label` | Ground truth / solution string |
| `sample.metadata` | Dict with extra fields (question, type, etc.) |
| `sample.multimodal_inputs` | Dict like `{"images": [PIL.Image], "videos": [...]}` |
| `sample.tokens` | Token ID list (prompt + response) |
| `sample.loss_mask` | Per-token loss mask |
| `sample.status` | `Sample.Status.COMPLETED / TRUNCATED / ABORTED` |

### RedAccel → Relax data field mapping

| RedAccel Field | Relax Equivalent |
| --- | --- |
| `prompts[i]` | `sample.prompt` |
| `completions[i]` | `sample.response` |
| `solutions[i]` | `sample.label` |
| `kwargs["extra_info"][i]["question"]` | `sample.metadata["question"]` |
| `kwargs["extra_info"][i]["type"]` | `sample.metadata["type"]` |
| `multi_modal_data["image"]` | `sample.multimodal_inputs["images"]` |

### Parquet data columns

| Column | Relax Argument | Purpose |
| --- | --- | --- |
| `prompt` | `--input-key prompt` | Chat messages for the model |
| `reward_model` / `answer` | `--label-key reward_model` | Ground truth for reward |
| `images` | `--multimodal-keys '{"image":"images"}'` | Image columns |
| `extra_info` | `--metadata-key extra_info` | Extra metadata dict |

______________________________________________________________________

## Launch script transformation

### Environment setup

```bash
# RedAccel
source ${DIR}/env.sh "$@"
export PYTHONPATH=$DIR/../:$PYTHONPATH
redaccel-cli rlx setup <(echo 'sed -i ...')

# Relax
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
cd "${PROJECT_ROOT}"
export PYTHONPATH=/root/Megatron-LM/:${PROJECT_ROOT}:$PYTHONPATH
```

### Model config

```bash
# RedAccel: inline in env.sh
# Relax: source a model config
source "${SCRIPT_DIR}/../../scripts/models/qwen3-vl-30B-A3B.sh"
```

### Checkpoint

```bash
# RedAccel
DIST_CKPT_PATH=/path/to/mcore/checkpoint

# Relax
CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/ModelName
    --ref-load ${MODEL_DIR}/ModelName
    --save ${SAVE_DIR}/ModelName-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
    # --load ${SAVE_DIR}/resume-checkpoint   # for resuming
)
```

### Launch command

```bash
# RedAccel
redaccel-cli train "${cmd[@]}"

# Relax
ray start --head --num-gpus 8 --disable-usage-stats
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
```

______________________________________________________________________

## Example: Chat reward (sly_chat)

Migrating `VideoAgentChatScoreV7` (non-agentic, LLM-as-Judge based).

### Target structure

```
examples/sly_chat/
├── __init__.py
├── reward_chat.py          # compute_score + reward_func
├── llm_judge.py            # Judge prompt templates
└── run_chat.sh             # Launch script
```

### Key changes

1. Remove `@rewards_registry.register()` and `class VideoAgentChatScoreV7(GRPORewards)`
2. Keep `compute_score()` as-is (already a standalone function)
3. Replace `call_gemini_flash` → OpenAI SDK call
4. Add `async def reward_func(args, sample, **kwargs)` wrapper
5. No env/rollout needed (pure completion, no tool calling)

______________________________________________________________________

## Example: Agent reward (sly_agent)

Migrating `VideoAgentSearchGroupScore` (agentic with search tools).

### Target structure

```
examples/sly_agent/
├── __init__.py
├── base_env.py             # Copy from deepeyes or import
├── env_agent.py            # VideoSearchEnv(BaseInteractionEnv) + build_env()
├── reward_agent.py         # compute_score + reward_func
├── llm_judge.py            # Judge helpers (get_acc, get_density_score)
├── search_tools.py         # unified_search_async equivalent
├── agent_config.yaml       # max_turns, env module path
├── rollout.py              # Copy from deepeyes, update DEFAULT_ENV_MODULE
└── run_agent.sh            # Launch script
```

### Key changes

1. `VideoSearchTools(ToolBase)` → `VideoSearchEnv(BaseInteractionEnv)`
2. `execute(action_string)` → `step(response_text)`: parse tool calls, run search, return obs dict
3. Group bonus logic: either inline in per-sample reward (simplified) or implement batch `reward_func`
4. `_extract_action` helper: keep as-is, move to env
5. Search tools (`unified_search_async.py`): keep, fix imports for `httpx` (remove `redaccel` deps)
6. Add `build_env(sample, args)` factory
7. Copy rollout.py from deepeyes, set `DEFAULT_ENV_MODULE = "examples.sly_agent.env_agent"`
