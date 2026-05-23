# On-Policy Distillation (OPD)

On-Policy Distillation (OPD) enables knowledge transfer from a large teacher model to a smaller student model by training the student on its own rollout data while matching the teacher's token-level log-probabilities. OPD is orthogonal to the advantage estimator—it acts as a KL penalty term that can be combined with any estimator (GRPO, PPO, REINFORCE++, etc.).

## Key Parameters

| Parameter                 | Description                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--use-opd`               | Enable On-Policy Distillation. Required flag when using OPD.                                                                               |
| `--opd-type`              | OPD type: `sglang` or `megatron`. Must be set when `--use-opd` is enabled.                                                                 |
| `--opd-kl-coef`           | OPD KL penalty coefficient (default: 1.0). Controls the weight of the distillation signal relative to the RL advantage.                    |
| `--opd-teacher-load`      | Path to the teacher model's Megatron checkpoint. **Must** be set when `--opd-type=megatron`, **must not** be set when `--opd-type=sglang`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for the teacher model.                                                                                            |

## How It Works

OPD modifies the advantage calculation by subtracting a KL penalty term, encouraging the student to match the teacher's output distribution:

$$\\hat{A}_t = A_t - \\lambda_{\\text{opd}} \\cdot D\_{\\text{KL}}(P\_{\\text{teacher}} | P\_{\\text{student}})\_t$$

Where $A_t$ is the original advantage from the base estimator (e.g., GRPO), $\\lambda\_{\\text{opd}}$ is `--opd-kl-coef`, and $D\_{\\text{KL}}$ is the token-level reverse KL divergence.

Therefore, OPD can be combined with any advantage estimator, including GRPO, PPO, REINFORCE++, and GSPO.

## Two Teacher Modes

### SGLang Mode (`--opd-type sglang`)

The teacher model runs on an external SGLang server, and the teacher's log-probs are obtained during the rollout phase.

**Use cases**: Teacher architecture differs from student, or teacher model is too large to be loaded together with the training model.

**Workflow**:

1. An external SGLang server runs the teacher model.
2. During rollout, after the reward is computed for each sample, the framework automatically sends the sample to the teacher server to obtain token-level log-probs and stores them in `sample.teacher_log_probs`.
3. During training, the KL penalty is computed from the stored teacher log-probs and applied to advantages.

> **Note**: OPD sglang mode does NOT occupy `--custom-rm-path` or `--custom-reward-post-process-path`. Users can freely use custom reward functions alongside OPD.

**Configuration**:

```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

### Megatron Mode (`--opd-type megatron`)

The teacher model is directly loaded into Megatron via `--opd-teacher-load`, and the teacher's log-probs are computed during the training forward pass.

**Use cases**: Teacher and student/reference model have the same architecture and can fit in GPU memory together.

**Workflow**:

1. The teacher model is loaded as an additional Megatron model during initialization.
2. During the training forward pass, the teacher model computes log-probs for each sample.
3. The KL penalty is computed inline and applied to advantages.

**Configuration**:

```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_model
```

> **Note**: The teacher checkpoint must be in Megatron format (`torch_dist` or `torch`). You can use `tools/convert_hf_to_torch_dist.py` to convert from HuggingFace format.

## Running Examples

Complete example scripts are available in `examples/on_policy_distillation/`

## Preliminary Results

Using Qwen3-8B-Base model, performing SFT on a portion of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, then applying On-Policy Distillation with Qwen3-32B teacher on the remaining data, the Math500 evaluation results are as follows:

| Model                                        | Pass@1 |
| -------------------------------------------- | ------ |
| Qwen3-8B-Base + SFT                          | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation | 94%    |

## Backend Support

| Backend  | SGLang Teacher | Megatron Teacher   |
| -------- | -------------- | ------------------ |
| Megatron | ✅             | ✅                 |
