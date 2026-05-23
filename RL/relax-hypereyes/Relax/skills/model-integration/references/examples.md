# Model Integration Examples

Concrete code examples extracted from the Relax codebase.

## Table of Contents

1. [Simple Converter: Qwen2](#simple-converter-qwen2)
2. [Complex Converter: Qwen3.5 (MoE + Vision + GDN)](#complex-converter-qwen35)
3. [Custom TP All-Gather: Qwen3.5 GDN](#custom-tp-all-gather)
4. [Custom TP Chunk: Qwen3.5 GDN](#custom-tp-chunk)
5. [Converter Registration](#converter-registration)
6. [Model Config Script](#model-config-script)
7. [Launch Script (Colocate)](#launch-script-colocate)
8. [Launch Script (Fully-Async)](#launch-script-fully-async)

---

## Simple Converter: Qwen2

`slime/backends/megatron_utils/megatron_to_hf/qwen2.py` — 标准 dense Transformer，71 行。

```python
import re
import torch

def convert_qwen2_to_hf(args, name, param):
    # 全局参数
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    # 注意力头维度计算
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # 每层参数
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Attention
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            # GQA 拆分：Megatron 按 [Q_group0, K0, V0, Q_group1, K1, V1, ...] 排列
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]

        # MLP (SwiGLU)
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        # Fused LayerNorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # QK Norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
```

---

## Complex Converter: Qwen3.5

`slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py` — 含 MoE、MTP、Vision、GDN 线性注意力。

关键差异点（相比简单 Transformer）：

```python
from slime.utils.misc import get_hf_config

def convert_qwen3_5_to_hf(args, name, param):
    # 1. MTP (Multi-Token Prediction) 层代理
    if "mtp.layers" in name:
        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    # 2. 归一化 module prefix (Qwen3.5 多了 language_model 层级)
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model."):]
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    # 3. Vision 子模型单独处理
    if name.startswith("module.module.vision_model."):
        return convert_qwen3_5_vision_to_hf(args, name, param)

    # 4. MoE experts (grouped gemm)
    if rest == "mlp.experts.linear_fc1":
        return [(f"{prefix}.mlp.experts.gate_up_proj", param)]

    # 5. MoE experts (ungrouped, per-expert)
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest == "linear_fc1":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
            ]

    # 6. GDN 线性注意力 — in_proj 需要从 Megatron 拼接格式拆回 HF 4 张量
    if rest == "self_attention.in_proj.weight":
        config = get_hf_config(args.hf_checkpoint).text_config
        qkv, z, b, a = gdn_mca_to_hf(config, param)
        return [
            (f"{prefix}.linear_attn.in_proj_qkv.weight", qkv),
            (f"{prefix}.linear_attn.in_proj_z.weight", z),
            (f"{prefix}.linear_attn.in_proj_b.weight", b),
            (f"{prefix}.linear_attn.in_proj_a.weight", a),
        ]

    # 7. ZeroCenteredRMSNorm → 标准 RMSNorm 需 +1
    elif rest == "self_attention.out_norm.weight":
        return [(f"{prefix}.linear_attn.norm.weight", param + 1)]
```

---

## Custom TP All-Gather

`slime/backends/megatron_utils/update_weight/common.py` — Qwen3.5 GDN 自定义 TP 分片示例。

```python
def all_gather_param(args, name: str, param: torch.nn.Parameter) -> torch.Tensor:
    # ... 通用 skip 逻辑省略 ...

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()

    # conv1d.weight: TP 按 q/k/v 三分量独立分片
    if "self_attention.conv1d.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads
        qk_dim = qk_dim // tp_size    # 每个 TP rank 持有的 qk 维度
        v_dim = v_dim // tp_size

        q = param[torch.arange(qk_dim)]
        k = param[torch.arange(qk_dim) + qk_dim]
        v = param[torch.arange(v_dim) + qk_dim * 2]

        full_weights = []
        for c in [q, k, v]:
            gathered = [torch.empty_like(c.data) for _ in range(tp_size)]
            dist.all_gather(gathered, c.data, group=tp_group)
            full_weights.append(torch.cat(gathered, dim=0))
        return torch.cat(full_weights, dim=0)

    # in_proj.weight: TP 按 [q, k, v, z, b, a] 六段独立分片
    if "self_attention.in_proj.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        # 计算每个 TP rank 持有的各段维度 ...
        full_weights = []
        for c in torch.split(param, [qk_local, qk_local, v_local, v_local, vh_local, vh_local], dim=0):
            gathered = [torch.empty_like(c.data) for _ in range(tp_size)]
            dist.all_gather(gathered, c.data, group=tp_group)
            full_weights.append(torch.cat(gathered, dim=0))
        return torch.cat(full_weights, dim=0)
```

---

## Custom TP Chunk

`relax/checkpoint_engine/utils.py` — `all_gather_param` 的严格逆操作。

```python
def chunk_param(args, name, target_param, full_param):
    # ... 通用 skip 逻辑省略 ...

    # conv1d.weight: 按 q/k/v 三分量独立切片
    if "self_attention.conv1d.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads

        q_full, k_full, v_full = torch.split(full_param, [qk_dim, qk_dim, v_dim], dim=0)
        shards = []
        for component in [q_full, k_full, v_full]:
            chunks = torch.chunk(component, tp_size, dim=0)
            shards.append(chunks[tp_rank])
        return torch.cat(shards, dim=0)

    # in_proj.weight: 按 [q, k, v, z, b, a] 六段独立切片
    if "self_attention.in_proj.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        # 拆分为 6 段，每段独立 chunk 取 tp_rank 对应的分片
        segments = torch.split(full_param, [qk_dim, qk_dim, v_dim, v_dim, num_v_heads, num_v_heads], dim=0)
        shards = []
        for seg in segments:
            chunks = torch.chunk(seg, tp_size, dim=0)
            shards.append(chunks[tp_rank])
        return torch.cat(shards, dim=0)
```

---

## Converter Registration

`slime/backends/megatron_utils/megatron_to_hf/__init__.py`

```python
# 顶部 import（按字母序）
from .qwen3_5 import convert_qwen3_5_to_hf

# 路由函数（注意顺序：具体 → 通用）
def _convert_to_hf_core(args, model_name, name, param):
    if "glm4moelite" in model_name or "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3omni" in model_name:
        converted_named_tensors = convert_qwen3omni_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen3_5" in model_name:                          # ← 更具体
        converted_named_tensors = convert_qwen3_5_to_hf(args, name, param)
    elif "qwen3vl" in model_name:
        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:   # ← 更通用
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    ...
```

---

## Model Config Script

`scripts/models/qwen35-9B.sh`

```bash
MODEL_ARGS=(
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 12288
    --use-gated-attention        # qwen3.5 特有

    --normalization RMSNorm
    --apply-layernorm-1p         # ZeroCenteredRMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate      # qwen3.5 特有
)
```

---

## Launch Script (Colocate)

```bash
ray job submit ... \
   python3 relax/entrypoints/train.py \
   --colocate \                          # 共卡模式
   --megatron-to-hf-mode bridge \        # 推荐使用 Bridge
   --hf-checkpoint /path/to/hf/model \
   --ref-load /path/to/ref/model \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} ...
```

---

## Launch Script (Fully-Async)

```bash
ray job submit ... \
   python3 relax/entrypoints/train.py \
   --fully-async \                       # 全异步模式
   --megatron-to-hf-mode bridge \
   --hf-checkpoint /path/to/hf/model \
   --resource '{"actor": [1, 4], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
    --max-staleness 2 \                   # Rollout 可容忍的权重版本滞后
   --num-data-storage-units 1 \
   --num-iters-per-train-update 8 \
   ${MODEL_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} ...
```
