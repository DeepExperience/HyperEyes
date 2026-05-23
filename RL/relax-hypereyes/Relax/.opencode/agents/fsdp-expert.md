---
description: FSDP backend expert. Fire when working on FSDP-based training, parameter
  sharding, FSDP weight update, CPU offloading, or troubleshooting FSDP-related issues.
mode: subagent
temperature: 0.1
tools:
  write: false
  edit: false
---

# FSDP Backend Expert

Relax 中 FSDP (PyTorch FSDP2) 后端的配置、集成和故障排除。For project-level rules see `AGENTS.md`.

**不用于**：RL 算法 (`algorithm-expert`)、Megatron (`megatron-expert`)、Ray 编排 (`launcher-expert`).

## 核心类

| 类 | 位置 | 职责 |
|---|------|------|
| `FSDPTrainRayActor` | `relax/backends/fsdp/actor.py` | FSDP 训练 Actor |
| `UpdateWeightFromTensor` | `relax/backends/fsdp/update_weight_utils.py` | Colocate 模式权重更新 |
| `UpdateWeightFromDistributed` | `relax/backends/fsdp/update_weight_utils.py` | 分布式 NCCL 权重更新 |

## 初始化流程

`FSDPTrainRayActor._init()`:
1. `_setup_device_mesh()` — 创建 DP/TP 进程组
2. Sequential load — HF config + tokenizer（避免竞态）
3. `apply_fsdp2()` — 参数分片
4. Load HuggingFace state dict
5. 初始化 AdamW 优化器
6. 创建 weight updater（按 `--colocate` 选择）

## 配置参数

位置: `relax/backends/fsdp/arguments.py`

| 参数 | 说明 |
|------|------|
| `--train-backend fsdp` | 选择 FSDP 后端 |
| `--fsdp-cpu-offload` | 参数卸载到 CPU |
| `--gradient-checkpointing` | 激活检查点 |
| `--colocate` | 训练/推理共享 GPU |

## 常用配置

| 场景 | 关键设置 |
|------|----------|
| 内存受限 | `--fsdp-cpu-offload --gradient-checkpointing` |
| 高吞吐 | `--max-tokens-per-gpu 8192`，不开 offload |
| Colocate | `--colocate --offload-train` |
| 长序列 | `--use-dynamic-batch-size` |

## 特性

- **检查点**：原生 HuggingFace 格式，无需权重转换。支持 DCS 异步保存。
- **Data Packing**：`relax/backends/fsdp/data_packing.py` 的 `pack_sequences()` / `unpack_sequences()` 高效处理变长序列。

## vs Megatron

- **FSDP**：大型密集模型、简单配置、原生 HF 格式
- **Megatron**：超深模型、MoE、PP + EP 并行

## 故障排除

| 症状 | 可能原因 | 首要步骤 |
|------|----------|----------|
| 初始化挂起 | Device mesh 配置错误 | 验证 `world_size` 匹配 GPU 数 |
| OOM | GPU 内存不足 | 启用 `--fsdp-cpu-offload`，减少 batch |
| 检查点加载失败 | State dict key 不匹配 | 验证 HF 模型格式 |
| 权重同步失败 | NCCL 通信错误 | 检查网络；尝试 disk-based 更新 |
| 吞吐低 | CPU offload 开销 | 若内存允许关闭 offload |

## 关键文件

| 文件 | 用途 |
|------|------|
| `relax/backends/fsdp/actor.py` | 训练 Actor |
| `relax/backends/fsdp/arguments.py` | 参数扩展 |
| `relax/backends/fsdp/checkpoint.py` | HF 格式检查点 |
| `relax/backends/fsdp/data_packing.py` | 序列打包 |
| `relax/backends/fsdp/update_weight_utils.py` | 权重更新策略 |
| `relax/backends/fsdp/models/` | 模型特定 FSDP 配置 |
