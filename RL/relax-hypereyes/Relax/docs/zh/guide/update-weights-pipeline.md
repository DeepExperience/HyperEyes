# Colocate 模式 update_weights 流水线优化

## 概述

在 Colocate 模式下，Actor 训练完成后需要将 Megatron 格式的权重转换为 HF 格式，并通过 CUDA IPC 传输给 SGLang 推理引擎。由于权重按 chunk 分批传输，原始实现中 chunk 间严格串行（发送 → 阻塞等待 → 下一个），导致 `update_weights_time` 耗时较大。

本优化通过 **chunk 间流水线化**（延迟 `ray.get`，使 IPC 传输与下一个 chunk 的 HF 转换重叠）和 **移除不必要的 barrier**，将耗时降低约一倍。

______________________________________________________________________

## 问题背景

### 场景描述

在多机多卡环境下训练大规模 MoE 模型时，`perf/update_weights_time` 相比预期偏高约一倍。

### 权重更新调用链

```
MegatronTrainRayActor.update_weights()
  └─ UpdateWeightFromTensor.update_weights()
       ├─ rank 0: pause_generation + flush_cache
       ├─ dist.barrier(gloo)                          ← 必要
       ├─ weights_getter()                            ← 获取 Megatron 本地权重
       │
       ├─ for chunk in get_hf_weight_chunks():        ← 多个 chunk
       │     ├─ _send_hf_params(chunk)                  HF 转换 + 序列化 + Gloo gather
       │     └─ ray.get(refs)                         ← 阻塞等待！
       │
       └─ rank 0: continue_generation
```

核心瓶颈在于调用链中的 `ray.get(refs)` —— 每个 chunk 必须等待 IPC 传输完成后才开始下一个，多个 chunk 严格串行，占据了总耗时的大部分。

______________________________________________________________________

## 优化方案

### Chunk 间流水线化

**核心思路**：将 `ray.get` 延迟到下一个 chunk 发送完成后再执行，使当前 chunk 的 IPC 传输与下一个 chunk 的 HF 转换重叠。

**优化前**（严格串行）：

```
Chunk 0: [HF转换][序列化+gather][IPC传输][等待]
Chunk 1:                                       [HF转换][序列化+gather][IPC传输][等待]
Chunk 2:                                                                             [HF转换]...
```

**优化后**（流水线重叠）：

```
Chunk 0: [HF转换][序列化+gather][IPC传输]
Chunk 1:                        [HF转换][序列化+gather][等待prev][IPC传输]
Chunk 2:                                               [HF转换][序列化+gather][等待prev][IPC传输]
                                                                                              [等待last]
```

优化后的代码：

```python
prev_refs: list[ObjectRef] = []
prev_long_lived_tensors = None
for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
    if prev_refs:
        ray.get(prev_refs)
    del prev_long_lived_tensors
    prev_refs = refs
    prev_long_lived_tensors = long_lived_tensors
if prev_refs:
    ray.get(prev_refs)
del prev_long_lived_tensors
```


______________________________________________________________________

## 预期收益

- **chunk 循环耗时**：流水线重叠显著缩短
- **总 update_weights_time**：整体降低约一倍

### 资源影响

| 资源 | 影响 |
|------|------|
| GPU 显存 | 峰值略有增加（额外一个 chunk 的 `long_lived_tensors`） |
| CPU 内存 | 无变化 |
| 网络带宽 | 无变化（总传输量不变，只是时序重叠） |

### 适用范围

此优化对以下场景收益最大：

- **MoE / 大参数模型**：chunk 数量多，流水线重叠收益显著
- **Colocate 模式**：CUDA IPC 路径下 IPC 传输与 HF 转换可真正并行
- **多节点部署**：跨节点引擎时延更高，流水线重叠收益更明显


______________________________________________________________________

## 下一步

- [Distributed Checkpoint](./distributed-checkpoint.md) — 非 Colocate 部署中使用的 DCS 权重同步机制
