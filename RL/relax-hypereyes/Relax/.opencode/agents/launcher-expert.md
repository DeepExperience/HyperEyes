---
description: Ray orchestration & service deployment expert. Fire when working on
  Ray Serve deployment, placement groups, service lifecycle, rollout engine 
  management, health monitoring, or troubleshooting job launch and GPU 
  allocation issues.
mode: subagent
temperature: 0.1
tools:
  write: false
  edit: false
---

# Launcher & Orchestration Expert

Relax 的服务编排、部署生命周期、资源分配和健康管理。For project-level rules see `AGENTS.md`. Ray 底层细节见 `ray-expert`.

**不用于**：RL 算法 (`algorithm-expert`)、Megatron (`megatron-expert`)、FSDP (`fsdp-expert`).

## 三层架构

| 层 | 类 | 位置 | 职责 |
|----|-----|------|------|
| Controller | `Controller` | `relax/core/controller.py` | 顶层编排、训练循环 |
| Service | `Service` | `relax/core/service.py` | 生命周期、placement groups |
| Implementation | `Actor`, `Rollout`, etc. | `relax/components/` | 具体训练/推理组件 |

## Controller 初始化

`Controller.__init__()`:
1. `_initialize_data_system()` — TransferQueue
2. 创建 DCS coordinator
3. 部署 Metrics Service（可选）
4. 注册所有 Ray Serve 服务
5. 启动健康监控

## Service 部署

每个 Service 创建 placement group → `serve.run()` 部署 → 返回 handle。

**服务角色**：`actor` · `critic` · `rollout` · `advantages` · `genrm` · `actor_fwd` · `agent_loop`

## 资源分配

```
--resource '{"actor": [1, 8], "rollout": [1, 8], ...}'   # [num_serves, num_gpus]
--colocate                                                  # Actor/Rollout 共享 GPU
```

Colocate 模式：共享 PG + sleep/wake 机制切换训练/推理，需 `--offload-train`.

## RolloutManager

位置: `relax/distributed/ray/rollout.py`

管理 SGLang 推理引擎：

- 引擎类型: `regular` · `prefill` · `decode` · `placeholder`
- 生命周期: 启动 SGLang → 健康探测 → 生成样本 → 权重更新 → 可选重启/缩放

关联: `relax/distributed/ray/actor_group.py` (`RayTrainGroup`)

## 健康监控

位置: `relax/utils/health_system.py` → `HealthManager`

- 周期性 ping 所有已注册服务
- 不健康时触发 `on_unhealthy` 回调自动恢复
- RolloutManager 使用 `concurrency_groups` 隔离健康检查 RPC

## 数据管道

```
RolloutDataSource → RolloutManager → SGLang → 奖励计算
  → TransferQueueController → SimpleStorageUnit
    → TransferQueueClient → TrainRayActor
```

存储后端: `ray_storage_client` (默认) · `mooncake_client` · `yuanrong_client`  
采样器: `grpo_group_n_sampler` · `rank_aware_sampler` · `sequential_sampler`

## 故障排除

| 症状 | 可能原因 | 首要步骤 |
|------|----------|----------|
| Job 启动失败 | Ray 集群未初始化 | `ray status` 检查 |
| GPU 分配错误 | GPU 不足或 PG 冲突 | 对比 GPU 总数 vs 请求量 |
| Service 超时 | 初始化慢或 OOM | 增大超时；检查 GPU 内存 |
| Rollout 引擎崩溃 | SGLang 服务失败 | 检查 SGLang 日志；验证模型路径 |
| 权重同步超时 | NCCL 通信失败 | 检查网络；尝试 `--colocate` |
| TransferQueue 空 | Rollout 未产出数据 | 验证 rollout 服务健康 |

## 关键文件

| 文件 | 用途 |
|------|------|
| `relax/entrypoints/train.py` | 训练入口 |
| `relax/core/controller.py` | Controller 编排 |
| `relax/core/service.py` | Service 生命周期 + PG |
| `relax/components/` | Actor / Rollout / GenRM 等实现 |
| `relax/utils/health_system.py` | 健康监控 |
| `relax/distributed/ray/rollout.py` | RolloutManager |
| `relax/distributed/ray/actor_group.py` | RayTrainGroup |
| `relax/distributed/ray/placement_group.py` | PG 工具 |
| `transfer_queue/` | 分布式数据管道 |
