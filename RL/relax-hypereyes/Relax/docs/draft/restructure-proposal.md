# Relax 仓库目录重构提案

> **状态**: Draft v2  
> **作者**: 禹哲  
> **日期**: 2025-07  
> **范围**: `relax/` + `slime/` 合并；`slime_plugins/` 删除；`transfer_queue/` 不动

---

## 1. 现状问题分析

### 1.1 循环依赖

`relax/` 与 `slime/` 存在严重双向耦合：

| 方向 | 导入数 | 主要内容 |
|------|--------|---------|
| `relax/` → `slime/` | 14 处 | `load_function`, `Sample`, `async_utils.run`, `placement_group`, `cp_utils`, `ppo_utils`, `distributed_utils`, `processing_utils`, `timer`, `apprise_utils`, `clearml_utils`, `tensorboard_utils`, `megatron_to_hf`, `update_weight` |
| `slime/` → `relax/` | 48 处 | 主要是 `get_logger`（40+ 处），少量 `relax.utils.utils`、`relax.metrics.client`、`relax.checkpoint_engine` |

### 1.2 `slime/utils/` 过度膨胀

39 个 Python 文件混杂：参数解析、分布式通信、多模态处理、监控适配器、内存管理等完全无关的职责。

### 1.3 顶层目录过多

当前 `relax/` 下有 **10 个** 同级目录（`controller/`, `services/`, `impl/`, `metrics/`, `checkpoint_engine/` …），层次扁平但职责边界模糊。

### 1.4 命名问题

| 问题 | 具体表现 |
|------|---------|
| `impl/` 含义模糊 | "实现"什么？实际是 RL 服务组件（Actor/Critic/Rollout 等 Ray Serve Deployment） |
| `services/` 仅一个文件 | 单文件目录，与 `impl/` 的关系不清晰 |
| `checkpoint_engine/` 冗长 | 可以更简洁 |
| `*_hub` 后缀 | `rm_hub`、`filter_hub`、`middleware_hub` 不够直观 |
| `*_utils` 后缀泛滥 | `megatron_utils/`、`sglang_utils/`、`external_utils/`、`debug_utils/` 可以去掉后缀 |

### 1.5 其他

- `slime_plugins/`：整个目录仅含两个空子目录，零 Python 文件，零导入
- 死代码文件散落（详见 §4）

---

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| **按领域分组，而非按技术模式** | 不要出现 `impl/`、`services/` 这类按设计模式命名的目录 |
| **名字即文档** | 目录名一眼能看出职责，无需查看内部代码 |
| **浅层嵌套** | 顶层 ≤ 6 个目录，嵌套 ≤ 3 层 |
| **单一职责** | 每个目录有明确、单一的责任边界 |
| **依赖单向流动** | `core/` → `components/` → `engine/` / `backends/` → `distributed/` → `utils/` |

---

## 3. 目标目录结构

将 `relax/` 的 10 个同级目录 + `slime/` 的 5 个子包，整合为 **6 个** 顶层模块：

```
relax/
├── core/              编排层 — 训练循环、服务基类、全局注册表
├── components/        组件层 — RL 服务组件（Ray Serve Deployment）
├── engine/            引擎层 — Rollout 数据生成、奖励计算、请求路由
├── backends/          后端层 — Megatron 训练后端、SGLang 推理引擎
├── distributed/       分布式层 — Ray 集群管理、分布式 Checkpoint
└── utils/             基础设施 — 工具函数、指标监控、多模态处理
```

### 3.1 完整目录树

```
relax/
├── __init__.py
├── _version.py
│
│  ═══════════════════════════════════════════════════════════════
│  core/ — 编排层：训练循环 + 服务抽象 + 全局注册表
│  ═══════════════════════════════════════════════════════════════
├── core/
│   ├── __init__.py
│   ├── controller.py                ← relax/controller/controller.py
│   ├── service.py                   ← relax/services/service.py
│   └── registry.py                  ← relax/utils/const.py（重命名，语义更清晰）
│
│  ═══════════════════════════════════════════════════════════════
│  components/ — RL 服务组件（每个文件 = 一个 @serve.deployment）
│  ═══════════════════════════════════════════════════════════════
├── components/
│   ├── __init__.py
│   ├── base.py                      ← relax/impl/base.py
│   ├── actor.py                     ← relax/impl/actor.py         策略训练
│   ├── actor_fwd.py                 ← relax/impl/actor_fwd.py     前向推理 log-prob
│   ├── critic.py                    ← relax/impl/critic.py        价值估计
│   ├── advantages.py                ← relax/impl/advantages.py    优势计算
│   ├── genrm.py                     ← relax/impl/genrm.py         生成式奖励模型
│   └── rollout.py                   ← relax/impl/rollout.py       Rollout 服务编排
│
│  ═══════════════════════════════════════════════════════════════
│  engine/ — 数据生成引擎：Rollout 实现 + 奖励函数 + 采样过滤 + 路由
│  ═══════════════════════════════════════════════════════════════
├── engine/
│   ├── __init__.py
│   ├── rollout/                     ← slime/rollout/（核心 Rollout 实现）
│   │   ├── __init__.py
│   │   ├── base_types.py
│   │   ├── data_source.py
│   │   ├── sglang_rollout.py            SGLang 在线生成
│   │   ├── fully_async_rollout.py       全异步 Rollout
│   │   └── on_policy_distillation.py    在线策略蒸馏
│   ├── rewards/                     ← slime/rollout/rm_hub/（去掉 _hub 后缀）
│   │   ├── __init__.py
│   │   ├── dapo_genrm.py
│   │   ├── deepscaler.py
│   │   ├── f1.py
│   │   ├── gpqa.py
│   │   ├── ifbench.py
│   │   ├── math_dapo_utils.py
│   │   ├── math_utils.py
│   │   ├── multiple_choice.py
│   │   └── openr1mm.py
│   ├── filters/                     ← slime/rollout/filter_hub/（去掉 _hub 后缀）
│   │   ├── __init__.py
│   │   ├── base_types.py
│   │   └── dynamic_sampling_filters.py
│   └── router/                      ← slime/router/
│       ├── __init__.py
│       ├── router.py
│       └── middleware/              ← middleware_hub/（去掉 _hub 后缀）
│           ├── __init__.py
│           ├── radix_tree.py
│           └── radix_tree_middleware.py
│
│  ═══════════════════════════════════════════════════════════════
│  backends/ — 训练与推理后端
│  ═══════════════════════════════════════════════════════════════
├── backends/
│   ├── __init__.py
│   ├── megatron/                    ← slime/backends/megatron_utils/（去掉 _utils）
│   │   ├── __init__.py
│   │   ├── actor.py                     训练 Actor 入口
│   │   ├── arguments.py                 Megatron 参数扩展
│   │   ├── checkpoint.py                模型检查点
│   │   ├── ci_utils.py                  CI 工具
│   │   ├── cp_utils.py                  Context Parallelism
│   │   ├── data.py                      数据处理
│   │   ├── initialize.py                初始化
│   │   ├── loss.py                      损失计算
│   │   ├── misc_utils.py                杂项工具
│   │   ├── model.py                     模型定义
│   │   ├── model_provider.py            模型工厂
│   │   ├── sglang.py                    SGLang 集成桥接
│   │   ├── kernels/                     自定义 CUDA Kernels
│   │   │   ├── __init__.py
│   │   │   ├── fp8_kernel.py
│   │   │   └── int4_qat/
│   │   │       ├── fake_int4_quant_cuda.cu
│   │   │       └── setup.py
│   │   ├── weight_conversion/       ← megatron_to_hf/（语义化重命名）
│   │   │   ├── __init__.py
│   │   │   ├── deepseekv3.py
│   │   │   ├── glm4.py
│   │   │   ├── glm4moe.py
│   │   │   ├── llama.py
│   │   │   ├── mimo.py
│   │   │   ├── qwen2.py
│   │   │   ├── qwen3_5.py
│   │   │   ├── qwen3_next.py
│   │   │   ├── qwen3_omni_moe.py
│   │   │   ├── qwen3_vl.py
│   │   │   ├── qwen3moe.py
│   │   │   └── processors/
│   │   │       ├── __init__.py
│   │   │       ├── padding_remover.py
│   │   │       ├── quantizer_compressed_tensors.py
│   │   │       └── quantizer_fp8.py
│   │   └── weight_update/          ← update_weight/（语义化重命名）
│   │       ├── __init__.py
│   │       ├── common.py
│   │       ├── hf_weight_iterator_base.py
│   │       ├── hf_weight_iterator_bridge.py
│   │       ├── hf_weight_iterator_direct.py
│   │       ├── update_weight_from_distributed.py
│   │       └── update_weight_from_tensor.py
│   └── sglang/                      ← slime/backends/sglang_utils/（去掉 _utils）
│       ├── __init__.py
│       ├── arguments.py                 SGLang 参数
│       └── sglang_engine.py             SGLang 推理引擎
│
│  ═══════════════════════════════════════════════════════════════
│  distributed/ — 分布式计算基础设施：Ray 集群 + 分布式 Checkpoint
│  ═══════════════════════════════════════════════════════════════
├── distributed/
│   ├── __init__.py
│   ├── ray/                         ← slime/ray/
│   │   ├── __init__.py
│   │   ├── actor_group.py               RayTrainGroup 管理
│   │   ├── placement_group.py           GPU 资源分配
│   │   ├── ray_actor.py                 Ray 远程 Actor 基类
│   │   ├── train_actor.py               MegatronTrainRayActor
│   │   ├── rollout.py                   RolloutManager
│   │   ├── genrm.py                     GenRM Manager
│   │   └── utils.py
│   └── checkpoint/                  ← relax/checkpoint_engine/（精简命名）
│       ├── __init__.py
│       ├── config.py
│       ├── metrics.py
│       ├── utils.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── device_direct.py
│       ├── client/
│       │   ├── __init__.py
│       │   └── engine.py
│       └── coordinator/
│           ├── __init__.py
│           ├── service.py
│           └── topology.py
│
│  ═══════════════════════════════════════════════════════════════
│  utils/ — 基础设施层：工具函数 + 指标监控 + 数据处理 + 多模态
│  ═══════════════════════════════════════════════════════════════
└── utils/
    ├── __init__.py
    │
    │  ── 框架基础（原 relax/utils/ + slime/utils/ 合并） ──
    ├── logging_utils.py                 日志（全局基础设施）
    ├── arguments.py                 ← slime/utils/arguments.py     CLI 参数解析
    ├── async_utils.py               ← slime/utils/async_utils.py   异步工具
    ├── types.py                     ← slime/utils/types.py         核心类型（Sample 等）
    ├── misc.py                      ← slime/utils/misc.py          杂项（load_function 等）
    ├── utils.py                         通用工具函数
    ├── timer.py                     ← slime/utils/timer.py         计时器
    ├── http_utils.py                ← slime/utils/http_utils.py    HTTP 工具
    ├── memory_utils.py              ← slime/utils/memory_utils.py  内存管理
    ├── profile_utils.py             ← slime/utils/profile_utils.py 性能分析
    ├── reload_utils.py              ← slime/utils/reload_utils.py  热重载
    ├── health_system.py                 健康检查系统
    ├── health_monitor.py            ← slime/utils/health_monitor.py
    ├── tracking_utils.py            ← slime/utils/tracking_utils.py
    ├── megatron_bridge_utils.py     ← slime/utils/megatron_bridge_utils.py
    ├── genrm_client.py              ← slime/utils/genrm_client.py
    ├── checkpoint_write_patch.py    ← slime/utils/checkpoint_write_patch.py
    ├── rotate_ckpt.py                   检查点轮转
    ├── reloadable_process_group.py  ← slime/utils/reloadable_process_group.py
    ├── distributed_utils.py         ← slime/utils/distributed_utils.py
    ├── rocm_checkpoint_writer.py    ← slime/utils/rocm_checkpoint_writer.py
    │
    │  ── metrics/ — 指标采集与监控（合并 relax/metrics/ + 适配器） ──
    ├── metrics/
    │   ├── __init__.py
    │   ├── client.py                ← relax/metrics/client.py      指标客户端
    │   ├── service.py               ← relax/metrics/service.py     指标服务（Ray Serve）
    │   ├── timeline_trace.py        ← relax/metrics/timeline_trace.py
    │   ├── metric_checker.py        ← slime/utils/metric_checker.py
    │   ├── metric_utils.py          ← slime/utils/metric_utils.py
    │   ├── metrics_service_adapter.py ← slime/utils/metrics_service_adapter.py
    │   └── adapters/                    监控后端适配器
    │       ├── __init__.py
    │       ├── apprise.py           ← slime/utils/apprise_utils.py
    │       ├── clearml.py           ← slime/utils/clearml_utils.py
    │       ├── tensorboard.py       ← slime/utils/tensorboard_utils.py
    │       └── wandb.py             ← slime/utils/wandb_utils.py
    │
    │  ── data/ — 数据处理与加载 ──
    ├── data/
    │   ├── __init__.py
    │   ├── data.py                  ← slime/utils/data.py          数据加载
    │   ├── data_utils.py            ← slime/utils/data_utils.py    数据处理
    │   ├── streaming_dataset.py     ← slime/utils/streaming_dataset.py
    │   ├── stream_dataloader.py     ← relax/utils/stream_dataloader.py
    │   ├── mask_utils.py            ← slime/utils/mask_utils.py    Loss Mask
    │   ├── processing_utils.py      ← slime/utils/processing_utils.py  Tokenizer
    │   └── seqlen_balancing.py      ← slime/utils/seqlen_balancing.py
    │
    │  ── training/ — 训练专用工具 ──
    ├── training/
    │   ├── __init__.py
    │   ├── ppo_utils.py             ← slime/utils/ppo_utils.py     PPO/GRPO 优势计算
    │   ├── flops_utils.py           ← slime/utils/flops_utils.py   FLOPS 计算
    │   ├── eval_config.py           ← slime/utils/eval_config.py   评估配置
    │   ├── train_dump_utils.py      ← slime/utils/train_dump_utils.py
    │   ├── train_metric_utils.py    ← slime/utils/train_metric_utils.py
    │   ├── tensor_backper.py        ← slime/utils/tensor_backper.py
    │   └── routing_replay.py        ← slime/utils/routing_replay.py
    │
    │  ── multimodal/ — 多模态处理（原样保留） ──
    ├── multimodal/                  ← slime/utils/multimodal/
    │   ├── __init__.py
    │   ├── audio_utils.py
    │   ├── config.py
    │   ├── image_utils.py
    │   ├── process.py
    │   └── video_utils.py
    │
    │  ── external/ — 外部工具集成 ──
    ├── external/                    ← slime/utils/external_utils/（去掉 _utils）
    │   ├── __init__.py
    │   ├── command_utils.py
    │   └── typer_utils.py
    │
    │  ── debug/ — 调试工具 ──
    └── debug/                       ← slime/utils/debug_utils/（去掉 _utils）
        ├── __init__.py
        └── send_to_sglang.py
```

### 3.2 架构分层图

```
┌─────────────────────────────────────────────────────────┐
│                      core/                              │
│        controller.py ──→ registry.py ──→ service.py     │
│           训练循环         角色注册表      服务基类     │
└──────────────────────────┬──────────────────────────────┘
                           │ 实例化
┌──────────────────────────▼──────────────────────────────┐
│                   components/                           │
│    Actor  Critic  Rollout  ActorFwd  Advantages  GenRM  │
│    每个 = @serve.deployment Ray Serve 部署              │
└────────┬────────────────────────────┬───────────────────┘
         │ 委托生成                    │ 委托训练/推理
┌────────▼────────┐         ┌─────────▼──────────────────┐
│    engine/      │         │       backends/            │
│ rollout/        │         │  megatron/   训练后端      │
│ rewards/        │         │  sglang/     推理引擎      │
│ filters/        │         └─────────┬──────────────────┘
│ router/         │                   │
└────────┬────────┘                   │
         │                            │
┌────────▼────────────────────────────▼──────────────────┐
│                   distributed/                         │
│        ray/  Actor 组管理、Placement Group             │
│        checkpoint/  分布式权重同步                     │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│                      utils/                            │
│  metrics/   data/   training/   multimodal/   debug/   │
│  logging  arguments  types  async_utils  misc  ...     │
└────────────────────────────────────────────────────────┘
```

### 3.3 核心设计决策

#### 3.3.1 `impl/` → `components/`

`impl/` 是最模糊的命名之一。这些文件的本质是 **RL 服务组件** —— 每个文件对应一个 `@serve.deployment` 部署：

```python
# relax/components/actor.py
@serve.deployment(max_ongoing_requests=10, max_queued_requests=20)
@serve.ingress(app)
class Actor(Base):
    ...
```

命名为 `components/` 后，`from relax.components.actor import Actor` 立即传达了"这是一个可部署的组件"。

#### 3.3.2 `controller/` + `services/` → `core/`

- `controller/` 只有一个文件 `controller.py`
- `services/` 只有一个文件 `service.py`（`Service` 类 —— 封装 Ray Serve 部署生命周期）
- `const.py`（ROLES/ALGOS 注册表）与 Controller 紧密耦合

合并为 `core/` 后：3 个文件构成框架核心，职责清晰。`const.py` 重命名为 `registry.py` —— 它本质是角色→组件类的注册表。

#### 3.3.3 `engine/` 聚合 Rollout 流水线

`slime/rollout/`、`slime/rollout/rm_hub/`、`slime/rollout/filter_hub/`、`slime/router/` 都是数据生成流水线的一部分：

```
请求 → router/ 分发 → rollout/ 生成 → rewards/ 计算奖励 → filters/ 采样过滤
```

归入同一个 `engine/` 目录，体现完整的数据生成流水线。

#### 3.3.4 `ray/` + `checkpoint_engine/` → `distributed/`

两者都是分布式计算基础设施：
- `ray/`：管理 GPU 集群上的 Actor 组和 Placement Group
- `checkpoint/`（原 `checkpoint_engine/`）：跨节点权重同步

合并后减少一个顶层目录，且语义更聚焦。

#### 3.3.5 `metrics/` → `utils/metrics/`

`MetricsService` 本质是基础设施服务（采集指标、转发到 TensorBoard/W&B/ClearML），不是 RL 算法的核心。将其降级为 `utils/` 子模块，同时将散落在 `slime/utils/` 中的监控适配器（`apprise_utils.py`、`clearml_utils.py`、`tensorboard_utils.py`、`wandb_utils.py`）统一归入 `utils/metrics/adapters/`。

#### 3.3.6 `utils/` 子模块化

合并后 `utils/` 将有 **47+ 文件**。通过子模块拆分避免"垃圾桶"：

| 子模块 | 文件数 | 职责边界 |
|--------|--------|---------|
| `metrics/` | 11 | 指标采集、服务、适配器 |
| `data/` | 7 | 数据加载、处理、Tokenizer、Loss Mask |
| `training/` | 7 | PPO 工具、FLOPS、评估配置、训练 Dump |
| `multimodal/` | 5 | 图像/音频/视频处理 |
| `external/` | 2 | CLI 外部工具集成 |
| `debug/` | 1 | 调试工具 |
| 顶层散文件 | ~15 | logging、arguments、types 等基础设施 |

#### 3.3.7 去掉 `_hub` / `_utils` 后缀

| 旧名 | 新名 | 理由 |
|------|------|------|
| `rm_hub/` | `rewards/` | "奖励函数"比"奖励模型 Hub"更直观 |
| `filter_hub/` | `filters/` | 标准命名 |
| `middleware_hub/` | `middleware/` | 标准命名 |
| `megatron_utils/` | `megatron/` | 在 `backends/` 下已隐含 utils 语义 |
| `sglang_utils/` | `sglang/` | 同上 |
| `external_utils/` | `external/` | 在 `utils/` 下已隐含 utils 语义 |
| `debug_utils/` | `debug/` | 同上 |

---

## 4. 死代码分析

### 4.1 确认可删除的文件

以下文件在整个仓库中无任何导入、无脚本引用、无动态加载路径引用：

| 文件 | 行数 | 内容说明 | 判定理由 |
|------|------|---------|---------|
| `slime/rollout/sleep_rollout.py` | 12 | `sleep()` 函数，无限循环 `time.sleep(3600)` | 零导入、零引用；签名不匹配 `generate` 约定 |
| `slime/rollout/generate_hub/benchmarkers.py` | 25 | `generate_with_random_osl()` 基准测试桩函数 | 零导入、零引用 |
| `slime/rollout/generate_hub/__init__.py` | 1 | 仅含 `# TODO: maybe move sglang_rollout::generate to this folder` | 仅 TODO 注释 |
| `relax/utils/swagger_proxy.py` | ~700 | 独立反向代理服务，用于内网 HTTP 转发 | 零导入、零引用；独立工具，不属于训练框架 |
| `slime/rollout/sft_rollout.py` | 64 | `generate_rollout()` 用于 SFT 数据生成 | 零导入、零引用；理论上可通过 `--rollout-function-path` 动态加载但无实际使用 |

### 4.2 确认可删除的目录

| 目录 | 说明 |
|------|------|
| `slime_plugins/` | 整个目录仅含两个空子目录，零 Python 文件，全仓库零导入 |
| `slime/rollout/generate_hub/` | `__init__.py` 仅含 TODO、`benchmarkers.py` 已确认死代码 |

### 4.3 可清理的 `__main__` 测试代码

| 文件 | 行号 | 说明 |
|------|------|------|
| `slime/router/middleware_hub/radix_tree.py` | 620-681 | `if __name__ == "__main__"` 测试块，建议移至 `tests/` |

### 4.4 不是死代码的"可疑"文件

| 文件 | 活跃判定 |
|------|---------|
| `slime/rollout/filter_hub/dynamic_sampling_filters.py` | 通过 `--dynamic-sampling-filter-path` 动态加载 |
| `slime/utils/debug_utils/send_to_sglang.py` | CLI 调试入口（`python -m`） |
| `examples/deploy_metrics_service.py` | 独立部署入口脚本 |
| `slime/rollout/rm_hub/*.py` | 通过 `--reward-function-path` 动态加载 |

### 4.5 处理建议

| 操作 | 对象 | 时机 |
|------|------|------|
| **删除** | `slime/rollout/sleep_rollout.py` | P0 |
| **删除** | `slime/rollout/generate_hub/`（整个目录） | P0 |
| **删除** | `slime/rollout/sft_rollout.py` | P0 |
| **删除** | `relax/utils/swagger_proxy.py` | P0 |
| **删除** | `slime_plugins/`（整个目录） | P0 |
| **清理** | `radix_tree.py` L620-681 `__main__` 块 | P1 |

---

## 5. 变更清单

### P0 — 立即执行（零功能影响）

| # | 变更 |
|---|------|
| 1 | 删除 `slime_plugins/` |
| 2 | 删除死代码（§4.1 所列 5 个文件 + `generate_hub/` 目录） |

### P1 — 目录迁移与重组

| # | 变更 | 影响面 |
|---|------|-------|
| 3 | `relax/controller/` + `relax/services/` + `relax/utils/const.py` → `relax/core/` | `const.py` 重命名为 `registry.py` |
| 4 | `relax/impl/` → `relax/components/` | 7 个文件移动 |
| 5 | `slime/rollout/` → `relax/engine/rollout/` | 不含已删除的死代码 |
| 6 | `slime/rollout/rm_hub/` → `relax/engine/rewards/` | 重命名 |
| 7 | `slime/rollout/filter_hub/` → `relax/engine/filters/` | 重命名 |
| 8 | `slime/router/` → `relax/engine/router/`；`middleware_hub/` → `middleware/` | 重命名 |
| 9 | `slime/backends/megatron_utils/` → `relax/backends/megatron/` | 去掉 `_utils` |
| 10 | `slime/backends/megatron_utils/megatron_to_hf/` → `relax/backends/megatron/weight_conversion/` | 语义化重命名 |
| 11 | `slime/backends/megatron_utils/update_weight/` → `relax/backends/megatron/weight_update/` | 语义化重命名 |
| 12 | `slime/backends/sglang_utils/` → `relax/backends/sglang/` | 去掉 `_utils` |
| 13 | `slime/ray/` → `relax/distributed/ray/` | 8 个文件 |
| 14 | `relax/checkpoint_engine/` → `relax/distributed/checkpoint_service/` | 精简命名 |
| 15 | `relax/metrics/` → `relax/utils/metrics/` | 降级为 utils 子模块 |
| 16 | 监控适配器合并至 `relax/utils/metrics/adapters/` | `apprise_utils.py` → `apprise.py` 等 |
| 17 | `slime/utils/` 按职责拆分合并至 `relax/utils/` 及子模块 | 39 个文件分流 |
| 18 | `slime/utils/logging_utils.py` 删除 | 仅 re-export `relax.utils.logging_utils` |
| 19 | 删除 `slime/` 顶级目录 | 所有内容已迁入 `relax/` |

### P2 — 后续优化（可选）

| # | 变更 | 说明 |
|---|------|------|
| 20 | `radix_tree.py` `__main__` 测试代码迁至 `tests/` | 清理 |
| 21 | 合并 `health_system.py` + `health_monitor.py` | 功能有重叠 |
| 22 | `utils/multimodal/` 提升为 `relax/multimodal/` | 如多模态职责持续增长 |

---

## 6. 导入路径映射表

### 6.1 `relax/` 内部重组

| 旧路径 | 新路径 |
|--------|--------|
| `from relax.controller.controller import Controller` | `from relax.core.controller import Controller` |
| `from relax.services.service import Service` | `from relax.core.service import Service` |
| `from relax.utils.const import ALGOS, ROLES` | `from relax.core.registry import ALGOS, ROLES` |
| `from relax.impl.actor import Actor` | `from relax.components.actor import Actor` |
| `from relax.impl.base import Base` | `from relax.components.base import Base` |
| `from relax.impl.xxx import Xxx` | `from relax.components.xxx import Xxx` |
| `from relax.metrics.client import get_metrics_client` | `from relax.utils.metrics.client import get_metrics_client` |
| `from relax.metrics.service import MetricsService` | `from relax.utils.metrics.service import MetricsService` |
| `from relax.metrics.timeline_trace import ...` | `from relax.utils.metrics.timeline_trace import ...` |
| `from relax.checkpoint_engine.xxx import ...` | `from relax.distributed.checkpoint.xxx import ...` |
| `from relax.utils.stream_dataloader import ...` | `from relax.utils.data.stream_dataloader import ...` |

### 6.2 `slime/` → `relax/` 映射

| 旧路径 | 新路径 |
|--------|--------|
| `from slime.ray.xxx import ...` | `from relax.distributed.ray.xxx import ...` |
| `from slime.utils.async_utils import run` | `from relax.utils.async_utils import run` |
| `from slime.utils.misc import load_function` | `from relax.utils.misc import load_function` |
| `from slime.utils.types import Sample` | `from relax.utils.types import Sample` |
| `from slime.utils.ppo_utils import ...` | `from relax.utils.training.ppo_utils import ...` |
| `from slime.utils.data import ...` | `from relax.utils.data.data import ...` |
| `from slime.utils.processing_utils import ...` | `from relax.utils.data.processing_utils import ...` |
| `from slime.utils.mask_utils import ...` | `from relax.utils.data.mask_utils import ...` |
| `from slime.utils.distributed_utils import ...` | `from relax.utils.distributed_utils import ...` |
| `from slime.utils.apprise_utils import ...` | `from relax.utils.metrics.adapters.apprise import ...` |
| `from slime.utils.clearml_utils import ...` | `from relax.utils.metrics.adapters.clearml import ...` |
| `from slime.utils.tensorboard_utils import ...` | `from relax.utils.metrics.adapters.tensorboard import ...` |
| `from slime.utils.wandb_utils import ...` | `from relax.utils.metrics.adapters.wandb import ...` |
| `from slime.utils.metric_checker import ...` | `from relax.utils.metrics.metric_checker import ...` |
| `from slime.utils.metric_utils import ...` | `from relax.utils.metrics.metric_utils import ...` |
| `from slime.utils.metrics_service_adapter import ...` | `from relax.utils.metrics.metrics_service_adapter import ...` |
| `from slime.utils.timer import ...` | `from relax.utils.timer import ...` |
| `from slime.backends.megatron_utils.xxx import ...` | `from relax.backends.megatron.xxx import ...` |
| `from slime.backends.megatron_utils.megatron_to_hf import ...` | `from relax.backends.megatron.weight_conversion import ...` |
| `from slime.backends.megatron_utils.update_weight.xxx import ...` | `from relax.backends.megatron.weight_update.xxx import ...` |
| `from slime.backends.sglang_utils.xxx import ...` | `from relax.backends.sglang.xxx import ...` |
| `from slime.rollout.sglang_rollout import ...` | `from relax.engine.rollout.sglang_rollout import ...` |
| `from slime.rollout.rm_hub.xxx import ...` | `from relax.engine.rewards.xxx import ...` |
| `from slime.rollout.filter_hub.xxx import ...` | `from relax.engine.filters.xxx import ...` |
| `from slime.router.xxx import ...` | `from relax.engine.router.xxx import ...` |
| `from slime.router.middleware_hub.xxx import ...` | `from relax.engine.router.middleware.xxx import ...` |

### 6.3 脚本动态加载路径

| 旧路径 | 新路径 |
|--------|--------|
| `--rollout-function-path slime.rollout.xxx` | `--rollout-function-path relax.engine.rollout.xxx` |
| `--reward-function-path slime.rollout.rm_hub.xxx` | `--reward-function-path relax.engine.rewards.xxx` |
| `--dynamic-sampling-filter-path slime.rollout.filter_hub.xxx` | `--dynamic-sampling-filter-path relax.engine.filters.xxx` |
| `--custom-generate-function-path slime.rollout.xxx` | `--custom-generate-function-path relax.engine.rollout.xxx` |

---

## 7. 实施策略

### Phase 1: 清理死代码（P0，~0.5 天）

1. 删除 `slime_plugins/`
2. 删除死代码文件（§4.1 + §4.2）
3. `pre-commit run --all-files` 验证

### Phase 2: relax/ 内部重组（~1 天）

按依赖从低到高重组（避免中间态编译失败）：

1. `relax/metrics/` → `relax/utils/metrics/`
2. `relax/utils/const.py` → `relax/core/registry.py`
3. `relax/controller/` + `relax/services/` → `relax/core/`
4. `relax/impl/` → `relax/components/`
5. `relax/checkpoint_engine/` → `relax/distributed/checkpoint_service/`
6. 全量搜索替换内部导入路径
7. `pre-commit run --all-files` + `pytest tests/`

### Phase 3: slime/ → relax/ 迁移（~3 天）

按依赖从叶子到根迁移：

1. `slime/utils/` → `relax/utils/`（合并 + 子模块化）
2. `slime/backends/` → `relax/backends/`
3. `slime/ray/` → `relax/distributed/ray/`
4. `slime/router/` → `relax/engine/router/`
5. `slime/rollout/` → `relax/engine/`
6. 删除 `slime/` 目录
7. 全量搜索替换 `from slime.` → `from relax.`
8. 更新所有 `.sh` 脚本中的动态加载路径
9. 更新 `setup.py` / `pyproject.toml` 包声明
10. `pre-commit run --all-files` + `pytest tests/`

### Phase 4: 文档与配置更新（~1 天）

1. 更新 `AGENTS.md`
2. 更新 `README.md` / `README_zh.md`
3. 更新 `docs/` 中的所有 API 文档和指南
4. 更新 `skills/` 中的引用

---

## 8. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| `from slime.` 导入遗漏 | **高** | 全量 grep 验证；CI 中添加 `grep -r "from slime\." relax/` 断言 |
| 动态加载路径断裂 | **高** | 搜索所有 `.sh` 脚本和 `arguments.py` 中的 `slime.` 引用；可选：提供兼容层 |
| `setup.py` 包声明遗漏 | 中 | `pip install -e .` 后验证所有 `from relax.` 导入 |
| 外部用户脚本引用 `slime.` | 中 | 迁移指南 + 可选 `slime/__init__.py` deprecation warning |
| `relax/utils/` 文件名冲突 | **低** | 仅 `logging_utils.py` 存在于两处，`slime/` 版本可直接删除 |

---

## 9. 前后对比

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 顶层包数 | 3 (`relax/`, `slime/`, `slime_plugins/`) | 1 (`relax/`) |
| `relax/` 下同级目录 | 10 | **6** |
| 循环依赖 | `relax/` ⇄ `slime/`（62 处） | **无** |
| `utils/` 平铺文件 | 47+（合并后） | **~15**（其余按职责分到子模块） |
| 死代码文件 | 5 个 + 2 个空目录 | **0** |
| `_hub` / `_utils` 冗余后缀 | 7 处 | **0** |
| `impl/`、`services/` 等模糊命名 | 3 处 | **0** |
