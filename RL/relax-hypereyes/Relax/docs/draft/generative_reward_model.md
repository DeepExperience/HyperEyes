# GenRM (生成式奖励模型)

## 概述

GenRM (Generative Reward Model，生成式奖励模型) 是一种新型的奖励建模方法。与传统奖励模型不同，GenRM 直接使用大语言模型 (LLM) 来评估不同响应之间的偏好，无需训练单独的奖励模型。

这种方法的核心思想是：利用 LLM 强大的推理和理解能力，直接判断哪个回答更符合人类偏好或任务要求。相比于传统奖励模型，GenRM 具有以下优势：

- **零训练成本**：无需额外训练奖励模型，直接使用预训练 LLM
- **泛化能力强**：基于强大的 LLM 能力，对未见过的任务也有良好表现
- **灵活可控**：可以通过提示词 (prompt) 调整评估标准

## 架构

GenRM 在 Relax 框架中作为独立的 Ray Serve 服务运行，通过 HTTP 接口与其他组件交互。

```
┌──────────────────┐
│  Rollout (4 GPU) │ ──┐
└──────────────────┘   │
                       │  Reward
┌──────────────────┐   │  Request
│  GenRM (4 GPU)   │ ◄─┘
└──────────────────┘
        │
        │ 奖励分数
        ▼
┌──────────────────┐
│  Training (8 GPU)│  (Rollout 阶段卸载)
└──────────────────┘
```

### 核心组件

| 组件          | 路径                                 | 描述                           |
| ------------- | ------------------------------------ | ------------------------------ |
| GenRM Service | `relax/components/genrm.py`                | Ray Serve 部署，提供 HTTP 端点 |
| GenRM Manager | `relax/distributed/ray/genrm.py`           | 管理 SGLang 引擎的生命周期     |
| GenRM Client  | `relax/utils/genrm_client.py`              | HTTP 客户端，供奖励模型调用    |
| DAPO-GenRM    | `relax/engine/rewards/dapo_genrm.py`       | DAPO 训练流程的集成实现        |

### 服务端点

| 端点        | 方法 | 描述         |
| ----------- | ---- | ------------ |
| `/generate` | POST | 生成评估结果 |
| `/health`   | GET  | 健康检查     |
| `/metrics`  | GET  | 服务指标     |

## 快速开始

### 1. 启动训练

使用提供的脚本启动带 GenRM 的 DAPO 训练：

```bash
bash scripts/training/genrm/run-qwen3-4B-8xgpu-genrm.sh
```

该脚本配置如下资源：

- **4 GPU** 用于 Rollout 生成
- **4 GPU** 用于 GenRM 评估
- **8 GPU** 用于训练（共存模式，含卸载）

### 2. 健康检查

```bash
curl http://localhost:8000/genrm/health
```

## 配置参数

### GenRM 参数

| 参数                          | 默认值 | 描述                               |
| ----------------------------- | ------ | ---------------------------------- |
| `--genrm-model-path`          | None   | GenRM 模型路径（设置后启用 GenRM） |
| `--genrm-num-gpus`            | 1      | GenRM 使用的 GPU 总数              |
| `--genrm-num-gpus-per-engine` | 1      | 每个引擎使用的 GPU 数量            |
| `--genrm-engine-config`       | None   | 引擎初始化 JSON 配置               |
| `--genrm-sampling-config`     | None   | 采样参数 JSON 配置                 |

### 资源分配

**Colocated 模式**：

```bash
--genrm-engine-config '{"max_context_len": 8192}'
--genrm-sampling-config '{"temperature": 0.1, "max_response_len": 4096}'
--resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}'
--colocate
```

**Fully-Async 模式** (需要更多 GPU)：

```bash
--resource '{"actor": [1, 2], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0], "genrm": [1, 2]}'
--fully-async
```

## 使用方法

### 启用 GenRM 奖励模型

在训练命令中添加：

```bash
--genrm-model-path /path/model/model
--genrm-engine-config '{"max_context_len": 8192}'
--genrm-sampling-config '{"temperature": 0.1, "max_response_len": 4096}'
--rm-type dapo-genrm
```

这将使用 `dapo-genrm` 奖励模型实现，该实现会调用 GenRM 服务进行偏好评估。

### API 调用

#### 生成请求

```bash
curl -X POST http://localhost:8000/genrm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "比较以下两个回答：A) 答案A B) 答案B"}
    ]
  }'
```

响应格式：

```json
{
  "response": "评估结果..."
}
```

## 故障排除

### GenRM 未启用

确保设置了 `--genrm-model-path` 参数。只有当该参数不为 None 时，GenRM 才会启用。

### 资源不够

调整 `--resources` 参数

## 参考资料

- [GenRM 原始论文](https://arxiv.org/abs/2410.12832)
- [SGLang 文档](https://github.com/sgl-project/sglang)
- [示例代码](../examples/genrm/README.md)
