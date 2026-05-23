## Design

### Project Structure

```
relax/                       核心框架
├── core/                    编排层 — 训练循环、服务基类、全局注册表
│   ├── controller.py        Controller：部署编排、训练循环、健康管理
│   ├── service.py           Service：Ray Serve 部署生命周期封装
│   └── registry.py          全局常量与角色注册表
├── components/              组件层 — RL 服务组件（Ray Serve Deployment）
│   ├── actor.py             策略训练
│   ├── actor_fwd.py         前向推理 log-prob
│   ├── critic.py            价值估计
│   ├── advantages.py        优势计算
│   ├── genrm.py             生成式奖励模型
│   └── rollout.py           Rollout 服务编排
├── engine/                  引擎层 — Rollout 数据生成、奖励计算、请求路由
│   ├── rollout/             Rollout 引擎实现（SGLang rollout、on-policy distillation 等）
│   └── rewards/             奖励函数（deepscaler、math_utils、genrm 等）
├── backends/                后端层 — 训练后端与推理引擎
│   ├── megatron/            Megatron 训练后端（Actor、权重更新、HF 转换等）
│   └── sglang/              SGLang 推理引擎（引擎管理、进程清理）
├── distributed/             分布式层 — Ray 集群管理、分布式 Checkpoint
│   ├── ray/                 Ray Actor 组管理（rollout、genrm、训练组等）
│   └── checkpoint_service/  分布式 Checkpoint 服务（DCS）
├── entrypoints/             入口层 — 训练入口脚本
│   ├── train.py             主训练入口（信号处理、SGLang 清理）
│   └── deploy_metrics_service.py  独立指标服务部署
└── utils/                   基础设施 — 工具函数、指标监控、多模态处理
    ├── arguments.py         命令行参数解析
    ├── data/                数据集加载与流式处理
    ├── metrics/             指标采集与上报（WandB、Prometheus、Apprise）
    ├── training/            训练工具（tensor 备份等）
    └── ...                  日志、定时器、健康监控等
```

### Supporting Directories

- `transfer_queue/` — 分布式数据传输队列（fully-async 模式下组件间通信）
- `examples/` — 用户级示例（deepeyes、OPD 等）
- `scripts/` — 训练启动脚本与模型配置
- `configs/env.yaml` — 运行时环境配置
- `tests/` — 测试用例
- `requirements.txt` — 依赖管理
