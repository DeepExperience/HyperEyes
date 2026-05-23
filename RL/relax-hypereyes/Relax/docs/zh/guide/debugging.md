# 调试指南

用于排查 Relax 训练中精度问题和隔离训练/推理组件的实用指南。

## 精度排查

### 1. 检查 Rollout 生成的 Response

在日志中搜索 `Finish rollout`，定位当前的 rollout ID 和生成的 response。首先判断 response 是否为正常的、连贯的回复。

- **如果第一步就出现了乱码/胡言乱语**，一般是由于 checkpoint 加载异常或模型转换异常。一个比较彻底的方法是在对应模型的 SGLang `load_weights` 实现中保存所有参数，查看与加载的 checkpoint 中是否一致。如果所有参数更新都正确还出现问题，有可能是 SGLang 里有一些特殊的 buffer 在 `release_memory_occupation` 阶段被释放了。如果是用 pretrain 模型进行的测试，可以换成同结构模型的 instruct 版本，查看这种乱码是不是 pretrain 模型特有的。

- **如果第一步回复正常，但后续 step 出现乱码**，说明训崩了。需要更细致地去检查每一步的 reward 计算是否符合预期、超参数配置等因素。

### 2. 检查 log_probs 和 ref_log_probs

查看第一步打印的 rollout stats，确认 `log_probs` 和 `ref_log_probs` 是否完全相等（即第一步 KL = 0），且值较小。

- **如果不是完全相等**，一般是 Transformer Engine 中的某些非确定性 kernel 导致的。例如，在某些版本的 TE 里，Megatron 需要 `--attention-backend flash` 来强制使用 Flash Attention，从而避免 Context Parallelism (CP) 下 fused attention 的数值不稳定。

- **如果数值较大**（例如 > 1），一般有两种可能：
  - 值非常大，应该是训练配置有问题。
  - 值只是比 SFT loss 的状态略大（例如 instruct 模型的 logprob 到了 0.8），有可能是数据不符合训练的 chat template，或者不符合冷启动的分布。

### 3. 检查第一步的 KL 和 grad_norm

在推一训一（`num_steps_per_rollout == 1`）的配置下，检查第一步的 KL 是否为 0，`grad_norm` 是否较小。

此阶段的问题基本上是 Megatron / Transformer Engine 相关的 bug。例如：

- MoE 模型需要开启 `--moe-permute-fusion`。

## 训练推理单独调试

Relax 支持将训练和推理组件分开独立运行，从而实现：

- 在调试推理部分时，只用少量卡就可以启动任务。
- 在调试训练部分时，可以保证模型输入固定，去除 rollout 的随机性。

### 可用的调试参数

以下 CLI 参数用于开启隔离调试：

| 参数 | 描述 |
|---|---|
| `--debug-rollout-only` | 仅初始化 SGLang（跳过 Megatron）。用于推理调试。 |
| `--debug-train-only` | 仅初始化 Megatron（跳过 SGLang）。用于训练调试。 |
| `--save-debug-rollout-data <path>` | 将 rollout 结果保存到指定路径，供后续回放使用。 |
| `--load-debug-rollout-data <path>` | 从指定路径加载 rollout 数据。自动设置 `--debug-train-only`。 |
| `--dump-details <dir>` | 保存训练的全部细节（自动开启 rollout 数据保存）。 |

### 工作流 1：仅调试推理

使用 `--debug-rollout-only` 跳过 Megatron 初始化。只会启动 SGLang 引擎，可以用更少的 GPU 来测试推理部分。

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    # ... 其他 rollout 参数
```

可以配合 `--save-debug-rollout-data` 来保存 rollout 结果：

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他 rollout 参数
```

### 工作流 2：仅调试训练

使用 `--load-debug-rollout-data` 加载预先保存的 rollout 数据，仅运行训练流程。这会自动设置 `debug_train_only=True`，因此不会初始化 SGLang。

```bash
python3 relax/entrypoints/train.py \
    --load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他训练参数
```

这种方式特别适用于：

- 在不等待 rollout 的情况下调优并行配置（TP、PP、EP、CP）。
- 用确定性输入来复现和修复训练相关的问题。
- 快速迭代 loss 计算或优化器的修改。

### 工作流 3：全量细节转储

使用 `--dump-details` 保存训练的全部细节供事后分析。设置后会自动启用：

- `--save-debug-rollout-data`，保存路径为 `<dir>/rollout_data/{rollout_id}.pt`
- `--save-debug-train-data`，保存路径为 `<dir>/train_data/{rollout_id}_{rank}.pt`

```bash
python3 relax/entrypoints/train.py \
    --dump-details /path/to/dump/dir \
    # ... 其他参数
```

::: tip 提示
`--dump-details` 在收集 bug 报告数据时也非常有用 — 它会捕获复现问题所需的所有信息。
:::
