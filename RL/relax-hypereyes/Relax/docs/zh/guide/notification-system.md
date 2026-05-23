# 通知系统

Relax 集成了 [Apprise](https://github.com/caronc/apprise) 通知库，支持向各种通知服务发送训练进度和警报。

## 功能特性

### 1. 训练启动通知

训练开始时，发送包含项目名称和实验名称的启动通知。

### 2. 训练指标更新

定期发送训练指标报告，包括：

- 当前训练步数
- 关键指标的当前值
- 与上次更新相比的变化（带符号）

跟踪的关键指标：

- `rollout/raw_reward` - 原始奖励
- `rollout/reward` - 处理后的奖励
- `train/grad_norm` - 梯度范数
- `train/entropy_loss` - 熵损失
- `train/policy_loss` - 策略损失
- `train/value_loss` - 价值损失
- `train/learning_rate` - 学习率
- `train/kl_divergence` - KL 散度

### 3. 训练完成通知

训练正常完成时，发送包含最终指标摘要的完成通知。

### 4. 异常警报

如果训练异常退出（未正确调用 finish），自动发送警报通知。

## 使用方法

### 安装依赖

```bash
pip install apprise
```

### 配置通知 URL

使用 `--notify-urls` 参数指定一个或多个通知服务 URL（逗号分隔）：

```bash
python relax/entrypoints/train.py \
    --notify-urls "redcity://your_webhook_key?msgtype=markdown&freq=10" \
    ... 其他训练参数 ...
```

### 多个通知服务

您可以同时配置多个通知服务：

```bash
--notify-urls "redcity://webhook1?msgtype=markdown,mailto://user:pass@gmail.com,slack://tokenA/tokenB/tokenC"
```

## 支持的通知服务

Apprise 支持 80+ 种通知服务，包括：

### 企业内部服务

- **RedCity**: `redcity://webhook_key?msgtype=markdown&freq=10`

### 邮件服务

- **Gmail**: `mailto://user:pass@gmail.com`
- **Outlook**: `mailto://user:pass@outlook.com`

### 即时通讯

- **飞书 (Feishu/Lark)**: `feishu://token_id/token_secret`
- **Slack**: `slack://tokenA/tokenB/tokenC`
- **Discord**: `discord://webhook_id/webhook_token`
- **Microsoft Teams**: `msteams://TokenA/TokenB/TokenC`
- **Telegram**: `tgram://bot_token/chat_id`

### 其他服务

- **Webhook**: `json://hostname/path` 或 `xml://hostname/path`
- 更多服务，请参见 [Apprise 文档](https://github.com/caronc/apprise/wiki)

## 通知格式示例

### 训练启动通知

```markdown
# 🚀 Training Started

**Project**: MyProject
**Experiment**: Experiment-001
**Status**: Started

---
*Training metrics will be updated periodically*
```

### 训练指标报告

```markdown
# 📊 Training Report - Step 100

- **Raw Reward**: 0.75 (+0.05)
- **Grad Norm**: 1.23 (+0.15)
- **Entropy Loss**: 0.45 (-0.02)
- **Policy Loss**: 0.32 (-0.01)
```

### 训练完成通知

```markdown
# ✅ Training Completed

**Status**: Training completed successfully

**Final Metrics**:

- Raw Reward: 0.85
- Grad Norm: 1.10
- Entropy Loss: 0.40
- Policy Loss: 0.28
```

### 异常警报

```markdown
# ⚠️ Training Terminated Abnormally

**Status**: Training process exited abnormally

Please check logs for detailed information.
```

## 飞书接入示例

### 1. 创建飞书自定义机器人

1. 在飞书群聊中，点击 **设置** → **群机器人** → **添加机器人** → **自定义机器人**
2. 复制 Webhook URL，格式如：`https://open.feishu.cn/open-apis/bot/v2/hook/{token}`
3. （可选）配置签名校验，获取 `token_secret`

### 2. 配置通知 URL

```bash
# 无签名校验
python relax/entrypoints/train.py \
    --notify-urls "feishu://{token}" \
    ...

# 有签名校验
python relax/entrypoints/train.py \
    --notify-urls "feishu://{token}/{token_secret}" \
    ...
```

### 3. 与其他通知服务组合

```bash
--notify-urls "feishu://{token}/{token_secret}?freq=10,redcity://webhook_key?msgtype=markdown"
```

## 配置建议

### 频率控制

对于高频训练（每秒多次更新），建议：

1. 使用 `freq` 参数控制通知频率（例如，`freq=10` 表示每 10 次发送 1 次）
2. 或在应用层面控制调用频率

### 消息格式

- 支持 Markdown 的服务（如 RedCity）应使用 `msgtype=markdown`
- 纯文本服务将自动转换格式

### 安全性

- 不要在代码中硬编码敏感信息（如密码、令牌）
- 建议通过环境变量或配置文件传递
- 示例：`--notify-urls "$NOTIFY_URL"`

## 故障排除

### 通知未发送

1. 检查 Apprise 是否已安装：`pip list | grep apprise`
2. 检查日志中的错误消息
3. 验证通知 URL 格式是否正确
4. 测试通知服务是否可达

### 通知频率过高

- 使用 `freq` 参数限制频率
- 或在代码中添加条件检查，仅在特定步骤发送

### 格式问题

- 确保通知服务支持 Markdown（如 RedCity）
- 对于不支持的服务，Apprise 将自动降级为纯文本

## 参考资源

- [Apprise 官方文档](https://github.com/caronc/apprise)
- [支持的服务列表](https://github.com/caronc/apprise/wiki)
- [URL 格式文档](https://github.com/caronc/apprise/wiki/URLBasics)
