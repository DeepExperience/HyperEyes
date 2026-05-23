# Apprise Notification Integration

Slime integrates the [Apprise](https://github.com/caronc/apprise) notification library, supporting sending training progress and alerts to various notification services.

## Features

### 1. Training Startup Notification

When training starts, a startup notification is sent containing the project name and experiment name.

### 2. Training Metrics Updates

Periodically sends training metrics reports, including:

- Current training step
- Current values of key metrics
- Changes compared to the previous update (with sign)

Tracked key metrics:

- `rollout/raw_reward` - Raw reward
- `rollout/reward` - Processed reward
- `train/grad_norm` - Gradient norm
- `train/entropy_loss` - Entropy loss
- `train/policy_loss` - Policy loss
- `train/value_loss` - Value loss
- `train/learning_rate` - Learning rate
- `train/kl_divergence` - KL divergence

### 3. Training Completion Notification

When training completes normally, a completion notification is sent with a summary of final metrics.

### 4. Exception Alerts

If training exits abnormally (without calling finish properly), an alert notification is automatically sent.

## Usage

### Install Dependencies

```bash
pip install apprise
```

### Configure Notification URLs

Specify one or more notification service URLs using the `--notify-urls` parameter (comma-separated):

```bash
python relax/entrypoints/train.py \
    --notify-urls "redcity://your_webhook_key?msgtype=markdown&freq=10" \
    ... other training parameters ...
```

### Multiple Notification Services

You can configure multiple notification services simultaneously:

```bash
--notify-urls "redcity://webhook1?msgtype=markdown,mailto://user:pass@gmail.com,slack://tokenA/tokenB/tokenC"
```

## Supported Notification Services

Apprise supports 80+ notification services, including:

### Enterprise Internal Services

- **RedCity**: `redcity://webhook_key?msgtype=markdown&freq=10`

### Email Services

- **Gmail**: `mailto://user:pass@gmail.com`
- **Outlook**: `mailto://user:pass@outlook.com`

### Instant Messaging

- **Slack**: `slack://tokenA/tokenB/tokenC`
- **Discord**: `discord://webhook_id/webhook_token`
- **Microsoft Teams**: `msteams://TokenA/TokenB/TokenC`
- **Telegram**: `tgram://bot_token/chat_id`

### Other Services

- **Webhook**: `json://hostname/path` or `xml://hostname/path`
- For more services, see [Apprise Documentation](https://github.com/caronc/apprise/wiki)

## Notification Format Examples

### Training Startup Notification

```markdown
# 🚀 Training Started

**Project**: MyProject
**Experiment**: Experiment-001
**Status**: Started

---
*Training metrics will be updated periodically*
```

### Training Metrics Report

```markdown
# 📊 Training Report - Step 100

- **Raw Reward**: 0.75 (+0.05)
- **Grad Norm**: 1.23 (+0.15)
- **Entropy Loss**: 0.45 (-0.02)
- **Policy Loss**: 0.32 (-0.01)
```

### Training Completion Notification

```markdown
# ✅ Training Completed

**Status**: Training completed successfully

**Final Metrics**:

- Raw Reward: 0.85
- Grad Norm: 1.10
- Entropy Loss: 0.40
- Policy Loss: 0.28
```

### Exception Alert

```markdown
# ⚠️ Training Terminated Abnormally

**Status**: Training process exited abnormally

Please check logs for detailed information.
```

## Advanced Usage

### Custom Alerts

You can send custom alerts using the `send_alert` method of `_AppriseAdapter`:

```python
from relax.utils.metrics.adapters.apprise import _AppriseAdapter

adapter = _AppriseAdapter(args)
adapter.send_alert(
    title="Custom Alert",
    message="Abnormal situation detected, please check",
    level="warning"  # info, warning, error
)
```

### Code Integration

```python
from relax.utils.tracking_utils import init_tracking, log, finish_tracking

# Initialize (including Apprise)
init_tracking(args)

# Log metrics (will automatically send notifications)
log(args, {
    "step": 100,
    "rollout/raw_reward": 0.75,
    "train/grad_norm": 1.23,
}, step_key="step")

# Call when training completes
finish_tracking(args)
```

## Configuration Recommendations

### Frequency Control

For high-frequency training (multiple updates per second), it is recommended to:

1. Use the `freq` parameter to control notification frequency (e.g., `freq=10` means send 1 out of every 10)
2. Or control the call frequency at the application level

### Message Format

- Services that support Markdown like RedCity should use `msgtype=markdown`
- Plain text services will automatically convert the format

### Security

- Do not hardcode sensitive information (such as passwords, tokens) in code
- It is recommended to pass them through environment variables or configuration files
- Example: `--notify-urls "$NOTIFY_URL"`

## Troubleshooting

### Notifications Not Sent

1. Check if Apprise is installed: `pip list | grep apprise`
2. Check logs for error messages
3. Verify that the notification URL format is correct
4. Test if the notification service is reachable

### Notification Frequency Too High

- Use the `freq` parameter to limit frequency
- Or add conditional checks in code to send only at specific steps

### Format Issues

- Ensure the notification service supports Markdown (like RedCity)
- For services that don't support it, Apprise will automatically downgrade to plain text

## References

- [Apprise Official Documentation](https://github.com/caronc/apprise)
- [Supported Services List](https://github.com/caronc/apprise/wiki)
- [URL Format Documentation](https://github.com/caronc/apprise/wiki/URLBasics)
