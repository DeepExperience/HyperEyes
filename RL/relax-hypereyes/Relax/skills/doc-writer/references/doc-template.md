# Doc Page Template

This file provides the standard template for Relax documentation pages. Both English and Chinese versions follow this structure.

## English Template

```markdown
# Feature Name

Brief one-line description of the feature.

## Overview

2-3 paragraph explanation of what this feature does, why it exists, and how it fits into the Relax architecture.

## Architecture

(Optional — include when the feature has meaningful internal structure)

Use ASCII art diagrams:

\```
┌─────────────────┐         ┌─────────────────┐
│   Component A   │ ──────> │   Component B   │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Component C   │
                            └─────────────────┘
\```

### Component Responsibilities

| Component | Responsibility | Implementation |
|---|---|---|
| **ComponentA** | What it does | How it's implemented |
| **ComponentB** | What it does | How it's implemented |

## Features

1. **Feature A**: Description
2. **Feature B**: Description
3. **Feature C**: Description

## Quick Start

### 1. Setup / Deploy

\```python
from relax.module import ClassName

# Minimal working example
instance = ClassName(required_param="value")
instance.start()
\```

### 2. Basic Usage

\```python
# Show the most common usage pattern
result = instance.do_something(input_data)
\```

## Configuration

\```yaml
feature_name:
  enabled: true
  param_a: "value"
  param_b: 100
\```

Or via CLI arguments:

\```bash
python train.py --param-a value --param-b 100
\```

## API Reference

### ClassName

#### method_name

\```python
instance.method_name(
    param_a: str,
    param_b: int = 10,
    param_c: Optional[float] = None
) -> ReturnType
\```

Description of what the method does.

**Parameters:**
- `param_a` — description
- `param_b` — description (default: 10)
- `param_c` — description (default: None)

**Returns:** Description of return value.

## Usage Examples

### Example 1: Common Scenario

\```python
# Realistic example with context
\```

### Example 2: Advanced Scenario

\```python
# More complex usage
\```

## Best Practices

1. **Practice A**: Explanation
2. **Practice B**: Explanation
3. **Practice C**: Explanation

## Troubleshooting

### Problem A

Check:
1. First thing to verify
2. Second thing to verify
3. Third thing to verify

### Problem B

If symptom occurs:
- Cause and fix

## Next Steps

- [Related Feature A](./related-a.md) — Brief description
- [Related Feature B](./related-b.md) — Brief description
- [Configuration](./configuration.md) — How to configure this feature
```

## Chinese Template

The Chinese version mirrors the English structure exactly. Key differences:

```markdown
# 功能名称

功能的简短一行描述。

## 概述

2-3 段解释该功能做什么、为什么存在、如何融入 Relax 架构。

## 架构

(可选 — 当功能有有意义的内部结构时包含)

\```
┌─────────────────┐         ┌─────────────────┐
│   组件 A        │ ──────> │   组件 B          │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   组件 C        │
                            └─────────────────┘
\```

## 功能特性

1. **功能 A**：描述
2. **功能 B**：描述

## 快速开始

### 1. 部署 / 安装

\```python
from relax.module import ClassName

# 最小可运行示例
instance = ClassName(required_param="value")
instance.start()
\```

## 配置

(同英文版，代码块保持不变，注释翻译为中文)

## API 参考

(同英文版，签名保持不变，描述翻译为中文)

## 使用示例

(同英文版，代码保持不变，注释和说明翻译为中文)

## 最佳实践

1. **实践 A**：说明
2. **实践 B**：说明

## 故障排除

### 问题 A

检查：
1. 第一个要验证的事项
2. 第二个要验证的事项

## 下一步

- [相关功能 A](./related-a.md) — 简短描述
- [相关功能 B](./related-b.md) — 简短描述
```

## Conventions

### Code Blocks

- Always specify language: ` ```python `, ` ```bash `, ` ```yaml `, ` ```typescript `
- Code must use **real** import paths and function signatures from the codebase
- Comments in English docs are in English; comments in Chinese docs are in Chinese

### VitePress Containers

```markdown
::: tip Tip Title
Helpful tip content
:::

::: warning Warning Title
Warning content
:::

::: danger Danger Title
Dangerous/critical content
:::
```

Chinese equivalents:
```markdown
::: tip 提示
提示内容
:::

::: warning 警告
警告内容
:::

::: danger 危险
危险/关键内容
:::
```

### Internal Links

- Use relative paths: `[Architecture](./architecture.md)`
- For cross-category links: `[API Overview](../api/overview.md)`
- Never use absolute URLs for internal docs

### Tables

Use standard Markdown tables with alignment:
```markdown
| Column A | Column B | Column C |
|---|---|---|
| value | value | value |
```

### ASCII Diagrams

Use box-drawing characters for architecture diagrams:
- Corners: `┌ ┐ └ ┘`
- Lines: `─ │`
- Arrows: `▼ ▲ ► ◄ ──>`
- Intersections: `┬ ┴ ├ ┤ ┼`
