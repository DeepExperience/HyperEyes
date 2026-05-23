# Docker Release Guidelines

## Overview

We publish two types of Docker images for Relax:

### 1. Stable Version

- Based on specific SGLang releases
- Patches are stored and maintained for these versions
- Recommended for production use

### 2. Latest Version

- Aligns with `lmsysorg/sglang:latest`
- Contains the most recent features and improvements
- Recommended for development and testing

## Pre-Release Testing

Before each update, we perform comprehensive testing on the following models using H100 GPUs:

| Model              | Sync | Async |
| ------------------ | ---- | ----- |
| Qwen3-4B           | ✓    | ✓     |
| Qwen3-30B-A3B      | ✓    | ✓     |
| Qwen3-omni-30B-A3B | ✓    | ✓     |
| Qwen3.5-35B-A3B    | ✓    | ✓     |

## Testing Modes

- **Sync**: Synchronous training mode
- **Async**: Asynchronous training mode

All models are tested in both modes to ensure stability and compatibility across different training scenarios.
