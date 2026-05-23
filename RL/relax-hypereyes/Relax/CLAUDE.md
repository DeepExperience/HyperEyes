# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Relax is a high-performance RL post-training engine for multi-modal and agentic AI, built on Ray Serve, Megatron-LM, and SGLang. It supports algorithms like GRPO, DAPO, GSPO, SAPO, and On-Policy Distillation across Qwen3/Qwen3-VL/Qwen3-Omni models.

## Common Commands

```bash
pip install -e .                      # Install package in dev mode
pip install -r requirements.txt       # Install dependencies
make format                           # Run pre-commit (ruff, isort, mdformat, docformatter)
make lint                             # Run flake8 + mypy on relax/
make test                             # Run pytest tests/
pytest tests/test_foo.py::test_bar    # Run a single test
pre-commit run --all-files            # Lint + format (must run before commits)
make docs-dev                         # Start VitePress docs dev server
```

## Architecture

Three-layer service-oriented architecture on Ray Serve:

1. **Controller** (`relax/core/controller.py`) - Top-level training loop, service orchestration, health monitoring
2. **Service** (`relax/core/service.py`) - Lifecycle management, GPU placement groups, Ray Serve deployment wrapper
3. **Components** (`relax/components/`) - Concrete RL services: Actor, Critic, Rollout, Advantages, GenRM, ActorFwd

Two execution modes:

- **Colocate (Sync)** - Actor & Rollout time-share same GPUs
- **Fully Async** - Independent GPU clusters per role with streaming data via TransferQueue

Key subsystems:

- **Backends** (`relax/backends/`) - Megatron training backend (TP/PP/CP/EP) and SGLang inference engine
- **Engine** (`relax/engine/`) - Rollout generation, pluggable reward functions (`engine/rewards/`), data filtering, request routing
- **Distributed** (`relax/distributed/`) - Ray cluster management, DCS (NCCL broadcast-based weight sync)
- **Algorithm Registry** (`relax/core/registry.py`) - Maps algorithm names to component roles

Entry point: `python relax/entrypoints/train.py [args]` with CLI args defined in `relax/utils/arguments.py` (extends Megatron-LM parser).

## Code Standards

- **Ruff** formatting, line width 119, double quotes, `isort` with `relax` as first-party
- Copyright header required on all `relax/` Python files: `# Copyright (c) 2026 Relax Authors. All Rights Reserved.`
- Logging: `relax.utils.logging_utils.get_logger(__name__)` only - no `print()` or `logging.getLogger()`
- No wildcard imports; explicit type annotations; heavy optional deps imported inside functions
- No GPU-CPU sync in hot paths (`.item()`, `.tolist()`, tensor printing)
- Composition over inheritance, max 2 levels deep
- Test naming: `test_<module>_<behavior>()`; GPU tests use `@pytest.mark.skipif`

## Hard Rules

- Run `pre-commit run --all-files` before committing
- Minimum change principle: only touch files/lines directly involved
- No formatting-only or unrelated code changes
- No hardcoded secrets, paths, or endpoints

## Distributed Code Rules

Scope: `relax/backends/**`, `relax/distributed/ray/**`, `relax/distributed/checkpoint_service/**`

- No global process groups at module level; always pass `process_group` explicitly
- Use `dist.get_rank(group)` not `dist.get_rank()`
- All ranks must call all-reduce; broadcast requires explicit `src`; barriers only for debugging

## Ask First

Confirm with user before modifying:

- `relax/utils/arguments.py` or `relax/backends/megatron/arguments.py` parameter parsing
- Adding new dependencies
- Controller / Service / Launcher logic
- Deleting or renaming public APIs

When unsure, leave `TODO(agent)` comment and explain constraints.

## Commits

Follow `skills/git-commit/SKILL.md`. Only create local commits, never push.
