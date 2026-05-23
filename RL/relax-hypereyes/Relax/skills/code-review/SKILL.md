---
name: code-review
description: Expert code review of current git changes with a senior engineer 
  lens. Detects SOLID violations, security risks, Python anti-patterns, and 
  ML/distributed training issues. Tailored for the Relax reinforcement learning 
  framework.
---

# Code Review Expert

Perform a structured review of the current git changes with focus on SOLID principles, architecture, removal candidates, security risks, and Python-specific issues. Tailored for the Relax project (PyTorch distributed training, Ray actors, ML pipeline code).

Default to review-only output unless the user asks to implement changes.

For project structure, coupling points, and key files, see `AGENTS.md`.

## Severity Levels

| Level  | Name     | Description                                                                       | Action                             |
| ------ | -------- | --------------------------------------------------------------------------------- | ---------------------------------- |
| **P0** | Critical | Security vulnerability, data loss risk, correctness bug, training corruption      | Must block merge                   |
| **P1** | High     | Logic error, significant SOLID violation, performance regression, gradient issues | Should fix before merge            |
| **P2** | Medium   | Code smell, maintainability concern, minor SOLID violation, missing type hints    | Fix in this PR or create follow-up |
| **P3** | Low      | Style, naming, documentation, minor suggestion                                    | Optional improvement               |

______________________________________________________________________

## Workflow

### 1) Preflight Context

- Use `git status -sb`, `git diff --stat`, and `git diff` to scope changes.
- If needed, use `rg` or `grep` to find related modules, usages, and contracts.
- Identify entry points, ownership boundaries, and critical paths (training loop, loss computation, checkpoint saving).

**Edge cases:**

- **No changes**: Ask if they want to review staged changes or a specific commit range.
- **Large diff (>500 lines)**: Summarize by file first, then review in batches.
- **Cross-package changes**: When multiple `relax/` subpackages are modified, verify import compatibility first.

### 2) SOLID + Architecture Smells

- Load `references/solid-checklist.md` for specific prompts.
- When you propose a refactor, explain *why* it improves cohesion/coupling and outline a minimal, safe split.
- If refactor is non-trivial, propose an incremental plan instead of a large rewrite.

### 3) Removal Candidates + Iteration Plan

- Load `references/removal-plan.md` for template.
- Identify code that is unused, redundant, or feature-flagged off.
- Distinguish **safe delete now** vs **defer with plan**.

### 4) Security and Reliability Scan

- Load `references/security-checklist.md` for coverage.
- Check for: command injection, path traversal, pickle deserialization, secret leakage, race conditions, distributed race conditions.
- Call out both **exploitability** and **impact**.

### 5) Python-Specific Quality Scan

- Load `references/code-quality-checklist.md` for coverage.
- Check for: missing type hints, exception handling issues, resource management, mutable defaults, import problems.

### 6) ML/Training-Specific Scan

- Load `references/python-ml-checklist.md` for coverage.
- Check for: shape/dtype/device mismatches, gradient issues, memory leaks, distributed training bugs, numerical stability.

### 7) Output Format

```markdown
## Code Review Summary

**Files reviewed**: X files, Y lines changed
**Overall assessment**: [APPROVE / REQUEST_CHANGES / COMMENT]

---

## Findings

### P0 - Critical
(none or list)

### P1 - High
1. **[file:line]** Brief title
   - Description of issue
   - Suggested fix

### P2 - Medium
...

### P3 - Low
...

---

## Removal/Iteration Plan
(if applicable)
```

**Clean review**: If no issues found, state what was checked and any residual risks.

### 8) Next Steps Confirmation

After presenting findings, ask user how to proceed:

1. **Fix all** - Implement all suggested fixes
2. **Fix P0/P1 only** - Address critical and high priority issues
3. **Fix specific items** - User specifies which issues to fix
4. **No changes** - Review complete

**Important**: Do NOT implement any changes until user explicitly confirms.

______________________________________________________________________

## Resources

| File                        | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `solid-checklist.md`        | SOLID smell prompts and refactor heuristics for Python           |
| `security-checklist.md`     | Python security and runtime risk checklist                       |
| `code-quality-checklist.md` | Python-specific error handling, performance, boundary conditions |
| `removal-plan.md`           | Template for deletion candidates and follow-up plan              |
| `python-ml-checklist.md`    | PyTorch/ML-specific issues for distributed training              |
