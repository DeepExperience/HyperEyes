# Removal & Iteration Plan (Relax Project)

## Priority Levels

- **P0**: Immediate (security risk, blocking other work)
- **P1**: Current sprint
- **P2**: Backlog / next iteration

______________________________________________________________________

## Template

### Safe to Remove Now

| Field | Details |
|-------|---------|
| **Location** | `path/to/file.py:line` |
| **Type** | Unused function / Dead class / Deprecated module / Feature flag |
| **Rationale** | Why remove |
| **Impact** | None / Low — no active consumers |
| **Steps** | 1. Remove code 2. Remove tests 3. Remove config |

### Defer Removal

| Field | Details |
|-------|---------|
| **Location** | `path/to/file.py:line` |
| **Why defer** | Active consumers / needs migration |
| **Preconditions** | What must happen first |
| **Breaking changes** | API/contract changes |

______________________________________________________________________

## Checklist Before Removal

### Code Analysis

- [ ] Searched codebase for all references (`rg`, `grep`)
- [ ] Checked for dynamic usage (`getattr`, string-based references in YAML/JSON)
- [ ] Checked `__init__.py` exports and `__all__`

### Relax-Specific Checks

- [ ] Ray actor registrations
- [ ] Argument parser registrations in `relax/utils/arguments.py`
- [ ] Loss function / reward function registries
- [ ] Shell scripts in `scripts/`
- [ ] Config files in `configs/`
- [ ] Pickle/checkpoint serialization compatibility
