# SOLID & Code Smell Prompts (Relax Project)

## SOLID Quick Reference

| Principle | Key Question | Red Flag |
|-----------|-------------|----------|
| **SRP** | "What is the single reason this module would change?" | File mixes unrelated concerns (e.g., data loading + loss computation + logging) |
| **OCP** | "Can I add a new variant without touching existing code?" | Growing `if/elif/else` chains for new reward types, backends, etc. |
| **LSP** | "Can I substitute any subclass without the caller knowing?" | `NotImplementedError` in overridden methods; `isinstance` checks for subclass type |
| **ISP** | "Do all implementers use all methods?" | ABC with many abstract methods, most left as stubs by implementers |
| **DIP** | "Can I swap the implementation without changing business logic?" | High-level logic directly instantiating concrete I/O / storage types |

______________________________________________________________________

## Common Code Smells

| Smell | Signs |
|-------|-------|
| **Long function** | Function >30 lines, multiple nesting levels |
| **Feature envy** | Method uses more data from another class than its own |
| **Data clumps** | Same group of parameters passed together repeatedly |
| **Primitive obsession** | Using dicts instead of dataclasses for structured data |
| **Shotgun surgery** | One change requires edits across many files |
| **Dead code** | Unreachable or never-called code |
| **Magic numbers** | Hardcoded values without named constants |

______________________________________________________________________

## Refactor Heuristics

1. Split by responsibility, not by size
2. Introduce abstraction only when needed (Rule of Three)
3. Keep refactors incremental — isolate behavior before moving
4. Preserve behavior first — add tests before restructuring
5. Prefer composition over inheritance (AGENTS.md: hierarchy ≤ 2)
6. Use dataclasses for data containers — avoid dicts when structure is known
7. Prefer protocols over ABCs — structural subtyping is more Pythonic
