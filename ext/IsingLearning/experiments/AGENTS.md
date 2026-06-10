# Experiment Agent Instructions

These instructions apply to all experiment code under `ext/IsingLearning/experiments`.

- `ProcessAlgorithm`s must be small atomic operations. They should set a field, copy a state, install one sample, accumulate one result, or perform another single-purpose operation.
- Do not put phase orchestration, learning loops, nested simulation flows, or multi-step training logic inside one `ProcessAlgorithm`.
- Compose work with `StatefulAlgorithms.@Routine` and `StatefulAlgorithms.@CompositeAlgorithm`. Nesting and sequencing belongs at the Routine/Composite level, and composed routines are the intended optimized package path.
- If a phase needs repeated temperature scheduling plus dynamics, follow the existing manager pattern: build a reusable `phase_step = ..._phase_step_algorithm(...)` Routine, then call `@repeat steps phase_step()`.
- Do not use inline `@repeat steps begin ... end` blocks for manager training phases when the block contains multiple process calls. Give the block a named Routine instead so routing, state ownership, and composition stay explicit.
- Keep worker-local mutable buffers in Routine state. Share only static model storage such as adjacency/parameter arrays deliberately and document that sharing at construction.
- Before adding a new manager experiment, check an existing working file in the same experiment family and mirror its ProcessAlgorithm/Routine/Composite structure.
