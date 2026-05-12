# Processes Suggested Improvements

Notes from using `deps/Processes` in the current InteractiveIsing codebase, especially for Langevin dynamics and the IsingLearning XOR/MNIST training path.

## What Works Well

- The core idea is useful: a `Process` combines an algorithm, a context, and a lifecycle.
- `@Routine` and `@CompositeAlgorithm` make multi-phase workflows concise.
- The package is already good for interactive simulation, where algorithms run repeatedly and expose state.
- Capturers are useful for equilibrium-propagation-style code because plus/minus phase states can be kept without hand-written plumbing.
- Context inspection is valuable; being able to look at `worker.context.dynamics.state`, captures, buffers, and managed values helped debug real bugs.

## Suggested Improvements

### 1. Context Collision Handling

The repeated `GeneralState` overlap warnings are useful, but easy to ignore.

Suggested change:

- Make collisions opt-in when possible.
- Provide a small helper/macro for intentional overrides.
- Include the full context path in the warning/error.

Example problem:

- `:equilibrium_state`, `:x`, and `:y` are merged from multiple routines.
- In training code this is usually intentional, but the warning does not distinguish intentional sharing from accidental shadowing.

### 2. Lifecycle Documentation

The most important docs should explain exact guarantees for:

- `Processes.reset!`
- `run`
- `wait`
- `close`
- `copyprocess`
- context reuse after `close`

The main practical question is: after each lifecycle call, which state is guaranteed reset, preserved, or invalid?

This matters for worker reuse in minibatch training.

### 3. Execution Trace / Dataflow View

A debugging helper that prints the resolved routine tree would help a lot.

Useful output:

- subroutine order
- aliases used by each subroutine
- state keys read/written
- captured outputs
- algorithm instances and context paths

This is not about unclear aliasing semantics in the package itself; it is about making intended sharing/ownership visible from user code.

### 4. Dynamic Context Versus Cached Init Data

The `BlockLangevin` issue came from caching `active_spins` at init while `apply_input` later changed the active index set.

Suggested improvement:

- Add a convention or debug check for init-time cached values derived from mutable context state.
- Document when an algorithm should recompute from context inside `step!` instead of caching in `init`.

This is especially relevant for `ToggledIndexSet`, clamping, and learning workflows.

### 5. Deterministic Debug Mode

Training/debugging would benefit from a simple deterministic execution mode:

- single worker
- fixed RNG propagation
- no async scheduling
- strict lifecycle checks
- optional validation that inputs/targets remain clamped

The bespoke XOR loop was useful because it made these checks explicit.

### 6. Better Process Error Context

When a process fails, include:

- process id/name
- current routine/subroutine path
- context path
- last algorithm return value if available
- relevant managed state

This would make failures in nested `CompositeAlgorithm`s much faster to localize.

### 7. Context Schema / Required Inputs

It would help if algorithms could expose a small schema:

- required context keys
- optional context keys
- managed keys
- expected element types/shapes where known

This could be used for preflight checks before running training loops.

### 8. Examples Focused on Non-Interactive Use

Most benefits are obvious in interactive simulation. More examples should show:

- reusable worker process loops
- deterministic validation workers
- batched training
- capturer use in two-phase algorithms
- safe process copying

## Current Assessment

Processes is useful and worth continuing. It speeds up development for composable interactive workflows. For learning code, it still needs better lifecycle docs, tracing, and strict debug tools so failures are easier to reason about.

The main design issue is not that sharing/aliasing is undefined. The main issue is that user code needs better tools to inspect and validate that the sharing it intends is what the resolved process is actually doing.
