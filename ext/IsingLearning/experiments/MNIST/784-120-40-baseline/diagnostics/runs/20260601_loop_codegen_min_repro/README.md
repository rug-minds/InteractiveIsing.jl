# NonGenerated Loop Codegen Reproducer

This folder contains a reduced reproducer for the `Processes.loop(::Repeat, ::NonGenerated)` codegen stall seen with the MNIST contrastive worker context.

The script avoids loading MNIST data. It builds the reduced 120-40 field-input graph, creates one synthetic input/target sample, and runs one real repeated loop iteration with `Repeat(2)`.

## Commands

Run from the InteractiveIsing repository root:

```powershell
$script = "ext/IsingLearning/experiments/MNIST/784-120-40-baseline/diagnostics/runs/20260601_loop_codegen_min_repro/loop_codegen_reproducer.jl"
julia --project=ext/IsingLearning $script no-inline-local 2
julia --project=ext/IsingLearning $script typed-ref 2
julia --project=ext/IsingLearning $script bad-local 2
```

## Variants

`bad-local` reproduces the problematic shape:

```julia
nextcontext = @inline Processes._step!(...)
stablecontext = nextcontext
```

On this machine it did not reach the first marker inside `bad_local_loop!` after 90 seconds, while the Julia process was consuming CPU. That indicates the stall happens during specialization/codegen for the method, before function body execution.

`typed-ref` uses the current workaround shape:

```julia
refcontext = Base.RefValue{typeof(initial_context)}(initial_context)
refcontext[] = @inline Processes._step!(..., refcontext[], ...)
```

Observed result: completed one real loop-body `_step!`; stage wall was about 42.9 seconds.

`no-inline-local` keeps the local variable shape but removes call-site `@inline` from `_step!`.

Observed result: completed one real loop-body `_step!`; stage wall was about 5.52 seconds.

## Interpretation

The reproducer points at an interaction between:

- a very large concrete immutable `ProcessContext` return type,
- a loop-carried local context variable,
- call-site `@inline` on `_step!`.

The typed `RefValue` workaround avoids the codegen stall, but it is not a runtime-performance fix for the full learning benchmark.
