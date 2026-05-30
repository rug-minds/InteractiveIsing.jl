# Process Performance Considerations

This note records performance patterns observed while diagnosing the local MNIST
worker in:

`ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/diagnostics/runs/20260528_135023_bespoke_direct_metropolis_subset`

The goal is to keep normal `Process` execution close to the cost of calling the
same sampler logic directly.

## Baseline Observations

For the MNIST single-worker sample test with 2 samples:

| Path | Total | Seconds/sample | Steps/second |
| --- | ---: | ---: | ---: |
| Direct fused loop | 0.381 s | 0.190 s | 2.73e6 |
| Normal Process before local inlining | 5.647 s | 2.823 s | 1.84e5 |
| Normal Process after local `@inline`/stripdown | 2.875 s | 1.438 s | 3.61e5 |
| Destructured normal Process | 2.254 s | 1.127 s | 4.60e5 |

The pure Metropolis process test did not show this problem:

| Path | Steps | Seconds | Steps/second |
| --- | ---: | ---: | ---: |
| Custom loop | 100000 | 0.301 s | 3.33e5 |
| Normal Process | 100000 | 0.283 s | 3.53e5 |

So the overhead is not inherent to a simple `Process` wrapping one Metropolis
step. It comes from the composed MNIST worker path.

## Slow Pattern: Context Widening On First Run

Problem shape:

```julia
@ProcessAlgorithm function StatefulStep(
    model,
    @managed(step_idx = Ref(0)),
)
    value = compute_value(model, step_idx[])
    step_idx[] += 1
    return (; value)
end
```

This initializes a context with only `step_idx`, then widens the subcontext on
the first run by adding `value`. Normal `Process` can survive this by replacing
its context. `InlineProcess` stores a concrete context type, so the first run can
fail or force warmup workarounds.

Better pattern:

```julia
@ProcessAlgorithm function StatefulStep(
    model,
    @managed(step_idx = Ref(0)),
    @managed(value = initial_value(model)),
)
    value = compute_value(model, step_idx[])
    step_idx[] += 1
    return (; value)
end
```

What we changed:

- `GeometricTemperatureSchedule` now initializes `current_T`.
- `ReverseAnnealTemperatureSchedule` now initializes `current_T`.
- `PowerLawTemperatureSchedule` now initializes `current_T`.

## Slow Pattern: Returning Unused Side-Effect Outputs

Problem shape:

```julia
@ProcessAlgorithm function CaptureThing!(model, best, buffer)
    score = compute_score(model)
    if score < best[]
        best[] = score
        buffer .= state(model)
    end
    return (; score)
end
```

If `score` is not consumed downstream, returning it only adds context state and
can widen the context. It also communicates that downstream code may rely on the
value.

Better pattern:

```julia
@ProcessAlgorithm function CaptureThing!(model, best, buffer)
    score = compute_score(model)
    if score < best[]
        best[] = score
        buffer .= state(model)
    end
    return nothing
end
```

What we changed:

- `CaptureBestEnergyState!` no longer returns `energy`; it updates
  `best_energy` and `best_state` only.

## Slow Pattern: Passing Full Context Through Helper Functions

Problem shape:

```julia
function helper!(context)
    context.a[] += context.b
    work!(context.model, context.buffer)
end

function Processes.step!(algo::MyAlgorithm, context)
    helper!(context)
    return nothing
end
```

This moves the full context below the top-level process boundary. In this
diagnostic, that correlated with worse performance and made it harder for the
compiler to keep the context scalarized.

Better pattern:

```julia
function helper!(state::S, work::W) where {S<:NamedTuple,W<:NamedTuple}
    state.a[] += state.b
    work!(work.model, work.buffer)
end

function Processes.step!(algo::MyAlgorithm, context)
    (; a, b, model, buffer) = context
    state = (; a, b)
    work = (; model, buffer)
    helper!(state, work)
    return nothing
end
```

What we tested:

- `destructured_process_timing.jl` keeps raw `context` at the `step!` boundary.
- Below `step!`, calls receive small named tuples like `sample`, `buffers`,
  `phase`, and `counters`.

This improved the normal Process path versus the original full worker path, but
did not close the full gap to the direct fused loop.

## Slow Pattern: Non-Inlined Tiny Wrappers Around Process Calls

Problem shape:

```julia
function run_once!(process)
    reset!(process)
    run(process)
    wait(process)
end
```

For hot timing loops, wrappers that take process/context objects can become a
surprising source of overhead if they are not inlined.

Better pattern:

```julia
@inline function run_once!(process::P) where {P<:Processes.Process}
    Processes.reset!(process)
    @inline run(process)
    @inline wait(process)
    return process
end
```

What we changed in diagnostics:

- Added `@inline` to helper definitions that take process/context objects.
- Added `@inline` at call sites in the hot timing loops.

This made the normal Process stripdown several times faster than the original
single-worker measurement, but still slower than the direct fused loop.

## FuncWrapper Output Slots

After fixing local IsingLearning widening, the worker context probe initially
showed:

```text
before
:FuncWrapper_3 => Symbol[]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]

after
:FuncWrapper_3 => [:stats]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]
```

That widening came from the wrapped `finish_contrastive_sample!` output being
routed as `stats`.

For the local MNIST worker, we removed the direct function calls from the DSL and
replaced them with explicit process algorithms:

```julia
InstallLocalSampleBias!(mnist_model, x)
InstallLocalNudgedSampleBias!(mnist_model, x, y)
FinishLocalContrastiveSample!(
    gradient,
    mnist_model,
    x,
    y,
    free_state,
    nudged_state,
    nsamples,
    ncorrect,
    nskipped,
    total_loss,
)
```

`FinishLocalContrastiveSample!` owns initialized `loss`, `correct`, and
`skipped` fields, so its subcontext has the same shape before and after a run.

Follow-up probe:

```text
before
:InstallLocalSampleBias!_1 => Symbol[]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:InstallLocalNudgedSampleBias!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]
:FinishLocalContrastiveSample!_1 => [:loss, :correct, :skipped]

after
:InstallLocalSampleBias!_1 => Symbol[]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:InstallLocalNudgedSampleBias!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]
:FinishLocalContrastiveSample!_1 => [:loss, :correct, :skipped]
```

This local workaround removes the remaining observed widening from the MNIST
worker. The package-level `FuncWrapper` question is still relevant for other DSL
function calls that return routed outputs.

## Practical Checklist

- Stateful process algorithms should initialize every field they return.
- Prefer scalar managed state like `@managed(step_idx::Int = 0)` over
  `@managed(step_idx = Ref(0))` unless mutation through shared identity is
  explicitly needed.
- Side-effect-only process algorithms should return `nothing`.
- Prefer explicit `@ProcessAlgorithm`s over direct DSL function calls for hot
  routed outputs, especially when output slots need stable initial context shape.
- Keep raw `context` at the top-level `Processes.step!` boundary when possible.
- Destructure context once, then pass small typed named tuples or explicit
  values to helper functions.
- Mark hot helpers and their call sites with `@inline` when they take process,
  context, or context-derived tuples.
- Avoid relying on abstract fields stored inside `IsingGraph` in hot capture
  paths. Prefer passing the initialized dynamics-owned `hamiltonian` from the
  sampler context to energy helpers.
- Benchmark against a direct loop that does the same physical work before
  optimizing the manager or worker scheduling layer.
- Check context shape before and after one run when `InlineProcess` needs a warm
  context or when first-run timing differs heavily from steady-state timing.

## Notes From Follow-Up Optimizations

Two cleanup changes were tested after the first stripdown:

- `GeometricTemperatureSchedule` and `ReverseAnnealTemperatureSchedule` were
  changed from `Ref`-based `step_idx` state to scalar managed `Int` state.
- The single-hidden worker was switched to `GeometricDynamicsTemperatureSchedule`
  and `ReverseAnnealDynamicsTemperatureSchedule`, which update the dynamics
  context `T` instead of mutating `isinggraph.temp`. `Metropolis.step!` now uses
  `context.T` directly.
- `CaptureBestEnergyStateFromHamiltonian!` was added so capture can evaluate
  energy with the initialized dynamics-owned Hamiltonian instead of reading the
  abstract `isinggraph.hamiltonian` field.

The timing effect was not the main missing factor. On the same 2-sample worker
stripdown, the full-worker normal process was still about `0.63 s/sample`.
The single-field diagnostic improved from `0.623 s/sample` to `0.542 s/sample`
for the full worker, so double `MagField` evaluation remains a real but partial
cost.

One package-level issue remains: `@ProcessAlgorithm` currently does not handle
some `where` type selectors on generated signatures correctly. A signature like
`f(x::X, y::Y) where {X,Y}` can expand to code where a type variable is not in
scope. Until that is fixed in Processes, hot process algorithms may need either
abstract annotations on the DSL-facing wrapper or manual `ProcessAlgorithm`
definitions that forward to typed helper functions.
