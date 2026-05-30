# Processes Context Widening Diagnostic

Diagnostic folder:

`ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/diagnostics/runs/20260528_135023_bespoke_direct_metropolis_subset`

## What triggered this

`single_process_inline_vs_process_samples.jl` tried to run the same local MNIST worker routine as:

- a normal `Processes.Process`
- a synchronous `Processes.InlineProcess`
- a direct fused reference loop

The first `InlineProcess` version failed when calling `run(worker; threaded = false)` after constructing the inline worker from the initial context. The failure was a conversion error: the context returned by the first loop run had a wider concrete type than the context stored in the `InlineProcess.context` field.

This matters because `InlineProcess` stores its context type in the struct type parameter:

```julia
InlineProcess{TD, ContextType, Lt, Mode}
```

If the first run adds fields to any subcontext, the returned `ProcessContext` no longer fits in the original `ContextType`.

## Observed field widening

The initial context had these relevant subcontext shapes:

```julia
free_temperature::SubContext{:free_temperature, @NamedTuple{
    step_idx::Base.RefValue{Int64},
    total_steps::Int64,
}}

nudge_temperature::SubContext{:nudge_temperature, @NamedTuple{
    step_idx::Base.RefValue{Int64},
    total_steps::Int64,
}}

FuncWrapper_3::SubContext{:FuncWrapper_3, @NamedTuple{}}

CaptureBestEnergyState!_1::SubContext{:CaptureBestEnergyState!_1, @NamedTuple{}}
```

Before fixing `ext/IsingLearning/src/Tools/LearningProcessTools.jl`, one run returned these widened subcontext shapes:

```julia
free_temperature::SubContext{:free_temperature, @NamedTuple{
    step_idx::Base.RefValue{Int64},
    total_steps::Int64,
    current_T::Float32,
}}

nudge_temperature::SubContext{:nudge_temperature, @NamedTuple{
    step_idx::Base.RefValue{Int64},
    total_steps::Int64,
    current_T::Float32,
}}

FuncWrapper_3::SubContext{:FuncWrapper_3, @NamedTuple{
    stats::@NamedTuple{
        loss::Float32,
        correct::Bool,
        skipped::Bool,
    },
}}

CaptureBestEnergyState!_1::SubContext{:CaptureBestEnergyState!_1, @NamedTuple{
    energy::Float32,
}}
```

Those added fields are loop outputs from:

- `GeometricTemperatureSchedule`: returns `(; current_T)`
- `ReverseAnnealTemperatureSchedule`: returns `(; current_T)`
- `CaptureBestEnergyState!`: returns `(; energy)`
- wrapped `finish_contrastive_sample!`: routed as `stats`

## Local fixes applied

The temperature and capture widenings were local IsingLearning issues, not
package-level `Processes` issues:

- `GeometricTemperatureSchedule` now initializes `current_T` as managed state.
- `ReverseAnnealTemperatureSchedule` now initializes `current_T` as managed state.
- `CaptureBestEnergyState!` no longer returns `energy`; the MNIST routines use it
  only for its side effect of updating `best_energy`/`best_state`.
- `PowerLawTemperatureSchedule` in `ext/IsingLearning/src/Dynamics.jl` received
  the same initialized `current_T` treatment.

A one-step local MNIST worker probe after the fix showed stable keys for the
local IsingLearning algorithms:

```text
before
:FuncWrapper_1 => Symbol[]
:FuncWrapper_2 => Symbol[]
:FuncWrapper_3 => Symbol[]
:FuncWrapper_4 => Symbol[]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]

after
:FuncWrapper_1 => Symbol[]
:FuncWrapper_2 => Symbol[]
:FuncWrapper_3 => [:stats]
:FuncWrapper_4 => Symbol[]
:free_temperature => [:step_idx, :total_steps, :current_T]
:CaptureBestEnergyState!_1 => Symbol[]
:nudge_temperature => [:step_idx, :total_steps, :current_T]
```

## FuncWrapper removal in the local MNIST worker

The local MNIST worker was then changed to avoid plain function calls in the DSL
for the sample-bias and finish/update steps. It now uses explicit process
algorithms:

- `InstallLocalSampleBias!`
- `InstallLocalNudgedSampleBias!`
- `FinishLocalContrastiveSample!`

`FinishLocalContrastiveSample!` initializes `loss`, `correct`, and `skipped` in
its process subcontext, then updates those fields on every run. A follow-up
one-step probe showed no `FuncWrapper_*` subcontexts and no context widening:

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

## Error shape

The error was a `MethodError: Cannot convert ProcessContext{... widened ...} to ProcessContext{... initial ...}` when `InlineProcess` tried to store the returned persistent context back into its typed `context` field.

The first workaround in `single_process_inline_vs_process_samples.jl` was to construct a normal `Process`, run it once, take the stable post-run context, and then construct the `InlineProcess` from that warmed context:

```julia
warm_process = Processes.Process(...)
run(warm_process)
wait(warm_process)
stable_context = Processes.context(warm_process)

inline_worker = Processes.InlineProcess(
    algorithm;
    context = stable_context,
    repeats = 1,
    threaded = false,
)
```

That allows the inline path to run because its stored `ContextType` already includes the widened fields.

## Why this is likely a Processes package issue

The process routine is type-unstable across the first run because outputs are added to subcontexts only after execution. Normal `Process` can tolerate this because it can replace its runtime context with a new concrete context. `InlineProcess` cannot tolerate it because its context field type is fixed.

Possible package-level fixes to evaluate:

- Initialize output fields in subcontexts during `init`, so first-run and steady-state context types match.
- Make `InlineProcess` reject algorithms whose first run widens context types, with a clearer error.
- Add a supported `warm = true` construction path that computes the stable context type before creating the final `InlineProcess`.
- Ensure `FuncWrapper`/scheduled algorithm outputs have declared output slots in the initial context when the routine is resolved.

## Related timing note

After constructing `InlineProcess` from the warmed context, the 2-sample smoke run printed:

```text
path,nsamples,total_seconds,seconds_per_sample,samples_per_second,steps_per_second
direct_fused,2,0.38055,0.190275,5.255546,2.72605153e6
normal_process,2,5.646506,2.823253,0.354201,183724.246
inline_process,2,2.974451,1.487226,0.672393,348770.22
```

This suggests `InlineProcess` reduces overhead substantially versus normal `Process`, but the composed worker routine is still far slower than the fused direct loop.
