# Experiment File Migration Plan

This note is for updating experiment files after the local MNIST worker
diagnostics. `ProcessPerformanceConsiderations.md` is the package-facing note
for `StatefulAlgorithms` performance and algorithm authoring. This file is the practical
experiment migration checklist.

## Goal

Bring other experiment scripts in line with the optimized local MNIST worker:

- no first-run context widening,
- no hot direct function calls inside the `@Routine` DSL when a stable
  `@ProcessAlgorithm` is better,
- side-effect-only algorithms return `nothing`,
- any algorithm-owned outputs are initialized with `@managed`,
- timing-sensitive worker paths avoid passing full context below `step!`.

## Already Updated

Canonical single-hidden local worker:

`ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl`

Changes already applied:

- `install_sample_bias!(...)` DSL call replaced by `InstallLocalSampleBias!`.
- `install_nudged_sample_bias!(...)` DSL call replaced by
  `InstallLocalNudgedSampleBias!`.
- `finish_contrastive_sample!(...)` plus `update_worker_stats!(...)` DSL calls
  replaced by `FinishLocalContrastiveSample!`.
- `FinishLocalContrastiveSample!` initializes `loss`, `correct`, and `skipped`
  with `@managed`.
- Context-shape probe showed the same subcontext keys before and after one run.

Shared IsingLearning tools:

`ext/IsingLearning/src/Tools/LearningProcessTools.jl`

- `GeometricTemperatureSchedule` initializes `current_T`.
- `ReverseAnnealTemperatureSchedule` initializes `current_T`.
- `CaptureBestEnergyState!` returns `nothing`.

`ext/IsingLearning/src/Dynamics.jl`

- `PowerLawTemperatureSchedule` initializes `current_T`.

## Files To Update Next

Canonical two-hidden local MNIST manager:

`ext/IsingLearning/experiments/MNIST/two-hidden-local-28x28-to-14x14-readout/mnist_local_paper_manager_grid.jl`

Observed patterns:

```julia
install_sample_bias!(mnist_model, x)
install_nudged_sample_bias!(mnist_model, x, y)
stats = finish_contrastive_sample!(gradient, mnist_model, x, y, free_state, nudged_state)
update_worker_stats!(nsamples, ncorrect, nskipped, total_loss, stats)
```

Migration:

- Add `InstallLocalSampleBias!` equivalent for this file's model type.
- Add `InstallLocalNudgedSampleBias!` equivalent.
- Add `FinishLocalContrastiveSample!` equivalent that does finish + stats
  counter update in one process algorithm.
- Replace the four direct DSL calls in `free_phase_algorithm`,
  `nudged_phase_algorithm`, and `local_worker_algorithm`.
- Probe before/after subcontext keys and confirm no `FuncWrapper_*` widening.

Canonical inlaid-input MNIST trainer:

`ext/IsingLearning/experiments/MNIST/inlaid-55x55-pixel-readout/mnist_inlaid_input_training.jl`

Observed pattern:

```julia
stats = finish_contrastive_sample!(gradient, inlaid_model, x, y, free_state, nudged_state)
update_worker_stats!(nsamples, ncorrect, nskipped, total_loss, stats)
```

Migration:

- Add an inlaid-specific finish/update process algorithm with initialized
  `loss`, `correct`, and `skipped` fields.
- Replace the direct finish/update DSL calls.
- Check whether any direct bias or target install calls are also routed through
  the DSL in this file; if yes, wrap them in side-effect-only process algorithms.
- Probe before/after subcontext keys.

Historical run snapshots:

Examples:

`ext/IsingLearning/experiments/MNIST/two-hidden-local-28x28-to-14x14-readout/experiments/current/*/mnist_local_paper_manager_grid.jl`

Do not update historical run snapshots unless deliberately re-running from that
exact snapshot. Prefer updating the canonical architecture script, then launching
new dated runs from it.

Backup/reference files:

`mnist_local_manager_grid_OLD.jl`

Do not migrate backup files. Keep them as historical comparison sources.

## Recommended ProcessAlgorithm Pattern

For side-effect-only steps:

```julia
# Install free-phase sample fields in a stable process subcontext.
StatefulAlgorithms.@ProcessAlgorithm function InstallSampleBias!(
    model,
    x,
)
    install_sample_bias!(model, x)
    return nothing
end
```

For finish/update steps with owned outputs:

```julia
# Finish a sample and update counters without routed FuncWrapper outputs.
StatefulAlgorithms.@ProcessAlgorithm function FinishContrastiveSample!(
    gradient,
    model,
    x,
    y,
    free_state,
    nudged_state,
    nsamples,
    ncorrect,
    nskipped,
    total_loss,
    @managed(loss = zero(Float32)),
    @managed(correct = false),
    @managed(skipped = false),
)
    stats = finish_contrastive_sample!(gradient, model, x, y, free_state, nudged_state)
    loss = stats.loss
    correct = stats.correct
    skipped = stats.skipped

    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    nskipped[] += skipped ? 1 : 0
    total_loss[] += loss
    return (; loss, correct, skipped)
end
```

For very hot paths, consider inlining the body of `finish_contrastive_sample!`
inside the process algorithm, as done in the single-hidden worker, so the process
algorithm owns both the stats computation and counter update.

## Macro Documentation Note

When using `StatefulAlgorithms.@ProcessAlgorithm` qualified, docstrings attached directly
to the macro call can fail because Julia tries to document the expanded block.
Use a plain comment above the macro call, or import the macro unqualified in a
module where docstrings are known to work.

Example:

```julia
# Side-effect-only graph reset for process routines.
StatefulAlgorithms.@ProcessAlgorithm function ResetGraphState!(graph)
    resetstate!(graph)
    return nothing
end
```

## Verification Checklist

For each migrated experiment file:

1. Run a one-sample context-shape probe.
2. Print relevant subcontext keys before and after one process run.
3. Confirm no `FuncWrapper_*` subcontext appears in the hot worker routine.
4. Confirm every stateful process algorithm has the same keys before and after.
5. Run a 1-2 sample timing diagnostic with warmup.
6. Compare against the current single-hidden reference timing:

```text
normal Process no-widening worker: about 0.658 s/sample
direct fused reference: about 0.190 s/sample
```

7. Only after the timing and context shape look sane, start training diagnostics.

## Search Commands

Use these to find likely migration targets:

```powershell
rg -n "stats = .*\\(|finish_contrastive_sample!|update_worker_stats!" ext/IsingLearning/experiments/MNIST -g "*.jl"
rg -n "install_sample_bias!|install_nudged_sample_bias!" ext/IsingLearning/experiments/MNIST -g "*.jl"
rg -n "return \\(; .*\\)" ext/IsingLearning/experiments/MNIST ext/IsingLearning/src -g "*.jl"
```

The last command is broad; inspect results manually and only change process
algorithms where the returned field is either uninitialized or unused.
