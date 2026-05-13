# XOR Latency Characterization

This folder measures where the first-epoch latency comes from for two small
XOR learning paths:

- `profile_manager_timeavg.jl`: persistent `ProcessManager` path used by the
  time-averaged XOR experiment.
- `profile_simple_langevin.jl`: direct/manual worker path used by the older
  simple Langevin XOR experiment.

The measurements were run from a fresh Julia process, not DaemonMode, because
the goal was to separate cold compilation from actual dynamics runtime.

## Commands

Manager path:

```bash
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/latency_characterization/profile_manager_timeavg.jl
```

Direct/manual path:

```bash
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/latency_characterization/profile_simple_langevin.jl
```

Both scripts accept these environment overrides:

- `ISING_LATENCY_WORKERS`
- `ISING_LATENCY_MINIT`
- `ISING_LATENCY_EVAL_REPEATS`
- `ISING_LATENCY_FREE`
- `ISING_LATENCY_NUDGED`
- `ISING_LATENCY_TEMP`
- `ISING_LATENCY_STEPSIZE`
- `ISING_LATENCY_BETA`

The full characterization below used the script defaults:

- workers: `8`
- repeated initial states per XOR case: `8`
- validation repeats: `16`
- free relaxation: `600`
- nudged relaxation: `600`
- temperature: `0.005`
- Langevin stepsize: `0.4`
- clamping beta: `2.0`

## Output Files

Full manager run:

- `runs/manager_timeavg_latency_20260513_013204/README.md`
- `runs/manager_timeavg_latency_20260513_013204/manager_timeavg_latency.csv`

Full direct/manual run:

- `runs/simple_langevin_latency_20260513_014357/README.md`
- `runs/simple_langevin_latency_20260513_014357/simple_langevin_latency.csv`

Earlier smoke runs with very small relaxation settings are also kept under
`runs/`; those were used only to validate that the profiling scripts ran.

## Main Result

The first-epoch delay is mostly compilation and specialization, not the
`ProcessManager` itself.

The two largest cold phases are:

| path | phase | wall seconds | compile time |
|---|---:|---:|---:|
| manager | `manager init_timeavg_trainer` | 119.19 | 118.93 |
| manager | `manager train1: ProcessManager.run!` | 182.57 | 228.41 |
| direct/manual | `simple init_simple_trainer` | 107.74 | 107.52 |
| direct/manual | `simple train1: worker loop` | 256.18 | 322.24 |

`compile_time` can exceed wall time in these threaded/process phases; treat it
as evidence that the phase triggered heavy compilation, not as a literal
fraction of elapsed time.

## Where Time Goes

### Manager Path

Full run:

| phase | wall seconds | compile time | interpretation |
|---|---:|---:|---|
| include experiment file | 9.28 | 1.50 | package/script loading, not the main problem |
| initialize trainer/managers/workers | 119.19 | 118.93 | cold specialization of composite/process/context stack |
| validation `ProcessManager.run!` | 40.58 | 40.69 | first execution of validation process path compiles |
| training `ProcessManager.run!` | 182.57 | 228.41 | first execution of full free/plus/minus training process path compiles |
| optimizer update | 0.30 | 0.30 | negligible compared with process compilation |
| sync/reinit workers | 0.12 | 0.12 | not the bottleneck |

The manager itself is not the dominant overhead. The costly call is the first
actual execution of the process graph inside `ProcessManager.run!`.

### Direct/Manual Path

Full run:

| phase | wall seconds | compile time | interpretation |
|---|---:|---:|---|
| include experiment file | 9.14 | 1.41 | package/script loading, not the main problem |
| graph construction | 1.67 | 1.66 | small |
| initialize trainer/workers | 107.74 | 107.52 | cold specialization of process/context stack |
| validation | 9.32 | 9.31 | first validation path compiles |
| training worker loop | 256.18 | 322.24 | first execution of free/plus/minus workers compiles and runs dynamics |
| optimizer/broadcast | 0.26 | 0.26 | negligible |

The direct/manual path is not faster on the first training call. It actually
spent more wall time in the first training worker loop than the manager path
for this setup.

## Important Observation

Creating a fresh trainer later in the same Julia process still recompiles a lot:

| path | phase | wall seconds | compile time |
|---|---:|---:|---:|
| manager | `manager warm init_timeavg_trainer` | 115.67 | 115.37 |
| manager | `manager train101: ProcessManager.run!` | 178.02 | 231.56 |
| direct/manual | `simple warm init_simple_trainer` | 105.57 | 105.30 |
| direct/manual | `simple train101: worker loop` | 257.95 | 312.81 |

So the right optimization is not just "keep Julia alive"; it is "do not rebuild
fresh composite/process/trainer structures while experimenting." Keep one
trainer/manager alive and mutate its runtime parameters or graph buffers.

## Interpretation

The first epoch feels stuck because the first training trajectory exercises a
large, highly typed stack:

- composite algorithm resolution
- context construction and merging
- identifiable algorithm wrappers
- Langevin process execution
- capture algorithms
- clamping terms
- gradient-buffer accumulation
- worker finalization/closing

Julia is specializing all of that for the concrete graph, context, algorithm,
and buffer types. That is useful after compilation, but painful when scripts
construct new concrete process objects repeatedly.

The `ProcessManager` does not appear to be the core latency problem. In this
measurement, the manager path had the same qualitative compilation bottleneck
as the manual path and a lower wall time for the first full training run.

## Practical Next Steps

1. Keep a persistent trainer and persistent managers while tuning.
   Do not recreate workers/composites every experiment unless the topology
   actually changes.

2. Put mutable experimental parameters in a config object and update them
   between runs.
   This matches the pattern now started in the experiments.

3. Add a real warm-run benchmark that calls multiple epochs on the same
   persistent manager after the first epoch. The current direct/manual worker
   helper closes workers after use, so it is not suitable for a same-worker
   warm-epoch benchmark.

4. If cold start must improve, add explicit precompile workloads for the exact
   trainer/process construction and first `run!` path. Generic package
   precompilation alone will not cover these experiment-specific concrete types.

5. For interactive work, DaemonMode or a long-lived REPL helps package load time,
   but it does not solve recompilation caused by rebuilding fresh process graphs.

