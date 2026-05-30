# Performance And Design Notes

This file documents the design decisions behind the optimized
`mnist_local_manager_grid.jl` experiment path. It is meant as a checklist for
future edits to this experiment family, not as a complete result report.

## Current Baseline

Canonical experiment file:

`mnist_local_manager_grid.jl`

Reference diagnostic folder:

`diagnostics/runs/20260528_135023_bespoke_direct_metropolis_subset`

Measured manager timing after the single-field/process-path optimization:

| path | workers | samples | elapsed | extrapolated 1000-sample epoch |
| --- | ---: | ---: | ---: | ---: |
| old manager path | 32 | 273 | 67.695 s | 247.968 s |
| optimized manager path | 32 | 273 | 2.596 s | 9.509 s |

This is a `26.08x` manager-path speedup on the same diagnostic setup. The
comparison used `r=8`, Metropolis, `50/50` free/nudged sweeps, `3/3` reads,
batch size `32`, and `32` Julia threads.

## Design Goals

- Keep `ProcessManager` as the training orchestration layer.
- Reuse worker graphs, process contexts, buffers, and optimizer state.
- Share static or read-only model data across workers where possible.
- Keep per-sample input writes worker-local.
- Avoid work that scales with all possible matrix indices when only sparse
  nonzeros matter.
- Avoid passing widened process context objects through helper-function stacks
  on the hot path.
- Keep diagnostics small and comparable before launching real experiment grids.

## Graph And Parameter Layout

The source model owns the trainable parameters:

- sparse coupling values through `nonzeros(adj)`;
- base bias values through the base magnetic field;
- optimizer state for sparse couplings and bias values.

Worker models share the adjacency/coupling storage but own local runtime state:

- spin state;
- random number generator;
- combined magnetic field buffer;
- free/nudged best states;
- gradient accumulation buffers.

The important distinction is that training parameters are shared, while
per-sample simulation state is local. A worker must not mutate the source graph
state just to install an image or a target.

## Single-Field Bias Path

The optimized worker path uses one magnetic field per worker graph. That field
stores the effective bias for the current sample:

```julia
combined_bias .= base_bias
combined_bias .+= sample_buffer
```

`base_bias` points at the source model's trainable bias vector. The
`sample_buffer` is worker-local and preallocated. It contains image-induced
input fields and, for nudged phases, target fields.

This replaced the older two-field implementation where the graph carried both a
trainable base field and a mutable sample field. The two-field version was
semantically fine but expensive in the manager path because every field access
and Hamiltonian evaluation had more dynamic structure to traverse.

Important rules:

- Do not allocate a new sample field per sample.
- Do not mutate the source model's base bias when installing validation or
  training samples.
- Do not recover the sample field with `sample_magfield` in new hot code. That
  helper is now legacy-only for old two-field diagnostics.
- If an optimizer updates the source base bias, workers see the updated
  `base_bias` reference on the next sample install.

## Input Handling

Images are encoded as magnetic-field contributions rather than by writing fixed
input spins into graph state. This keeps the active sampling set static and
avoids per-sample graph-state mutation.

For this architecture, field-based input and fixed input spins are equivalent at
the level of the intended clamp semantics. The field path is preferred because
it avoids data movement and lets workers reuse a stable process context.

## Sampling Set And Proposals

The optimized path uses `LocalMNISTFlipProposer` with a static active index
list. Input sites are excluded up front, so proposal generation does not need a
dynamic `ToggledIndexSet` check on every proposal.

Important rules:

- Build the active index list once from the graph/layer layout.
- Sample only hidden/output sites during relaxation.
- Keep image information in the bias field, not in the active index structure.
- Do not rebuild proposer state per sample or per batch.

## Sparse Coupling Updates

The coupling matrix is sparse and should be treated as sparse in all hot
training code.

Gradient accumulation uses an edge layout over existing nonzeros. Adam updates
operate on the coupling `nzval` vector rather than indexing the full sparse
matrix as a dense parameter object.

Important rules:

- Iterate over known edge groups or CSC nonzero ranges.
- Avoid loops over all `(i, j)` pairs.
- Avoid sparse `A[i, j]` lookup inside inner loops when the CSC pointer or
  `nzval` index is already known.
- Keep symmetric edge updates explicit when both directed entries are stored.

## Metropolis Inner Loop

The generic CSC `weighted_neighbors_sum` path avoids a sparse diagonal lookup
inside every proposal. It scans the CSC column and skips the diagonal row while
accumulating neighbor contributions.

`Metropolis.step!` should use temperature already present in the dynamics
context, not recover it indirectly from the graph each step.

Important rules:

- Put scalar dynamics state such as `T` in the process/dynamics context.
- Keep temperature schedule counters as scalar managed state, not `Ref`s, unless
  mutation by shared reference is actually required.
- Avoid abstract graph/Hamiltonian access inside the proposal loop.

## Process Algorithm Shape

Hot process algorithms should own the high-level control flow and destructure
state locally. The full process context should not be passed down through a
stack of helper calls.

Preferred pattern:

```julia
@ProcessAlgorithm function Step!(...)
    @state model
    @state x
    @state base_bias
    @state sample_buffer

    # Extract the concrete pieces needed by local work.
    install_sample_bias!(model, x, base_bias, sample_buffer)
    ...
end
```

Avoid patterns where helper functions receive the whole process context and
then reach into it dynamically. That can cause context widening and prevent
scalar replacement.

Important rules:

- Put buffers needed by a process in `Init`, not in ad hoc closure state.
- Use `@state` for persistent process fields.
- Use explicit `@merge` when state sharing is intentional and the DSL warns
  about overlapping field names.
- Keep helper functions only where they clarify reused behavior. Do not hide
  critical hot-path state access behind unnecessary abstraction.

## Worker And Manager Reuse

The manager should create workers once and reuse them. A minibatch should send
jobs to existing workers, collect worker-local gradients/statistics, and then
flush into the source model once per minibatch.

Important rules:

- Do not rebuild process workers per batch.
- Do not rebuild graphs per sample.
- Keep enough jobs in a benchmark to fill the worker pool.
- Time the measured subset or epoch separately from model construction and
  first-run compilation.
- Always run a warmup before comparing timings.

## Diagnostics

Use small diagnostics before broad learning runs.

Useful diagnostics in the current folder:

- `manager_subset_timing.jl`: manager-level throughput with real minibatches.
- `worker_stripdown_timing.jl`: isolated worker/process timing.
- `metropolis_loop_vs_process_1e5.jl`: lower-level Metropolis/process
  comparison.

Some older diagnostics were written for the two-field path and may call
`sample_magfield` directly. Update them to the combined-field `base_bias` plus
`sample_buffer` path before using them as current performance evidence.

## Common Mistakes To Avoid

- Starting from a checkpoint unless checkpoint initialization is explicitly the
  experiment variable.
- Treating a minibatch timing as an epoch timing.
- Timing setup, compilation, or data loading as the hot epoch path.
- Using raw relaxation step counts without converting to full sweeps.
- Updating a full sparse matrix object with Adam instead of the nonzero vector.
- Installing input by mutating source graph state.
- Reintroducing dynamic input toggling when image input is already in the bias.
- Calling helper functions with the whole process context in hot process code.
- Letting validation/free-phase sampling overwrite trainable source bias state.

## Migration Checklist

When updating another local MNIST experiment file, check the following:

1. Worker graph uses a single combined magnetic field.
2. Worker `Init` contains `base_bias` and a preallocated `sample_buffer`.
3. Sample install writes `combined_bias = base_bias + sample_buffer`.
4. Active proposal indices are static and exclude fixed/input sites.
5. Coupling gradients and optimizer updates operate over sparse nonzero layout.
6. Dynamics temperature is stored in the dynamics/process context.
7. Validation uses worker-local state and does not mutate the source graph.
8. `ProcessManager` workers are reused across minibatches.
9. A warmed manager diagnostic is run before any broad grid.
10. Settings/results record workers, threads, sweeps, reads, batch size, radius,
    optimizer, learning rates, skipped samples, and measured epoch time.
