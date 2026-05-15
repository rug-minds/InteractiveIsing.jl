# Processes Usage Notes

Short notes for using `deps/Processes` correctly in the IsingLearning experiments.

## Algorithm Identity

- Do not manually assign ids to algorithms in experiment code.
- Do not wrap samplers in `Unique(...)` just to make branch names differ.
- If the same sampler type appears in free, plus, and minus phases, call it the
  same thing in each routine:

```julia
@alias dynamics = dynamics_algorithm
@repeat nsteps dynamics()
```

Then provide one matching input:

```julia
Input(:dynamics, model = graph, rng = rng)
```

This lets the registry merge the sampler entry. That is fine when the phases are
sequential and each phase resets the graph state before running.

Use `Unique(...)` only when there are genuinely multiple logical instances of the
same algorithm in the same composite that must both exist at the same time and be
targeted separately. If you use it, create it once and reuse/copy the composite;
do not generate fresh unique algorithm values per worker.

## Branch Behavior

Put phase-specific behavior inside the routine/composite, not in the sampler id.
For example:

```julia
plus = @Routine begin
    @state equilibrium_state
    @alias dynamics = dynamics_algorithm
    setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
    set_clamping_beta!(dynamics.model, beta)
    @repeat nudged_steps dynamics()
end
```

If extra behavior is needed around a sampler, wrap that behavior in a
`@Routine` or `@CompositeAlgorithm`.

## Contexts

- Do not write `@context name = algo()` unless the returned context is needed
  later.
- If you need a returned context, name it explicitly:

```julia
@context free = forward()
contrastive_gradient(free.dynamics.model, ...)
```

- Warnings about overlapping `GeneralState` fields are expected in these
  composites when branches share fields like `x`, `y`, or `equilibrium_state`.
- A `ProcessContext` stores subcontexts in `context.subcontexts` and resolves
  user access through overloaded `getproperty`. Use the same direct access
  pattern as the working code:

```julia
worker.context._state.buffers
worker.context.dynamics.model
worker.context.plus_capture.captured
```

- Do not use `hasproperty(worker.context, :dynamics)` as the test for whether a
  subcontext is reachable. `ProcessContext` access is registry/subcontext based,
  and `propertynames`/`hasproperty` can describe concrete struct fields rather
  than the user-facing subcontext lookup. If a required subcontext should exist,
  access it directly and let an error expose a malformed process.

## ProcessManager

- Let `ProcessManager` own persistent workers. Do not recreate the manager every
  epoch.
- If `workers` is omitted, the manager calls recipe `makeworker` once to build a
  template worker. It then uses the template task data for the remaining slots.
  Normal recipes should define `makeworker`; do not manually implement template
  copying inside `makeworker`.
- For `Process` workers, prefer recipe `makecontext` when each worker slot needs
  a custom context but the algorithm/task data should be reused. This keeps the
  expensive resolved composite from being rebuilt per worker.
- Define `copyworker` only when the default manager copy is semantically wrong
  for the worker type.
- `prepare!` should mutate the slot worker for the current job and then reset it:

```julia
prepare! = (slot, job, manager) -> begin
    ctx = slot.worker.context
    ctx._state.x .= job.x
    ctx._state.y .= job.y
    Processes.resetworker!(slot)
end
```

- For existing `Process` workers, default `start!`, `isdone`, and `finalize!`
  already call `run(worker)`, `isdone(worker)`, and `wait(worker); close(worker)`.
  Only override them when the worker is not a normal `Process` or when that
  lifecycle is intentionally different.
- Accumulate gradients in worker-local buffers during a batch. Use
  `FlushAtEnd()` for one merge after all dispatched jobs finish. This is the
  correct manager analogue of the working `_collect_batch_gradient!` path.
- Do not synchronize worker graphs after every single example unless the
  algorithm specifically requires online updates.
- After an optimiser update, synchronize graph parameters by mutating the graph
  objects already held by workers:

```julia
for worker in Processes.workers(manager)
    Processes.isdone(worker) && close(worker)
    IsingLearning.sync_graph_params!(worker.context.dynamics.model, params)
end
```

- Do not call `initcontext(worker.context, :dynamics)` after this kind of
  parameter sync. Reinitializing a subcontext rebuilds it from the process init
  path and can discard or overwrite in-place graph mutations. Use
  `reinitworker!`/partial reinit only when you intentionally want to rebuild
  context from new `Input`/`Override` values.
- If using manager state for mutable training buffers, keep the buffers in
  `manager.state` and let `flush!` write into those buffers. Do not store
  reusable worker subcontexts as results beyond `consume!`; copy scalar/vector
  values out instead.

## Shared Parameter Storage

Use this pattern when worker graphs should have local spin state and local
buffers, but trainable Hamiltonian parameters should come from one shared memory
owner. This is useful for experiments that want to avoid explicit parameter
sync after every optimiser step.

The important split is:

- shared: trainable parameter arrays, for example adjacency `nzval` and
  `MagField.b`;
- local per worker: graph state, clamping target/mask/beta, RNG, captures, and
  gradient buffers.

Do not create every worker from scratch for this. That rebuilds the process and
can recompile the same composite for each slot. Use `makeworker` once and
`makecontext` for copied slots:

```julia
recipe = (;
    makeworker = (idx, manager) -> make_template_worker(manager.state),
    makecontext = (idx, manager, template) -> begin
        ctx = deepcopy(template.context)
        relink_trainable_parameters!(ctx.dynamics.model, manager.state.params)
        ctx
    end,
    prepare! = (slot, job, manager) -> begin
        write_job!(slot.worker, job)
        Processes.resetworker!(slot)
    end,
)
```

If the Hamiltonian template would normally copy an array, pass the shared storage
through the existing parameter facilities, for example `NoEnsure(shared_b)`.
Then assert the aliases in `prepare!` while developing:

```julia
SparseArrays.getnzval(adj(worker_graph)) === params.w || error("J is not shared")
getparam(worker_graph.hamiltonian, MagField, :b) === params.b || error("b is not shared")
```

After `Optimisers.update`, keep the shared storage object stable and copy the
updated values into it:

```julia
opt_state, updated = Optimisers.update(opt_state, params, gradient)
params.w .= updated.w
params.b .= updated.b
```

Do not replace `params` with `updated` unless every worker is rebuilt or
relinked. Replacing the container breaks the aliases held by existing workers.

## Common Performance Trap

Fresh `Unique` wrappers create fresh identity types/values. If every worker gets
its own fresh unique sampler, Julia specializes/compiles the same learning
composite repeatedly. In the XOR latency smoke test, using separate
`free_dynamics`, `plus_dynamics`, and `minus_dynamics` sampler identities made
the first manager training `run!` compile for about 35 seconds. Using one merged
`dynamics` entry reduced that phase to about 2 seconds.

## Good Pattern For EqProp Workers

1. Build one composite with free, plus, and minus routines.
2. In every routine, alias the sampler as `dynamics`.
3. In plus/minus routines, restore from `equilibrium_state`, apply input/target,
   set the clamping beta, then run `dynamics`.
4. Capture plus/minus states with stable capture algorithms or explicit state
   fields.
5. Construct persistent workers through `ProcessManager` with a plain
   `makeworker` callback. If slots need specialized contexts, use
   `makecontext`; do not rebuild the whole worker per slot.
6. In `prepare!`, write the sample into `_state`, update only job-local state,
   and call `resetworker!(slot)`.
7. In `consume!`, record diagnostics from `slot.worker.context`.
8. In `flush!`, merge worker-local gradient buffers once.
9. After the optimiser update, synchronize `worker.context.dynamics.model`
   in-place for every worker.

## Debugging A Manager Port

When porting an existing working loop to `ProcessManager`, first recreate the
old behavior exactly:

- same number of workers,
- same graph temperature in worker graphs,
- same RNG/reseeding behavior,
- same gradient buffer scaling,
- same parameter sync target,
- same validation worker behavior.

If metrics stay flat while gradient norms are nonzero, first check parameter
sync. In particular, verify that `worker.context.dynamics.model` changes after
the optimiser step and that validation workers read the synchronized graph.
Do not add custom `copyworker`, custom ids, or subcontext reinitialization as a
first response; those change the semantics and can hide the actual mismatch.
