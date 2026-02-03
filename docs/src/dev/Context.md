# Context (Developer and User Notes)

The context system is the user-facing way to access and share algorithm state during a
process run, while still being driven by backend registry and scoping rules. It centers
around `ProcessContext`, `SubContext`, and `SubContextView`.

See:
- `deps/Processes/src/Context/Context.jl`
- `deps/Processes/src/Context/ProcessContexts.jl`
- `deps/Processes/src/Context/SubContext.jl`
- `deps/Processes/src/Context/SharingInterface.jl`
- `deps/Processes/src/Context/Init.jl`
- `deps/Processes/src/Context/ContextView.jl`

## ProcessContext

`ProcessContext` owns:

- `subcontexts`: a named tuple of `SubContext` objects (plus `globals`)
- `registry`: the `NameSpaceRegistry` that resolves algorithm names to subcontexts

Key access patterns:

- `pc.subcontext_name` or `pc[:subcontext_name]` for direct access by name.
- `pc[algo_instance]` uses the registry to resolve an instance to a subcontext name.
- `getglobal(pc)` / `getglobal(pc, :var)` for globals.

### Construction

- `ProcessContext(algos::LoopAlgorithm; globals = (;))`
  - Builds subcontexts from the algorithm registry.
  - Resolves shares and routes into per-subcontext metadata.
- `ProcessContext(func::Any; globals = (;))`
  - Uses `SimpleRegistry` and a single subcontext.

## SubContext

`SubContext{Name}` stores:

- `data`: local variables (a named tuple)
- `sharedcontexts`: whole-subcontext shares
- `sharedvars`: routed variables (stored internally as `SharedVars`)

You typically do not construct these by hand. They are created by `ProcessContext`.

## SubContextView (user-facing access)

`view(pc, algo)` returns a `SubContextView` for a scoped algorithm instance. From a view:

- Local variables resolve by name (e.g. `scv.M`).
- Shared subcontexts and routed variables are resolved transparently.
- Local variables take precedence over routed, which take precedence over shared.

You can merge into the view to update the context:

- `merge(scv, (; var1 = value, ...))` routes updates to the correct subcontext.
- `replace(scv, (; SubName = (; ...)))` replaces the entire subcontext during prepare.

## How to get keys / names

Names are derived from the registry:

- Scoped algorithms carry a stable name (see `IdentifiableAlgo`).
- When you call `view(pc, instance)`, the registry is consulted to resolve the
  instance to its scoped name.
- Routes can alias variable names, which become the user-facing keys in the view.

In short: user-facing keys are either local variable names, route aliases, or shared
context variable names, all resolved through the registry and view logic.

## Routes and Shares

Routes and shares are defined in `SharingInterface.jl` and resolved at context creation.

### Shares (whole subcontext sharing)

`Share(algo1, algo2; directional = false)` connects two subcontexts at the namespace
level. The result is a `SharedContext{from}` entry on the target subcontext, meaning
the entire subcontext is visible to the other side. If `directional = false`, it is
bi-directional and both sides see each other.

### Routes (variable-level sharing)

`Route(from, to, var_or_pair...)` moves specific variables from one subcontext to
another, optionally with aliases:

- `Route(A, B, :M)` routes `A.M` into `B.M`
- `Route(A, B, :M => :mag)` routes `A.M` into `B.mag`

Routes become `SharedVars{from_name, NT}` metadata on the destination subcontext.
Internally, `NT` is a NamedTuple type that maps original variable names to their
aliases. Alias lookup is done via `get_alias(sv, :original_name)`.
The view uses those aliases as the user-facing keys.

### Resolution

At `ProcessContext` creation:

- `Share` and `Route` are resolved via the registry to concrete subcontext names.
- `sharedcontexts` and `sharedvars` are attached to each `SubContext`.
- `SubContextView` uses that metadata to expose shared and routed values.
