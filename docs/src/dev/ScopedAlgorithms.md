# ScopedAlgorithms (Developer Notes)

Scoped algorithms are the backend identity and naming layer used by the Processes system.
They provide a stable, registry-friendly wrapper around algorithm instances while keeping
matching and scoping explicit.

## Core type

`ScopedAlgorithm{F, Name, Id}` wraps a concrete algorithm instance and carries:

- `Name`: a symbol used as the stable key in registries and contexts.
- `Id`: optional identity used to distinguish otherwise identical algorithms.

See `deps/Processes/src/Scoped/ScopedAlgorithms.jl`.

## Construction and naming

Common constructors and helpers:

- `ScopedAlgorithm(f, name::Symbol, id = nothing)`: wrap `f` with explicit name/id.
  If `f` is already scoped, it is re-scoped to the new name.
- `Autoname(f, i::Int, prefix = "", id = nothing)`: generate a name like
  `$(prefix)$(nameof(typeof(f)))_$(i)` and keep the optional `id`.
- `DefaultScope(f, prefix = "")`: name like `$(prefix)$(nameof(typeof(f)))_0` and
  set `id = :default`.
- `Unique(f)`: assign a fresh UUID-based id to force uniqueness.

## Identity and matching

Matching is defined via `isinstance` and `id` semantics:

- If either scoped value has a non-`nothing` id, both ids must match.
- If both ids are `nothing`, matching falls back to instance identity (`===`) of the
  underlying algorithm value.
- Default instances (`id = :default`) are treated as explicitly scoped.

These semantics are used by the registry when determining if two scoped values represent
the same logical algorithm.

## Container behavior

`ScopedAlgorithm` is a thin container:

- `thincontainer(::Type{<:ScopedAlgorithm}) = true`
- `contained_type` and `_unwrap_container` expose the underlying algorithm type/value.

This allows the thincontainer system to treat wrapped and unwrapped values as the same
identity for matching and registry lookups.

## API surface (backend)

Key functions used throughout Processes:

- `getname(sa::ScopedAlgorithm)` / `getname(::Type{<:ScopedAlgorithm})`
- `getalgorithm(sa::ScopedAlgorithm)` / `getfunc(sa::ScopedAlgorithm)`
- `id(sa::ScopedAlgorithm)` / `hasid(sa::ScopedAlgorithm)`
- `isdefault(sa::ScopedAlgorithm)`
- `changename(sa::ScopedAlgorithm, newname::Symbol)`
- `replacename(sa::ScopedAlgorithm, name_replacement::Pair)`
- `scopedalgorithm_label(sa::ScopedAlgorithm)` (for display)

Note: matching is centered on `isinstance` and thincontainer unwrapping rather than
structural equality. This is deliberate to keep registry keys stable even when values
are wrapped or routed through other container layers.
