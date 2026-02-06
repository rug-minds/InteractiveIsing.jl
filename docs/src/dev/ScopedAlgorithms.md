# IdentifiableAlgos (Developer Notes)

Scoped algorithms are the backend identity and naming layer used by the Processes system.
They provide a stable, registry-friendly wrapper around algorithm instances while keeping
matching and scoping explicit.

## Core type

`IdentifiableAlgo{F, Name, Id}` wraps a concrete algorithm instance and carries:

- `Name`: a symbol used as the stable key in registries and contexts.
- `Id`: optional identity used to distinguish otherwise identical algorithms.

See `deps/Processes/src/Scoped/IdentifiableAlgos.jl`.

## Construction and naming

Common constructors and helpers:

- `IdentifiableAlgo(f, name::Symbol, id = nothing)`: wrap `f` with explicit name/id.
  If `f` is already scoped, it is re-scoped to the new name.
- `Autokey(f, i::Int, prefix = "", id = nothing)`: generate a name like
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

`IdentifiableAlgo` is a thin container:

- `thincontainer(::Type{<:IdentifiableAlgo}) = true`
- `contained_type` and `_unwrap_container` expose the underlying algorithm type/value.

This allows the thincontainer system to treat wrapped and unwrapped values as the same
identity for matching and registry lookups.

## API surface (backend)

Key functions used throughout Processes:

- `getkey(sa::IdentifiableAlgo)` / `getkey(::Type{<:IdentifiableAlgo})`
- `getalgo(sa::IdentifiableAlgo)` / `getalgo(sa::IdentifiableAlgo)`
- `id(sa::IdentifiableAlgo)` / `hasid(sa::IdentifiableAlgo)`
- `isdefault(sa::IdentifiableAlgo)`
- `changename(sa::IdentifiableAlgo, newname::Symbol)`
- `replacecontextkeys(sa::IdentifiableAlgo, name_replacement::Pair)`
- `IdentifiableAlgo_label(sa::IdentifiableAlgo)` (for display)

Note: matching is centered on `isinstance` and thincontainer unwrapping rather than
structural equality. This is deliberate to keep registry keys stable even when values
are wrapped or routed through other container layers.
