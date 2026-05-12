# Registry and Scoping (Developer Notes)

The processes layer uses a static registry to map algorithm instances (and other scoped
values) to stable symbol keys that are later used by contexts. The primary registry type
is `NameSpaceRegistry`.

## What the registry is for

`NameSpaceRegistry` exists mainly to:

1. Orchestrate names between entries that were previously locally managed or assigned.
2. Auto-name entries that do not carry an assigned scope on entry, so scoping is opt-in
   for the user interface while still always enforced internally.
3. Provide static lookup by using values of `isbits` structs (and, via indirection,
   non-`isbits` values) as keys, and converting them into symbol keys used by contexts.

Each entry also tracks a multiplier. The multiplier is used later to infer how often an
algorithm is called relative to the top level of a composed algorithm.

## ScopedValue and matching

The registry expects entries to contain `ScopedValue` wrappers only. A `ScopedValue`
encodes a name and optional identity (`id`) while wrapping a concrete value or algorithm.
These wrappers are the basis for matching. The matching rules are driven by the registry
entries and work together with thin containers so that identity is stable under wrapping.

Auto-scoped values (currently denoted by a `nothing` id, possibly changing to `:auto`)
can be matched by unrolling to the base struct. This supports the opt-in naming model:
users can supply a plain struct and do not need to manage scope explicitly, while the
registry still assigns stable names in the background.

## Thin containers and identity

Thin containers form a lightweight wrapper system. When a value is wrapped by any number
of thin containers, the identity used by the registry is still the identity of the
unwrapped value. This means that the same underlying algorithm or value can be recognized
consistently even when wrapped by multiple layers.
