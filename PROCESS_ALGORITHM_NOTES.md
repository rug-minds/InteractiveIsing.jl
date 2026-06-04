# ProcessAlgorithm Notes

## What Failed

The target-free XOR worker originally used this `@ProcessAlgorithm` signature:

```julia
StatefulAlgorithms.@ProcessAlgorithm function AccumulateTargetFreeGradient!(
    isinggraph::G,
    target_state::S,
    free_state::F,
    buffers::B,
) where {G,S<:AbstractVector,F<:AbstractVector,B}
```

That does not currently expand correctly. Including the file failed at load time
with:

```text
UndefVarError: `S` not defined in `Main`
```

The error came from the generated code in `deps/StatefulAlgorithms/src/ProcessEntities/ProcessAlgorithmsNew.jl`.
The macro did not propagate the `where` type variables for these runtime inputs
into all generated definitions, so the generated implementation referenced `S`
outside the scope where `S` was bound.

## Working Pattern

Keep `@ProcessAlgorithm` functions as thin process-facing wrappers, and put the
specialized implementation in an ordinary Julia function:

```julia
function accumulate_target_free_gradient_body!(
    isinggraph::G,
    target_state::S,
    free_state::F,
    buffers::B,
) where {G,S<:AbstractVector,F<:AbstractVector,B}
    # typed implementation
end

StatefulAlgorithms.@ProcessAlgorithm function AccumulateTargetFreeGradient!(
    isinggraph,
    target_state,
    free_state,
    buffers,
)
    accumulate_target_free_gradient_body!(isinggraph, target_state, free_state, buffers)
    return nothing
end
```

This keeps the process interface macro-simple while still allowing Julia to
specialize the real work through the helper method. It also keeps unit behavior
deployable in routines and composites instead of placing a whole training phase
inside one opaque process step.

## Agent Rule

For this codebase, write `@ProcessAlgorithm` only for unit process operations.
Do not use typed runtime-input `where` signatures directly in the macro until
the macro supports them. Use a typed ordinary helper for the implementation, and
compose those unit algorithms with `@Routine`, `@CompositeAlgorithm`, and manager
loops.

## Routine Capture Gotcha

`@Routine` has a similar hygiene constraint: values from the surrounding script
scope must be copied into routine state or aliases before they are used inside
the generated routine body. This failed:

```julia
target_free_sign = config.target_free_sign

return StatefulAlgorithms.@Routine begin
    AccumulateTargetFreeGradient!(target_free_sign = target_free_sign)
end
```

The generated routine could not resolve `target_free_sign` and threw:

```text
UndefVarError: `target_free_sign` not defined in `Main`
```

Use a distinct outer binding and copy it into routine state:

```julia
sign_value = config.target_free_sign

return StatefulAlgorithms.@Routine begin
    @state target_free_sign = sign_value
    AccumulateTargetFreeGradient!(target_free_sign = target_free_sign)
end
```
