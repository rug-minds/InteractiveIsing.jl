#=
Index pickers are used with proposers.

They can be used to specify which parts of the system are part of a montecarlo update scheme
=#

export UniformIndexPicker, pick_idx, sampling_indices, consume_changed!

"""
Implements pick_idx(rng, u::UniformIndexPicker) to pick indices from a set with Uniform probability, 
and toggle the set on/off
"""
abstract type UniformIndexPicker end
@inline function pick_idx(rng::R, u::U) where {R <: AbstractRNG, U}
    error("pick_idx not implemented for $(typeof(u))")
end
@inline function pick_idx(u::U) where U
    @inline pick_idx(Random.default_rng(), u)
end

"""
Return the currently active indices for sweep-based algorithms.

Algorithms such as `LocalLangevin` need a concrete iterable of all indices that
should be visited in one sweep. Plain ranges, vectors, and sets can use themselves.
Custom `UniformIndexPicker`s should implement this method if they want to support
sweep-style algorithms in addition to `pick_idx`-based proposers.
"""
@inline sampling_indices(idxs) = idxs
@inline function sampling_indices(u::UniformIndexPicker)
    error("sampling_indices not implemented for $(typeof(u)); implement InteractiveIsing.sampling_indices for sweep-based algorithms")
end

"""
    consume_changed!(index_set) -> Bool

Return whether `sampling_indices(index_set)` changed since the last time this
flag was consumed, then clear the flag.

Static index sets can use the default `false`. Mutable index sets that alter
their active sweep list should specialize this method so sweep-based algorithms
can rebuild cached order state without comparing the full index list every step.
"""
@inline consume_changed!(idxs) = false

include("ToggledIndexSet.jl")
include("ToggledLayerIndexSet.jl")
include("Extensions.jl")
