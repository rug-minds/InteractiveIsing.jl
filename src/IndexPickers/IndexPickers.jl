#=
Index pickers are used with proposers.

They can be used to specify which parts of the system are part of a montecarlo update scheme
=#

export UniformIndexPicker, pick_idx, sampling_indices

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

include("ToggledIndexSet.jl")
include("ToggledLayerIndexSet.jl")
include("Extensions.jl")

