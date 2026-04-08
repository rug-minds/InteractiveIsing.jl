#=
Index pickers are used with proposers.

They can be used to specify which parts of the system are part of a montecarlo update scheme
=#

export UniformIndexPicker, pick_idx 

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

include("ToggledIndexSet.jl")
include("ToggledLayerIndexSet.jl")
include("Extensions.jl")


