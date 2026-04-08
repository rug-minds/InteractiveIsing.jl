@inline function pick_idx(rng::R, u::UnitRange{T}) where {R <: AbstractRNG,T}
    @inline rand(rng, u)
end

