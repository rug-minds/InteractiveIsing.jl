@inline function uniform_rand(rng::AbstractRNG, rangebegin::T, rangeend::T) where T
    return rand(rng, T)*(rangeend-rangebegin) + rangebegin
end

@inline function uniform_rand(rangebegin::T, rangeend::T) where T
    return rand(T)*(rangeend-rangebegin) + rangebegin
end

@inline function uniform_rand(rng::AbstractRNG, rangebegin::T, rangeend::T, num) where T
    return (rand(rng,T, num)).*(rangeend-rangebegin) .+ rangebegin
end

@inline function uniform_rand(rangebegin::T, rangeend::T, num) where T
    return (rand(T, num)).*(rangeend-rangebegin) .+ rangebegin
end

function reset_global_rng!(seed = nothing)
    newrng = nothing
    if isnothing(seed)
        rng = MersenneTwister()
    else
        rng = MersenneTwister(seed)
    end
    copyfields!(rng, newrng)
end

