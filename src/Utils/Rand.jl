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

