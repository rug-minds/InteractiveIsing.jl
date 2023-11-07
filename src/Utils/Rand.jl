@inline function uniform_rand(rng::AbstractRNG, rangebegin, rangeend)
    return rand(rng)*(rangeend-rangebegin) + rangebegin
end

@inline function uniform_rand(rangebegin, rangeend)
    return rand()*(rangeend-rangebegin) + rangebegin
end

@inline function uniform_rand(rng::AbstractRNG, rangebegin, rangeend, num)
    return (rand(rng, num)).*(rangeend-rangebegin) .+ rangebegin
end

@inline function uniform_rand(rangebegin, rangeend, num)
    return (rand(num)).*(rangeend-rangebegin) .+ rangebegin
end

