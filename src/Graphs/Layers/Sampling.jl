### SAMPLING
@inline sample_from_stateset(::Any, stateset::AbstractVector) = rand(stateset)
@inline sample_from_stateset(::Discrete, stateset::Tuple) = rand(stateset)
@inline sample_from_stateset(::Continuous, stateset::Tuple) = uniform_rand(stateset[1], stateset[2])
@inline sample_from_stateset(::Discrete, stateset::Tuple, num) = rand(stateset, num)
@inline sample_from_stateset(::Continuous, stateset::Tuple, num) = uniform_rand(stateset[1], stateset[2], num)


@inline sample_from_stateset(::Any, rng::AbstractRNG, stateset::AbstractVector) = rand(rng, stateset)
@inline sample_from_stateset(::Discrete, rng::AbstractRNG, stateset::Tuple) = rand(rng, stateset)
@inline sample_from_stateset(::Continuous, rng::AbstractRNG, stateset::Tuple) = uniform_rand(rng, stateset[1], stateset[end])

@inline sampleState(::Any, oldstate, rng, stateset) = rand(stateset)

@inline function sampleState(::Discrete, oldstate, rng, stateset)
    if length(stateset) == 2
        return oldstate == stateset[1] ? stateset[2] : stateset[1]
    else
        N = length(stateset)
        r = rand(1:N-1)
        if stateset[r] != oldstate
            return stateset[r]
        else
            return stateset[N]
        end
    end
end
@inline sampleState(::Continuous, oldstate, rng, stateset) = sample_from_stateset(Continuous(), rng, stateset)
