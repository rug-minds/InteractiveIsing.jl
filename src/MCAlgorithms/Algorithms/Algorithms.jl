### SAMPLING
@inline sample_from_stateset(::Any, stateset::AbstractVector) = rand(stateset)
@inline sample_from_stateset(::Type{Discrete}, stateset::Tuple) = rand(stateset)
@inline sample_from_stateset(::Type{Continuous}, stateset::Tuple) = uniform_rand(stateset[1], stateset[2])
@inline sample_from_stateset(::Type{Discrete}, stateset::Tuple, num) = rand(stateset, num)
@inline sample_from_stateset(::Type{Continuous}, stateset::Tuple, num) = uniform_rand(stateset[1], stateset[2], num)


@inline sample_from_stateset(::Any, rng::AbstractRNG, stateset::AbstractVector) = rand(rng, stateset)
@inline sample_from_stateset(::Type{Discrete}, rng::AbstractRNG, stateset::Tuple) = rand(rng, stateset)
@inline sample_from_stateset(::Type{Continuous}, rng::AbstractRNG, stateset::Tuple) = uniform_rand(rng, stateset[1], stateset[2])

@inline sampleState(::Any, oldstate, rng, stateset) = rand(stateset)
@inline sampleState(::Type{Discrete}, oldstate, rng, stateset) = oldstate == stateset[1] ? stateset[2] : stateset[1]
@inline sampleState(::Type{Continuous}, oldstate, rng, stateset) = sample_from_stateset(Continuous, rng, stateset)

include("Required/Required.jl")
include("Metropolis.jl")
include("LayeredMetropolis.jl")
include("SweepMetropolis.jl")

