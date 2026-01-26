### SAMPLING
@inline function randstate(rng::AbstractRNG, layer::IsingLayer{SType, StateSet}, old_state) where {SType, StateSet}
    if SType isa Discrete
        if length(StateSet) == 2
            return old_state == StateSet[1] ? StateSet[2] : StateSet[1]
        else
            N = length(StateSet)
            r = rand(rng, 1:N-1)
            if StateSet[r] != old_state
                return StateSet[r]
            else
                return StateSet[N]
            end
        end
    elseif SType isa Continuous
        return @inline uniform_rand(rng, StateSet[1], StateSet[end])
    else
        error("Unknown statetype for layer sampling")
    end
end

# ### GENERATING STATE
@inline Base.rand(layer::IsingLayer{StateType}) where {StateType} = sample_from_stateset(StateType, stateset(layer))
@inline Base.rand(layer::IsingLayer{StateType}, num::Integer) where {StateType} =  sample_from_stateset(StateType, stateset(layer), num)

@inline sample_from_stateset(::Any, stateset::T) where T = rand(stateset)
@inline sample_from_stateset(::Discrete, stateset::T) where T = rand(stateset)
@inline sample_from_stateset(::Continuous, stateset::T) where T = uniform_rand(stateset[1], stateset[2])
@inline sample_from_stateset(::Discrete, stateset::T, num) where T = rand(stateset, num)
@inline sample_from_stateset(::Continuous, stateset::T, num) where T = uniform_rand(stateset[1], stateset[2], num)


# @inline sample_from_stateset(::Any, rng::AbstractRNG, stateset::AbstractVector) = rand(rng, stateset)
# @inline sample_from_stateset(::Discrete, rng::AbstractRNG, stateset::Tuple) = rand(rng, stateset)

# @inline function sample_from_stateset(::Continuous, rng::AbstractRNG, stateset::Tuple)
#     uniform_rand(rng, stateset[1], stateset[end])
# end

# @inline sampleState(::Any, oldstate, rng, stateset) = rand(stateset)

# @inline function sampleState(::Discrete, oldstate, rng, stateset)
#     if length(stateset) == 2
#         return oldstate == stateset[1] ? stateset[2] : stateset[1]
#     else
#         N = length(stateset)
#         r = rand(1:N-1)
#         if stateset[r] != oldstate
#             return stateset[r]
#         else
#             return stateset[N]
#         end
#     end
# end
# @inline function sampleState(::Continuous, oldstate, rng, stateset)
#     num = sample_from_stateset(Continuous(), rng, stateset)
#     return num
# end
