### SAMPLING
@inline function randstate(rng::AbstractRNG, layer::IL, old_state) where {IL <: IsingLayer}
    SType = statetype(layer)
    StateSet = stateset(layer)
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