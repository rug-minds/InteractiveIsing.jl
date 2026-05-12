export LocalProposer

struct LocalProposer{I,L,S,D} <: AbstractProposer
    index_set::I
    layers::L
    state::S
    delta::D
end

"""
    LocalProposer(delta)

Local proposal object for `IsingGraph(..., LocalProposer(delta), ...)`.

For continuous layers, `delta` is the maximum absolute value change.
For discrete layers, `delta` is interpreted as the maximum index step in the
layer's `StateSet`.
"""
@inline LocalProposer(delta::Number) = LocalProposer(nothing, nothing, nothing, delta)

@inline statetype(proposer::LocalProposer) = eltype(proposer.state)

function LocalProposer(g::AbstractIsingGraph, delta)
    idx_set = index_set(g)
    _layers = layers(g)
    return LocalProposer(idx_set, _layers, g, delta)
end

@inline bind_proposer(g::AbstractIsingGraph, proposer::LocalProposer) = LocalProposer(g, proposer.delta)

@inline function _within_stateset(x, states)
    return first(states) <= x <= last(states)
end

@inline function _reflect_to_stateset(x, states)
    lo = first(states)
    hi = last(states)
    width = hi - lo

    width <= zero(width) && return lo

    y = mod(x - lo, 2 * width)
    return y <= width ? lo + y : hi - (y - width)
end

@inline function _local_continuous_state(rng::AbstractRNG, layer, old_state, delta)
    step = abs(delta)
    step == zero(step) && return old_state

    states = stateset(layer)
    proposal_state = old_state + (2 * rand(rng, typeof(old_state)) - one(typeof(old_state))) * step
    return _reflect_to_stateset(proposal_state, states)
end

@inline function _state_index(states, old_state)
    idx = findfirst(==(old_state), states)
    isnothing(idx) && throw(ArgumentError("old_state $old_state is not in layer StateSet $states"))
    return idx
end

@inline function _local_discrete_state(rng::AbstractRNG, layer, old_state, delta)
    states = stateset(layer)
    nstates = length(states)
    nstates <= 1 && return old_state

    max_step = max(1, floor(Int, abs(delta)))
    idx = _state_index(states, old_state)
    step = rand(rng, 1:max_step)
    step = rand(rng, Bool) ? step : -step
    newidx = idx + step

    # Stay put when the local index step leaves the StateSet. This preserves
    # symmetric non-null transition probabilities without a Hastings correction.
    return 1 <= newidx <= nstates ? states[newidx] : old_state
end

@inline function localrandstate(rng::AbstractRNG, layer::IL, old_state, delta) where {IL <: AbstractIsingLayer}
    SType = statetype(layer)
    if SType isa Discrete
        return _local_discrete_state(rng, layer, old_state, delta)
    elseif SType isa Continuous
        return _local_continuous_state(rng, layer, old_state, delta)
    else
        error("Unknown statetype for layer sampling")
    end
end

@inline function Base.rand(rng::AbstractRNG, proposer::LocalProposer)
    j = @inline pick_idx(rng, proposer.index_set)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[j]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(j, proposer.layers)
    proposal_state = @inline inline_layer_dispatch(
        layer -> (@inline localrandstate(rng, layer, oldstate, proposer.delta)),
        layer_idx,
        proposer.layers,
    )
    return FlipProposal{statetype(proposer)}(j, oldstate, proposal_state, layer_idx, false)::FlipProposal{statetype(proposer)}
end

@inline Base.rand(proposer::LocalProposer) = rand(Random.default_rng(), proposer)
