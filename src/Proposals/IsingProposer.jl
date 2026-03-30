
struct IsingGraphProposer{I,L,S} <: AbstractProposer
    index_set::I
    layers::L
    state::S
end

ed(proposer::IsingGraphProposer) = proposer.accepted_state

@inline statetype(proposer::IsingGraphProposer) = eltype(proposer.state)

function get_proposer(g::AbstractIsingGraph)
    idx_set = index_set(g)
    _layers = layers(g)
    return IsingGraphProposer(idx_set, _layers, g)
end

@inline function Base.rand(rng::AbstractRNG, proposer::IsingGraphProposer)
    j = @inline pick_idx(rng, proposer.index_set)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[j]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(j, proposer.layers)
    proposal_state = @inline inline_layer_dispatch(layer -> (@inline randstate(rng, layer, oldstate)), layer_idx, proposer.layers)
    return FlipProposal{statetype(proposer)}(j, oldstate, proposal_state, layer_idx, false)::FlipProposal{statetype(proposer)}
end

@inline Base.rand(proposer::IsingGraphProposer) = rand(Random.default_rng(), proposer)

random_proposal(g::AbstractIsingGraph) = rand(MersenneTwister(), get_proposer(g))
