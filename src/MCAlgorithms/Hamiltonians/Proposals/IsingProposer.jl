
struct IsingGraphProposer{I,L,S} <: AbstractProposer
    iterator::I
    layers::L
    state::S
end
ed(proposer::IsingGraphProposer) = proposer.accepted_state

statetype(::IsingGraphProposer{I,L,S}) where {I,L,S} = eltype(S)

function get_proposer(g::AbstractIsingGraph)
    iterator = ising_it(g)
    layers = g.layers
    return IsingGraphProposer(iterator, layers, g.state)
end

function Base.rand(rng::AbstractRNG, proposer::IsingGraphProposer)
    j = rand(rng, proposer.iterator)
    oldstate = proposer.state[j]::statetype(proposer)
    range_ends = range_end.(proposer.layers)
    layer_idx = tuple_searchsortedfirst(range_ends, j)
    proposal_state = @inline inline_layer_dispatch(layer -> randstate(rng, layer, oldstate), layer_idx, proposer.layers)
    return FlipProposal{:s, :j, statetype(proposer)}(j, oldstate, proposal_state, layer_idx, false)
end

function Base.rand(proposer::IsingGraphProposer)
    j = rand(proposer.iterator)
    oldstate = proposer.state[j]::statetype(proposer)
    range_ends = range_end.(proposer.layers)
    layer_idx = tuple_searchsortedfirst(range_ends, j)
    proposal_state = @inline inline_layer_dispatch(layer -> randstate(rng, layer, oldstate), layer_idx, proposer.layers)
    return FlipProposal{:s, :j, statetype(proposer)}(j, oldstate, proposal_state, layer_idx, false)
end
