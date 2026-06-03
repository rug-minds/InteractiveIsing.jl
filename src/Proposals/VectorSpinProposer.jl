export VectorSpinProposer, VectorSpinLocalProposer, random_unit_vector

"""
    VectorSpinProposer()

Single-site proposal for vector-spin models.

The proposer samples an active graph index and replaces that spin with a fresh
state. Unit-norm graphs get a fresh direction on the spin sphere; bounded
component graphs get a fresh vector whose components lie in the layer
`StateSet`.
"""
struct VectorSpinProposer{I,L,S} <: AbstractProposer
    index_set::I
    layers::L
    state::S
end

"""
    VectorSpinLocalProposer(delta)

Local vector-spin proposal that perturbs the current spin.

Unit-norm graphs renormalize after adding Gaussian noise. Bounded component
graphs perturb each component and reflect it back into the layer `StateSet`,
so both direction and magnitude can evolve locally.
"""
struct VectorSpinLocalProposer{I,L,S,D} <: AbstractProposer
    index_set::I
    layers::L
    state::S
    delta::D
end

VectorSpinProposer() = VectorSpinProposer(nothing, nothing, nothing)
VectorSpinLocalProposer(delta::Number) = VectorSpinLocalProposer(nothing, nothing, nothing, delta)

@inline statetype(proposer::VectorSpinProposer) = eltype(InteractiveIsing.state(proposer.state))
@inline statetype(proposer::VectorSpinLocalProposer) = eltype(InteractiveIsing.state(proposer.state))

"""
    random_unit_vector(rng, Val(D), Type{T})

Draw a `D`-component static vector with Euclidean norm one.
"""
@inline function random_unit_vector(rng::AbstractRNG, ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
    v = SVector{D,T}(ntuple(_ -> randn(rng, T), Val(D)))
    norm_v = norm(v)
    return iszero(norm_v) ? SVector{D,T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D))) : v / norm_v
end

"""
    random_bounded_vector(rng, Val(D), Type{T}, states)

Draw a `D`-component static vector with each component inside `states`.
"""
@inline function random_bounded_vector(rng::AbstractRNG, ::Val{D}, ::Type{T}, states) where {D,T<:AbstractFloat}
    lo = convert(T, first(states))
    hi = convert(T, last(states))
    return SVector{D,T}(ntuple(_ -> lo + (hi - lo) * rand(rng, T), Val(D)))
end

"""
    random_vector_state(rng, model, layer)

Draw one vector-spin state using the model's unit-norm or bounded-component
state convention.
"""
@inline function random_vector_state(rng::AbstractRNG, g::G, layer::L) where {G<:AbstractVectorSpinGraph,L<:AbstractIsingLayer}
    if vector_unit_norm(g)
        return random_unit_vector(rng, Val(spin_dimension(g)), eltype(g))
    end
    return random_bounded_vector(rng, Val(spin_dimension(g)), eltype(g), stateset(layer))
end

function _default_vector_spin_proposer(g::G) where {G<:AbstractVectorSpinGraph}
    idx_set = index_set(g)
    _layers = layers(g)
    return VectorSpinProposer(idx_set, _layers, g)
end

function _bind_vector_spin_local_proposer(g::G, proposer::VectorSpinLocalProposer) where {G<:AbstractVectorSpinGraph}
    idx_set = index_set(g)
    _layers = layers(g)
    return VectorSpinLocalProposer(idx_set, _layers, g, proposer.delta)
end

@inline bind_proposer(g::G, ::VectorSpinProposer) where {G<:AbstractVectorSpinGraph} = _default_vector_spin_proposer(g)
@inline bind_proposer(g::G, proposer::VectorSpinLocalProposer) where {G<:AbstractVectorSpinGraph} =
    _bind_vector_spin_local_proposer(g, proposer)
@inline bind_proposer(::G, proposer::AbstractProposer) where {G<:AbstractVectorSpinGraph} = proposer

function get_proposer(g::G) where {G<:AbstractVectorSpinGraph}
    proposer = getfield(g, :proposer)
    return bind_proposer(g, proposer)
end

@inline function Base.rand(rng::AbstractRNG, proposer::VectorSpinProposer)
    j = @inline pick_idx(rng, proposer.index_set)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[j]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(j, proposer.layers)
    proposal_state = @inline inline_layer_dispatch(
        layer -> (@inline random_vector_state(rng, proposer.state, layer)),
        layer_idx,
        proposer.layers,
    )
    return FlipProposal(j, oldstate, proposal_state, layer_idx, false)::FlipProposal{statetype(proposer)}
end

@inline function _local_unit_vector(rng::AbstractRNG, oldstate::SVector{D,T}, delta) where {D,T}
    noise = SVector{D,T}(ntuple(_ -> randn(rng, T), Val(D)))
    trial = oldstate + T(delta) * noise
    norm_trial = norm(trial)
    return iszero(norm_trial) ? oldstate : trial / norm_trial
end

"""
    _local_bounded_vector(rng, layer, oldstate, delta)

Perturb every component of a vector spin and reflect components into the
layer's bounded `StateSet`.
"""
@inline function _local_bounded_vector(rng::AbstractRNG, layer::L, oldstate::SVector{D,T}, delta) where {L<:AbstractIsingLayer,D,T<:AbstractFloat}
    step = abs(T(delta))
    step == zero(T) && return oldstate

    states = stateset(layer)
    return SVector{D,T}(ntuple(Val(D)) do k
        proposal_component = oldstate[k] + (2 * rand(rng, T) - one(T)) * step
        convert(T, _reflect_to_stateset(proposal_component, states))
    end)
end

"""
    local_vector_state(rng, model, layer, oldstate, delta)

Draw one local vector-spin proposal using the model's state convention.
"""
@inline function local_vector_state(
    rng::AbstractRNG,
    g::G,
    layer::L,
    oldstate::SVector{D,T},
    delta,
) where {G<:AbstractVectorSpinGraph,L<:AbstractIsingLayer,D,T<:AbstractFloat}
    if vector_unit_norm(g)
        return _local_unit_vector(rng, oldstate, delta)
    end
    return _local_bounded_vector(rng, layer, oldstate, delta)
end

@inline function Base.rand(rng::AbstractRNG, proposer::VectorSpinLocalProposer)
    j = @inline pick_idx(rng, proposer.index_set)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[j]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(j, proposer.layers)
    proposal_state = @inline inline_layer_dispatch(
        layer -> (@inline local_vector_state(rng, proposer.state, layer, oldstate, proposer.delta)),
        layer_idx,
        proposer.layers,
    )
    return FlipProposal(j, oldstate, proposal_state, layer_idx, false)::FlipProposal{statetype(proposer)}
end

@inline Base.rand(proposer::VectorSpinProposer) = rand(Random.default_rng(), proposer)
@inline Base.rand(proposer::VectorSpinLocalProposer) = rand(Random.default_rng(), proposer)

random_proposal(g::G) where {G<:AbstractVectorSpinGraph} = rand(MersenneTwister(), get_proposer(g))
