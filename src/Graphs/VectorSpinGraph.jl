export VectorSpinGraph, spin_dimension, spin_state_type, vectorstate, vector_unit_norm

"""
    VectorSpinGraph

Graph model for classical vector-valued spins.

Each graph node stores one `SVector{D,T}` spin, while `eltype(model)` remains
the scalar floating-point precision `T`. Keeping scalar precision as the model
element type preserves the existing Monte Carlo acceptance code, which expects
temperatures and energy differences to be scalar values. Set `unit_norm = true`
for classical `n`-vector / `O(n)` spins, or `unit_norm = false` for bounded
component vectors with variable magnitude.
"""
mutable struct VectorSpinGraph{T<:AbstractFloat,D,M,Layers,A,N,I} <: AbstractVectorSpinGraph{T,D}
    state::A
    adj::M
    temp::T
    proposer::AbstractProposer
    default_algorithm::ProcessAlgorithm
    hamiltonian::Hamiltonian
    index_set::I
    addons::Dict{Symbol,Any}
    layers::Layers

    function VectorSpinGraph(
        state::A,
        adj::M,
        temp::T,
        proposer::AbstractProposer,
        default_algorithm::ProcessAlgorithm,
        hamiltonian::Hamiltonian,
        index_set::I,
        addons::Dict{Symbol,Any},
        layers::Layers,
    ) where {T<:AbstractFloat,D,M,Layers,A<:AbstractVector{<:SVector{D,T}},I}
        return new{T,D,M,Layers,A,length(layers),I}(
            state,
            adj,
            temp,
            proposer,
            default_algorithm,
            hamiltonian,
            index_set,
            addons,
            layers,
        )
    end
end

Base.eltype(::VectorSpinGraph{T}) where {T} = T
Base.eltype(::Type{VectorSpinGraph{T}}) where {T} = T
Base.eltype(::Type{<:VectorSpinGraph{T}}) where {T} = T
Base.eltype(::AbstractVectorSpinGraph{T}) where {T} = T
Base.eltype(::Type{<:AbstractVectorSpinGraph{T}}) where {T} = T

"""
    spin_dimension(model)

Return the number of scalar components in each vector spin.
"""
@inline spin_dimension(::AbstractVectorSpinGraph{T,D}) where {T,D} = D
@inline spin_dimension(::Type{<:AbstractVectorSpinGraph{T,D}}) where {T,D} = D

"""
    spin_state_type(model)

Return the concrete static-vector storage type used for each spin.
"""
@inline spin_state_type(::G) where {T,D,G<:AbstractVectorSpinGraph{T,D}} = SVector{D,T}

"""
    vector_unit_norm(model)

Return whether vector-spin proposals and random initialization keep every spin
on the unit sphere.
"""
@inline vector_unit_norm(g::G) where {G<:AbstractVectorSpinGraph} = Bool(get(g, :unit_norm, true))

@inline state(g::VectorSpinGraph) = getfield(g, :state)
@inline graphstate(g::VectorSpinGraph) = getfield(g, :state)
@inline vectorstate(g::VectorSpinGraph) = graphstate(g)
@inline adj(g::VectorSpinGraph) = getfield(g, :adj)
@inline temp(g::VectorSpinGraph) = getfield(g, :temp)
@inline temp!(g::VectorSpinGraph, val) = setfield!(g, :temp, convert(eltype(g), val))
@inline hamiltonian(g::VectorSpinGraph) = getfield(g, :hamiltonian)
@inline index_set(g::VectorSpinGraph) = getfield(g, :index_set)
@inline addons(g::VectorSpinGraph) = getfield(g, :addons)
@inline layer(g::VectorSpinGraph, idx) = getfield(g, :layers)[idx]
@inline Base.getindex(g::VectorSpinGraph, idx) = IsingLayer(g, idx)
@inline Base.getindex(g::VectorSpinGraph) = IsingLayer(g, 1)
@inline Base.length(g::VectorSpinGraph) = length(getfield(g, :layers))
@inline Base.lastindex(g::VectorSpinGraph) = length(g)
@inline Base.size(g::VectorSpinGraph)::Tuple{Int32,Int32} = (Int32(nstates(g)), Int32(1))
@inline layers(g::VectorSpinGraph) = ntuple(i -> IsingLayer(g, i), length(g))
@inline layer_idxs(g::VectorSpinGraph) = range.(layers(g))
@inline graphidxs(g::VectorSpinGraph) = Int32(1):Int32(nstates(g))
@inline graph(g::VectorSpinGraph) = g
@inline nstates(g::VectorSpinGraph) = length(state(g))::Int
@inline nspins(g::VectorSpinGraph) = length(state(g))::Int
@inline nnodes(g::VectorSpinGraph) = length(state(g))::Int
@inline nStates(g::VectorSpinGraph) = length(state(g))::Int
@inline statelen(g::VectorSpinGraph) = length(state(g))
@inline sampling_indices(g::VectorSpinGraph) = sampling_indices(index_set(g))
@inline consume_changed!(g::VectorSpinGraph) = consume_changed!(index_set(g))
Base.get!(g::VectorSpinGraph, s, d) = get!(addons(g), s, d)
Base.get(g::VectorSpinGraph, s) = get(addons(g), s)
Base.get(g::VectorSpinGraph, s, d) = get(addons(g), s, d)

hasDefects(g::VectorSpinGraph) = hasDefects(index_set(g))
aliveList(g::VectorSpinGraph) = aliveList(index_set(g))
defectList(g::VectorSpinGraph) = defectList(index_set(g))
layerdefects(g::VectorSpinGraph) = layerdefects(index_set(g))
isDefect(g::VectorSpinGraph) = isDefect(index_set(g))

function processes(g::VectorSpinGraph)
    get!(g, :processes, Process[])::Vector{Process}
end

function process(g::VectorSpinGraph)
    ps = get!(g, :processes, Process[])
    isempty(ps) && return nothing
    return ps[end]
end

Processes.context(g::VectorSpinGraph) = context(process(g))

function Base.show(io::IO, g::VectorSpinGraph)
    print(io, "VectorSpinGraph{", eltype(g), ",", spin_dimension(g), "} with ", nstates(g), " spins")
    for (idx, layer) in enumerate(layers(g))
        print(io, "\n")
        Base.show(io, layer)
        idx == length(g) || print(io, "\n")
    end
end

"""
    initVectorSpinState(g, initial_state)

Return initialized vector-spin storage for `g`.
"""
function initVectorSpinState(g::G, initial_state) where {G<:AbstractVectorSpinGraph}
    if isnothing(initial_state)
        return initRandomState(g)
    elseif initial_state isa AbstractMatrix
        size(initial_state, 1) == spin_dimension(g) ||
            throw(ArgumentError("initial_state matrix first dimension $(size(initial_state, 1)) does not match spin dimension $(spin_dimension(g))"))
        size(initial_state, 2) == nstates(g) ||
            throw(ArgumentError("initial_state matrix second dimension $(size(initial_state, 2)) does not match graph state length $(nstates(g))"))
        return [spin_state_type(g)(ntuple(k -> convert(eltype(g), initial_state[k, i]), Val(spin_dimension(g)))) for i in 1:nstates(g)]
    elseif initial_state isa AbstractVector
        length(initial_state) == nstates(g) ||
            throw(ArgumentError("initial_state length $(length(initial_state)) does not match graph state length $(nstates(g))"))
        return spin_state_type(g).(initial_state)
    elseif initial_state isa Function
        result = initial_state(g)
        isnothing(result) && return copy(graphstate(g))
        return initVectorSpinState(g, result)
    else
        throw(ArgumentError("Unsupported vector-spin initial_state $(typeof(initial_state)); pass nothing, a vector, a D-by-N matrix, or a function."))
    end
end

"""
    initRandomState(g::AbstractVectorSpinGraph)

Initialize every vector spin independently.

Unit-norm graphs sample the spin sphere. Bounded vector graphs sample each
component from the owning layer's `StateSet`, so spin magnitudes are part of the
state.
"""
function initRandomState(g::G) where {G<:AbstractVectorSpinGraph}
    rng = Random.default_rng()
    _state = similar(graphstate(g))
    for layer in layers(g)
        layer_view = @view _state[graphidxs(layer)]
        if vector_unit_norm(g)
            layer_view .= (random_unit_vector(rng, Val(spin_dimension(g)), eltype(g)) for _ in eachindex(layer_view))
        else
            states = stateset(layer)
            lo = convert(eltype(g), first(states))
            hi = convert(eltype(g), last(states))
            @inbounds for i in eachindex(layer_view)
                layer_view[i] = spin_state_type(g)(ntuple(_ -> lo + (hi - lo) * rand(rng, eltype(g)), Val(spin_dimension(g))))
            end
        end
    end
    return _state
end

function reset!(g::VectorSpinGraph)
    state(g) .= initRandomState(g)
end
