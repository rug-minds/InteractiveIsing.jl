"""
    graph_constructor_vector_state_storage(state_storage, precision, dimension, total_length)

Return `(storage, should_initialize)` for `VectorSpinGraph` constructors.

When no storage is supplied, dense `Vector{SVector{D,T}}` storage is allocated
and marked for random initialization. Supplied storage is preserved as-is after
validating length and vector dimension.
"""
function graph_constructor_vector_state_storage(
    state_storage,
    ::Type{T},
    dimension::D,
    total_length::I,
) where {T<:AbstractFloat,D<:Integer,I<:Integer}
    dimension > 0 || throw(ArgumentError("VectorSpinGraph dimension must be positive, got $dimension"))
    if isnothing(state_storage)
        return Vector{SVector{dimension,T}}(undef, total_length), true
    end

    state_storage isa AbstractVector ||
        throw(ArgumentError("state must be an AbstractVector; got $(typeof(state_storage))."))
    length(state_storage) == total_length ||
        throw(ArgumentError("state length $(length(state_storage)) does not match graph state length $total_length"))

    S = eltype(state_storage)
    S <: StaticVector ||
        throw(ArgumentError("VectorSpinGraph state eltype must be a StaticVector; got $S."))
    length(S) == dimension ||
        throw(ArgumentError("state spin dimension $(length(S)) does not match requested dimension $dimension"))
    eltype(S) <: AbstractFloat ||
        throw(ArgumentError("state scalar eltype must be an AbstractFloat; got $(eltype(S))."))
    return state_storage, false
end

function _parse_vector_spin_hamiltonian_constructor_arg(args...)
    ham_idxs = findall(x -> x isa Hamiltonian, args)
    isempty(ham_idxs) && return VectorSpin(), args

    if length(ham_idxs) > 1
        hams = join(_hamiltonian_arg_summary.(args[ham_idxs]), ", ")
        throw(ArgumentError(
            "VectorSpinGraph accepts a single Hamiltonian argument, but got $(length(ham_idxs)): $hams. " *
            "Combine Hamiltonian terms with `+` in one expression, and check for an accidental comma between terms.",
        ))
    end

    ham_idx = only(ham_idxs)
    ham = args[ham_idx]
    if ham isa typeof(Ising())
        throw(ArgumentError("VectorSpinGraph uses vector-valued spins; pass `VectorSpin(...)`, `VectorExchange(...)`, `VectorField(...)`, or another vector-compatible Hamiltonian instead of scalar `Ising(...)`."))
    end

    args = remove_optional_parsed_arg(args, ham_idx)
    return ham, args
end

"""
    VectorSpinGraph(layers...; dimension, precision = Float32, ...)

Construct a vector-spin graph.

The graph topology and layer metadata follow the existing `IsingGraph`
constructors. Each node state is an `SVector{dimension,precision}`. The default
`unit_norm = true` gives classical `n`-vector / `O(n)` spins; `unit_norm = false`
keeps components bounded by the layer `StateSet` and allows variable magnitude.
The default Hamiltonian is [`VectorSpin`](@ref), i.e. exchange interaction plus
optional vector field.
"""
function VectorSpinGraph(
    layers::Union{IsingLayerData,AbstractWeightGenerator,Hamiltonian,AbstractProposer}...;
    dimension::Integer,
    precision = Float32,
    adj = nothing,
    diag = StateLike(OffsetArray, 0),
    index_set = nothing,
    initial_state = nothing,
    state = nothing,
    fastwrite = false,
    callback! = identity,
    unit_norm = true,
)
    ham, layers = _parse_vector_spin_hamiltonian_constructor_arg(layers...)
    proposer, layers = type_parse(AbstractProposer, layers...; default = VectorSpinProposer(), error = false)
    layers, between_layer_wgs = _parse_multilayer_constructor_args(layers)
    lengths = map(l -> length(l), layers)
    total_length = sum(lengths)
    state_storage, initialize_state = graph_constructor_vector_state_storage(state, precision, dimension, total_length)
    graph_precision = eltype(eltype(state_storage))
    graph_addons = Dict{Symbol,Any}(:unit_norm => Bool(unit_norm))

    # Fix layer ranges against the scalar precision used by the vector spins.
    layers = ntuple(length(layers)) do i
        oldlayer = layers[i]
        offset = i == 1 ? 0 : sum(lengths[1:(i - 1)])
        fix_layerdata(oldlayer, graph_precision, offset)
    end

    g_for_shape = VectorSpinGraph(
        state_storage,
        UndirectedAdjacency(total_length, total_length),
        graph_precision(1.0),
        proposer,
        IsingMetropolis(),
        EmptyHamiltonian(),
        1:total_length,
        copy(graph_addons),
        layers,
    )

    if isnothing(adj) || adj isa Type
        rows, cols, vals = init_connection_triplets_from_layers(graph_precision, total_length, layers...)
        for (layer_idx, wg) in between_layer_wgs
            layerrows, layercols, layervals = genLayerConnections(g_for_shape[layer_idx], g_for_shape[layer_idx + 1], wg)
            append!(rows, layerrows)
            append!(cols, layercols)
            append!(vals, layervals)
        end
        diag = diag(g_for_shape)
        adjtype = isnothing(adj) ? UndirectedAdjacency : adj
        adj = instantiate_adjacency_from_triplets(adjtype, rows, cols, vals, total_length; diag, fastwrite)
    else
        @assert size(adj, 1) == total_length "Adjacency matrix size must match total number of nodes in the graph\nexpected $(total_length)x$(total_length), got $(size(adj))"
    end

    it = isnothing(index_set) ? (1:total_length) : index_set(g_for_shape)

    g_with_adj = VectorSpinGraph(
        state_storage,
        adj,
        graph_precision(1.0),
        proposer,
        IsingMetropolis(),
        EmptyHamiltonian(),
        it,
        copy(graph_addons),
        layers,
    )

    ham = instantiate(ham, g_with_adj)

    g = VectorSpinGraph(
        state_storage,
        adj,
        graph_precision(1.0),
        proposer,
        IsingMetropolis(),
        ham,
        it,
        copy(graph_addons),
        layers,
    )

    if initialize_state || !isnothing(initial_state)
        graphstate(g) .= initVectorSpinState(g, initial_state)
    end
    callback!(g)
    return g
end

"""
    VectorSpinGraph(size1, args...; dimension, periodic = true, ...)

Construct a one-dimensional vector-spin graph using the same layer parser as
the scalar `IsingGraph` convenience constructor.
"""
function VectorSpinGraph(
    size1::Int,
    args...;
    dimension::Integer,
    periodic = true,
    precision = Float32,
    adj = nothing,
    diag = StateLike(OffsetArray, 0),
    initial_state = nothing,
    state = nothing,
    fastwrite = false,
    unit_norm = true,
)
    ham, args = _parse_vector_spin_hamiltonian_constructor_arg(args...)
    proposer, args = type_parse(AbstractProposer, args...; default = VectorSpinProposer(), error = false)
    layer = parse_isinglayer(size1, args...; periodic)
    return VectorSpinGraph(ham, proposer, layer; dimension, precision, adj, diag, initial_state, state, fastwrite, unit_norm)
end
