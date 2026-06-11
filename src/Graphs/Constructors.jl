"""
    apply_field_defaults(ham, g, field_defaults)

After `instantiate`, override Hamiltonian fields with user-specified (Type, fill_value) defaults.
Each kwarg maps a field name to a `(Type, fill_value)` tuple or a `Fill(val)`.

Example: `y = (Vector, 0.5f0)` overwrites Clamping's `y` field with a vector filled with 0.5.
"""
function apply_field_defaults(ham::HamiltonianTerms, g::AbstractIsingGraph, field_defaults)
    isempty(field_defaults) && return ham
    hams = hamiltonians(ham)
    newhams = map(hams) do h
        apply_field_defaults(h, g, field_defaults)
    end
    return HamiltonianTerms(newhams...)
end

function apply_field_defaults(h::Hamiltonian, g::AbstractIsingGraph, field_defaults)
    fnames = fieldnames(typeof(h))
    changed = false
    newfields = map(fnames) do fname
        if haskey(field_defaults, fname)
            spec = field_defaults[fname]
            changed = true
            return _make_field(spec, g)
        else
            return getfield(h, fname)
        end
    end
    changed || return h
    return typeof(h).name.wrapper(newfields...)
end

function _make_field(spec::Tuple{Type, Any}, g::AbstractIsingGraph)
    T = eltype(g)
    ArrayType, fillval = spec
    if ArrayType <: AbstractVector
        return ArrayType{T}(fill(convert(T, fillval), nstates(g)))
    else
        return fill(convert(T, fillval), nstates(g))
    end
end

function _make_field(spec::Fill, g::AbstractIsingGraph)
    T = eltype(g)
    return fill(convert(T, spec.val), nstates(g))
end

_hamiltonian_arg_summary(h::Hamiltonian) = string(nameof(typeof(h)))

function _hamiltonian_arg_summary(hts::HamiltonianTerms)
    return string("HamiltonianTerms(", join(_hamiltonian_arg_summary.(hamiltonians(hts)), " + "), ")")
end

function _parse_hamiltonian_constructor_arg(args...)
    ham_idxs = findall(x -> x isa Hamiltonian, args)
    isempty(ham_idxs) && return Ising(), args

    if length(ham_idxs) > 1
        hams = join(_hamiltonian_arg_summary.(args[ham_idxs]), ", ")
        throw(ArgumentError(
            "IsingGraph accepts a single Hamiltonian argument, but got $(length(ham_idxs)): $hams. " *
            "Combine Hamiltonian terms with `+` in one expression, and check for an accidental comma between terms.",
        ))
    end

    ham_idx = only(ham_idxs)
    ham = args[ham_idx]
    args = remove_optional_parsed_arg(args, ham_idx)
    return ham, args
end

function _parse_multilayer_constructor_args(args)
    layers = filter(x -> x isa IsingLayerData, args)
    between_layer_wgs = Tuple{Int, AbstractWeightGenerator}[]

    parsed_layers = 0
    pending_wg = nothing
    for arg in args
        if arg isa IsingLayerData
            parsed_layers += 1
            if !isnothing(pending_wg)
                push!(between_layer_wgs, (parsed_layers - 1, pending_wg))
                pending_wg = nothing
            end
        else
            parsed_layers == 0 && throw(ArgumentError("A between-layer weight generator must come after a layer in the IsingGraph constructor."))
            isnothing(pending_wg) || throw(ArgumentError("Consecutive between-layer weight generators are not supported in the IsingGraph constructor."))
            pending_wg = arg
        end
    end

    isnothing(pending_wg) || throw(ArgumentError("A between-layer weight generator must be followed by another layer in the IsingGraph constructor."))
    return layers, between_layer_wgs
end

"""
    graph_constructor_state_storage(state, precision, total_length)

Return `(storage, should_initialize)` for the public `IsingGraph` constructors.
When `state === nothing`, allocate the default dense storage and mark it for
normal initialization. When a custom state vector is supplied, validate that it
can be used as graph storage and preserve it as-is unless `initial_state` is
also supplied by the caller.
"""
function graph_constructor_state_storage(state_storage, precision, total_length)
    if isnothing(state_storage)
        return zeros(precision, total_length), true
    end
    state_storage isa AbstractVector ||
        throw(ArgumentError("state must be an AbstractVector; got $(typeof(state_storage))."))
    length(state_storage) == total_length ||
        throw(ArgumentError("state length $(length(state_storage)) does not match graph state length $total_length"))
    eltype(state_storage) <: AbstractFloat ||
        throw(ArgumentError("state eltype must be an AbstractFloat; got $(eltype(state_storage))."))
    return state_storage, false
end

"""
    graph_constructor_temperature(precision, temperature, temp)

Resolve the public graph-constructor temperature aliases and return the initial
graph temperature converted to the graph storage precision.
"""
function graph_constructor_temperature(::Type{P}, temperature::Temperature, temp::Temp) where {P <: AbstractFloat, Temperature, Temp}
    # Keep the two accepted spellings explicit so accidental conflicting
    # constructor inputs do not silently choose one temperature.
    if !isnothing(temperature) && !isnothing(temp)
        throw(ArgumentError("Pass only one of `temperature` or `temp` to IsingGraph."))
    end

    initial_temperature = isnothing(temperature) ? temp : temperature
    isnothing(initial_temperature) && return P(1.0)
    initial_temperature isa Real ||
        throw(ArgumentError("Initial graph temperature must be Real; got $(typeof(initial_temperature))."))
    return P(initial_temperature)
end

"""
    instantiate_adjacency_from_triplets(adjtype, rows, cols, vals, total_length; diag, fastwrite)

Build the graph adjacency from generated sparse triplets using the requested
adjacency storage type.
"""
function instantiate_adjacency_from_triplets(
    ::Type{T},
    rows,
    cols,
    vals,
    total_length::I;
    diag = nothing,
    fastwrite::B = false,
) where {T <: UndirectedAdjacency, I <: Integer, B <: Bool}
    sparse_connections = sparse(rows, cols, vals, total_length, total_length)
    return UndirectedAdjacency(sparse_connections, diag; fastwrite)
end

function instantiate_adjacency_from_triplets(
    adjtype::Type{T},
    rows,
    cols,
    vals,
    total_length::I;
    diag = nothing,
    fastwrite::B = false,
) where {T <: AbstractSparseMatrix, I <: Integer, B <: Bool}
    sparse_connections = sparse(rows, cols, vals, total_length, total_length)
    return convert(adjtype, sparse_connections)
end

"""
    IsingGraph(layers...; precision = Float32, temperature = nothing, temp = nothing, state = nothing, initial_state = nothing, ...)

Construct a graph from one or more layers, with optional between-layer weight
generators between adjacent layer arguments. `state` may be any
`AbstractVector{<:AbstractFloat}` with the total graph length; it is used
directly as the graph state storage. `initial_state` controls values written
into that storage. `temperature` sets the graph's initial temperature, and
`temp` is accepted as a short alias. If custom `state` is supplied without
`initial_state`, its current contents are preserved.
"""
function IsingGraph(layers::Union{IsingLayerData, AbstractWeightGenerator, Hamiltonian, AbstractProposer}...;
    precision = Float32, 
    adj = nothing,
    diag = StateLike(OffsetArray, 0),
    index_set = nothing,
    initial_state = nothing,
    temperature = nothing,
    temp = nothing,
    state = nothing,
    fastwrite = false,
    callback! = identity,
    )

    #Parse hamiltonian and filter
    ham, layers = _parse_hamiltonian_constructor_arg(layers...)
    proposer, layers = type_parse(AbstractProposer, layers...; default = IsingGraphProposer(), error = false)
    layers, between_layer_wgs = _parse_multilayer_constructor_args(layers)
    lengths = map(l -> length(l), layers)
    total_length = sum(lengths)
    state_storage, initialize_state = graph_constructor_state_storage(state, precision, total_length)
    graph_precision = eltype(state_storage)
    initial_temperature = graph_constructor_temperature(graph_precision, temperature, temp)

    # Fix the layers first
    layers = ntuple(length(layers)) do i
        oldlayer = layers[i]
        offset = 0
        if i != 1
            offset = sum(lengths[1:(i-1)])
        end
        newlayer = fix_layerdata(oldlayer, precision, offset)
    end

    # sparse_connections = init_connections_from_layers(precision, total_length, layers...)

    g_for_shape =  IsingGraph(
        # State
        state_storage,
        # Adjacency
        UndirectedAdjacency(total_length, total_length),
        # Temp
        initial_temperature,
        # Proposer
        proposer,
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        EmptyHamiltonian(),
        #Defects
        1:total_length,
        Dict{Symbol, Any}(),
        layers
    )

    if isnothing(adj) || adj isa Type
        rows, cols, vals = init_connection_triplets_from_layers(precision, total_length, layers...)
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

    if !isnothing(index_set)
        it = index_set(g_for_shape)
    else
        it = 1:total_length
    end

    g_with_adj = IsingGraph(
        # State
        state_storage,
        # Adjacency
        adj,
        # Temp
        initial_temperature,
        # Proposer
        proposer,
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        EmptyHamiltonian(),
        #Defects
        it,
        Dict{Symbol, Any}(),
        layers
    )

    ham  = instantiate(ham, g_with_adj)
    
    # Construct the graph
    g = IsingGraph(
        # State
        state_storage,
        # Adjacency
        adj,
        # Temp
        initial_temperature,
        # Proposer
        proposer,
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        ham,
        #Defects
        it,
        Dict{Symbol, Any}(),
        layers
    )
    
    if initialize_state || !isnothing(initial_state)
        graphstate(g) .= initState(g, initial_state)
    end
    # g.hamiltonian = instantiate(ham, g)
    callback!(g)
    return g
end

"""
    IsingGraph(size1::Int, args...; temperature = nothing, temp = nothing, ...)

Construct a single-layer graph from dimension arguments and layer options.
`temperature` sets the graph's initial temperature, while `temp` is the short
alias accepted by the layer-based constructor.
"""
function IsingGraph(size1::Int, args...; periodic = true, precision = Float32, adj = nothing, diag = StateLike(OffsetArray, 0), initial_state = nothing, temperature = nothing, temp = nothing, state = nothing, fastwrite = false)
    ham, args = _parse_hamiltonian_constructor_arg(args...)
    proposer, args = type_parse(AbstractProposer, args...; default = IsingGraphProposer(), error = false)

    layer = parse_isinglayer(size1, args...; periodic = periodic)

    return IsingGraph(ham, proposer, layer; precision, adj, diag, initial_state, temperature, temp, state, fastwrite)
end
