"""
    apply_field_defaults(ham, g, field_defaults)

After `reconstruct`, override Hamiltonian fields with user-specified (Type, fill_value) defaults.
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

"""
Multi Layer Constructor
    First the Layers,
    If there's weight generators in between the layers, use those to set connections between layers


"""
function IsingGraph(layers::Union{IsingLayerData, AbstractWeightGenerator, Hamiltonian}...;
    precision = Float32, 
    adj = nothing,
    diag = StateLike(OffsetArray, 0),
    fastwrite = false
    )

    #Parse hamiltonian and filter
    ham, layers = type_parse(Hamiltonian, layers...; default = Ising(), error = false)

    lengths = map(l -> length(l), layers)
    total_length = sum(lengths)

    # First note all weightgenerator indices
    wg_indices = findall(l -> l isa AbstractWeightGenerator, layers)
    # Setup (idx-)

    # Fix the layers first
    layers = ntuple(length(layers)) do i
        oldlayer = layers[i]
        offset = 0
        if i != 1
            offset = sum(lengths[1:(i-1)])
        end
        newlayer = fix_layerdata(oldlayer, precision, offset)
    end

    sparse_connections = init_connections_from_layers(precision, total_length, layers...)

    g_for_shape =  IsingGraph(
        # State
        zeros(precision, total_length),
        # Adjacency
        UndirectedAdjacency(total_length, total_length),
        # Temp
        precision(1.0),
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        EmptyHamiltonian(),
        #Defects
        GraphDefects(nothing),
        Dict{Symbol, Any}(),
        layers
    )

    if isnothing(adj)
        sparse_connections = init_connections_from_layers(precision, total_length, layers...)
        diag = diag(g_for_shape)
        adj = UndirectedAdjacency(sparse_connections, diag; fastwrite)
    else
        @assert size(adj, 1) == total_length "Adjacency matrix size must match total number of nodes in the graph\nexpected $(total_length)x$(total_length), got $(size(adj))"
    end

    g_with_adj = IsingGraph(
        # State
        zeros(precision, total_length),
        # Adjacency
        adj,
        # Temp
        precision(1.0),
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        EmptyHamiltonian(),
        #Defects
        GraphDefects(nothing),
        Dict{Symbol, Any}(),
        layers
    )

    ham  = reconstruct(ham, g_with_adj)


    # Construct the graph
    g = IsingGraph(
        # State
        zeros(precision, total_length),
        # Adjacency
        adj,
        # Temp
        precision(1.0),
        # Default Algo
        IsingMetropolis(),
        #Hamiltonians
        ham,
        #Defects
        GraphDefects(nothing),
        Dict{Symbol, Any}(),
        layers
    )
    graph!(defects(g), g)
    initRandomState(g)
    # g.hamiltonian = reconstruct(ham, g)
    return g
end

"""
Single Layer Constructor
"""
function IsingGraph(size1::Int, args...; periodic = true, precision = Float32, adj = nothing, diag = StateLike(OffsetArray, 0), fastwrite = false)
    ham, args = type_parse(Hamiltonian, args...; default = Ising(), error = false)

    layer = parse_isinglayer(size1, args...; periodic = periodic)

    return IsingGraph(ham, layer; precision, adj, diag)
end
