"""
Multi Layer Constructor
"""
function IsingGraph(layers::Union{IsingLayerData, Hamiltonian}...;
    precision = Float32, 
    adj = nothing,
    self = :nothing # :nothing, :homogeneous, :full
    )

    #Parse hamiltonian and filter
    ham, layers = type_parse(Hamiltonian, layers...; default = Ising(), error = false)

    lengths = map(l -> length(l), layers)
    total_length = sum(lengths)

    # Fix the layers first
    layers = ntuple(length(layers)) do i
        oldlayer = layers[i]
        offset = 0
        if i != 1
            offset = sum(lengths[1:(i-1)])
        end
        newlayer = fix_layerdata(oldlayer, precision, offset)
    end

    if isnothing(adj)
        sparse_connections = init_connections_from_layers(precision, total_length, layers...)
        if self == :homogeneous
            self = FillArray(precision(0.0), total_length)
        elseif self == :nothing
            self = StaticFill(precision(0.0), total_length)
        elseif self == :full
            self = zeros(precision, total_length)
        elseif self isa AbstractVector
            @assert length(self) == total_length "Length of self vector must match total number of nodes in the graph"
        else
            error("Invalid value for `self` keyword argument: $self. Must be :nothing, :homogeneous, :full, or a vector of appropriate length.")
        end

        adj = UndirectedAdjacency(sparse_connections, self)
    end

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
    g.defects.graph = g
    initRandomState(g)
    g.hamiltonian = reconstruct(ham, g)
    return g
end

"""
Single Layer Constructor
"""
function IsingGraph(size1::Int, args...; periodic = true, precision = Float32, adj = nothing, self = :nothing)
    ham, args = type_parse(Hamiltonian, args...; default = Ising(), error = false)

    layer = parse_isinglayer(size1, args...; periodic = periodic)

    return IsingGraph(ham, layer; precision, adj, self )
end
