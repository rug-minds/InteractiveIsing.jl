"""
ndefects not tracking correctly FIX
"""

# Coordinates that can be left uninitialized
mutable struct Coords{T}
    cs :: Union{Nothing,T}
end

Coords(;y = 0, x = 0, z = 0) = Coords{Tuple{Int32,Int32,Int32}}((Int32(y), Int32(x), Int32(z)))
Coords(n::Nothing) = Coords{Tuple{Int32,Int32,Int32}}(nothing)
Coords(val::Integer) = Coords{Tuple{Int32,Int32,Int32}}((Int32(val), Int32(val), Int32(val)))
export Coords
# TODO: Make the topology part of the layertype
mutable struct IsingLayer{T, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    graph::Union{Nothing, IsingGraphType}
    name::String
    # Internal idx of layer in shufflevec of graph
    internal_idx::Int32
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}

    connections::Dict{Pair{Int32,Int32}, Any} 

    timers::Vector{Timer}

    # defects::LayerDefects
    top::LayerTopology


    function IsingLayer(LayerType, g::GraphType, idx, start, length, width; name = "$(length)x$(width) Layer", coords = Coords(nothing), connections = Dict{Pair{Int32,Int32}, WeightGenerator}(), olddefects = 0, periodic::Union{Nothing,Bool} = true) where GraphType <: IsingGraph
        lsize = tuple(Int32(length), Int32(width))
        layer = new{LayerType, GraphType}(
            # Graph
            g,
            # Name
            name,
            # Layer idx
            Int32(idx),
            # Start idx
            Int32(start),
            # Size
            lsize,
            # Number of states
            Int32(lsize[1]*lsize[2]),
            #Coordinates
            coords,
            # Connections
            connections,
            # Timers
            Vector{Timer}(),
            # Layer data
            # LayerData(d(g), start, length, width)
        )
        # TODO: What's olddefects for
        # layer.defects = LayerDefects(layer, olddefects)
        layer.top = LayerTopology(layer, [1,0], [0,1]; periodic)

        return layer
    end
end
export IsingLayer

# IsingLayer(g, layer::AbstractIsingLayer) = IsingLayer(g, layeridx(layer), start(layer), glength(layer), gwidth(layer), olddefects = ndefects(layer))

mutable struct IsingLayerCopy{T, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    const graph::IsingGraphType
    layeridx::Int32
    state::Matrix{T}
    adj::Matrix{Vector{Tuple{Int32, Float32}}}
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}
    # const d::LayerData
    # defects::LayerDefects
    top::LayerTopology

    function IsingLayerCopy(layer::IsingLayer{A,B}) where {A,B}
        
        new{A, B}(
            # Graph
            layer.graph,
            # Layer idx at the time of copying
            layeridx(layer),
            # State
            copy(state(layer)),
            # Adj
            copy(adj(layer)),
            # Start idx
            layer.start,
            # Size
            layer.size,
            # Number of states
            layer.nstates,
            # Coordinates
            layer.coords,
            # Layer data
            # layer.d,
            # layer.defects,
            layer.top
        )
    end
end



@setterGetter IsingLayer coords size layeridx graph
@setterGetter IsingLayerCopy coords size

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::AbstractIsingLayer)
    showstr = "IsingLayer $(layeridx(layer)) with size $(size(layer))"
    if coords(layer) != nothing
        showstr *= " \t at coordinates $(coords(layer))"
    end
    print(io, showstr, "\n")
    println(io, " with connections:")
    for key in keys(connections(layer))
        println(io, "\tConnected to layer $(key[2]) using ")
        println("\t", (connections(layer)[key]))
    end
    print(io, "and $(ndefects(layer)) defects")
    return
end

Base.show(io::IO, layertype::Type{<:AbstractIsingLayer}) = print(io, "IsingLayer")

@inline state(l::IsingLayer) = reshape((@view state(graph(l))[graphidxs(l)]), size(l,1), size(l,2))
# @inline adj(l::IsingLayer) = reshape((@view adj(graph(l))[graphidxs(l)]), glength(l), gwidth(l))
@inline sp_adj(l::IsingLayer) = @view sp_adj(graph(l))[:, graphidxs(l)] 
@inline function set_sp_adj!(layer, wg, rcw)
    connections(layer)[internal_idx(layer) => internal_idx(layer)] = wg
    set_sp_adj!(graph(layer), rcw)
    notify(layer)
    return sp_adj(graph(layer))
end

@inline function set_sp_adj!(layer1, layer2, wg, rcw)
    connections(layer1)[internal_idx(layer1) => internal_idx(layer2)] = wg
    set_sp_adj!(graph(layer1), rcw)
    notify(layer1)
    return sp_adj(graph(layer1))
end
export state, adj

"""
Get the connections for an idx in the graph
"""
function conns(idx::Integer, g::IsingGraph)
    return sp_adj(g)[:, idx]
end

"""
Get the connections for an idx in the layer, given in graphidxs
"""
function conns(idx::Integer, layer::IsingLayer)
    return sp_adj(graph(layer))[:, idxLToG(idx,layer)]
end

function conns(idx::Integer, layer1::IsingLayer, layer2::IsingLayer)
    return sp_adj(graph(layer1))[:, idxLToG(idx,layer1)][graphidxs(layer2)]
end

# TODO:: Make a way to show the coordinates of the connections
"""
Get the connections for a coordinate of the layer, given in graphidxs
"""
conns(i::Integer, j::Integer, layer::IsingLayer) = conns(coordToIdx(i,j, layer), layer)
"""
Get the connections for a coordinate of the layer, given in layeridxs
"""
conns(i::Integer, j::Integer, layer1::IsingLayer, layer2::IsingLayer) = conns(coordToIdx(i, j, layer1),layer1)[graphidxs(layer2)]
"""
Get the coordinates of all the connected units for a unit at coordinates i,j for layer 1, given in layer coordinates of layer 2
Connections to self can be obtained by setting layer2 = layer1
"""
function conncoords(i::Integer, j::Integer, layer1::IsingLayer, layer2::IsingLayer = layer1)
    _conns = conns(i,j,layer1,layer2)
    return idxToCoord.(_conns.nzind, Ref(layer2))
end
export conns, conncoords

@inline wg(layer::IsingLayer) = try connections(layer)[internal_idx(layer) => internal_idx(layer)]; catch; return ""; end
@inline wg(layer1::IsingLayer, layer2::IsingLayer) = try connections(layer1)[internal_idx(layer1) => internal_idx(layer2)]; catch; return ""; end

function Base.resize!(layer::IsingLayer, len, wid)
    g = graph(layer)
    old_nstates = nStates(layer)
    new_nstates = len*wid
    extra_states = new_nstates - old_nstates
    if extra_states == 0
        return
    end
    _startidx = startidx(layer)
    _endidx = endidx(layer)
    if extra_states > 0
        insert!(state(g), _endidx+1, rand(len*wid))
        sp_adj(g, insertrowcol(g, _endidx+1:(_endidx+1 + extra_states)))
    else # extra_states < 0
        notidxs = graphidxs(layer)[end+extra_states+1:end]
        deleteat!(state(g), _startidx:_endidx)
        sp_adj(g, sp_adj(g)[Not(notidxs), Not(notidxs)])
    end
    return layer
end

# Get Graph
@inline function graph(layer::IsingLayer{T,G})::G where {T,G}
    g::G = layer.graph
    return g
end

@inline graph(layer::IsingLayer, g::IsingGraph) = layer.graph = g 


# Get current layeridx through graph
@inline layeridx(layer::IsingLayer) = externalidx(layers(graph(layer)), layer.internal_idx)
@inline idx(layer::IsingLayer) = internal_idx(layer)

@inline coords(layer::AbstractIsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::AbstractIsingLayer, val) = (layer.coords.cs = Int32.((val,val,val)))
export setcoords!

@inline reladj(layer::AbstractIsingLayer) = adjGToL(layer.adj, layer)
# @forward IsingLayer LayerData d
# @forward IsingLayer LayerDefects defects

# @forward IsingLayerCopy LayerData
# @forward IsingLayerCopy LayerDefects defects

# Setters and getters
# @forward IsingLayer IsingGraph g
@inline size(layer::AbstractIsingLayer)::Tuple{Int32,Int32} = layer.size
@inline size(layer::AbstractIsingLayer, i) = layer.size[i]
@inline glength(layer::AbstractIsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[1]::Int32
@inline gwidth(layer::AbstractIsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[2]::Int32
@inline maxdist(layer::AbstractIsingLayer) = maxdist(layer, periodic(layer))
@inline maxdist(layer::AbstractIsingLayer, ::Type) = max(size(layer)...)
@inline function maxdist(layer::AbstractIsingLayer, ::Type{Periodic})
    l, w = size(layer)
    maxdist = dist(1,1, 1 + l÷2, 1 + w÷2, top(layer))
    return maxdist
end
export maxdist

@inline coordToIdx(i,j,layer::AbstractIsingLayer) = coordToIdx(latmod(i, size(layer,1)), latmod(j, size(layer,2)), size(layer,1))
@inline idxToCoord(idx, layer::AbstractIsingLayer) = idxToCoord(idx, size(layer,1))
@inline c2i(i, j, layer::AbstractIsingLayer) = coordToIdx(i, j, layer)
@inline i2c(i, layer::AbstractIsingLayer) = idxToCoord(i, layer)

@inline startidx(layer::AbstractIsingLayer) = start(layer)
@inline endidx(layer::AbstractIsingLayer) = start(layer) + prod(size(layer)) - 1

@inline getindex(layer::AbstractIsingLayer, idx) = state(layer)[idx]
@inline getindex(layer::AbstractIsingLayer, i, j) = state(layer)[i,j]
@inline setindex!(layer::AbstractIsingLayer, val, idx) = state(layer)[idx] = val
@inline setindex!(layer::AbstractIsingLayer, val, i, j) = state(layer)[i,j] = val

@inline Base.in(idx::Integer, layer::IsingLayer) = idx ∈ graphidxs(layer)


"""
Range of idx of layer for underlying graph
"""
@inline graphidxs(layer::AbstractIsingLayer) = UnitRange{Int32}(start(layer):endidx(layer))
export graphidxs

bfield(layer::AbstractIsingLayer) = reshape((@view bfield(graph(layer))[graphidxs(layer)]), size(layer,1), size(layer,2))
clamps(layer::AbstractIsingLayer) = reshape((@view clamps(graph(layer))[graphidxs(layer)]), size(layer,1), size(layer,2))

# Inherited from Graph
@inline nStates(layer::AbstractIsingLayer) = length(graphidxs(layer))
@inline sim(layer::AbstractIsingLayer) = sim(graph(layer))


### DEFECTS
    """
    Get the indexes of all alive spins in the layer
    """
    aliveList(layer::AbstractIsingLayer) = aliveList(defects(layer))
    """
    Get the indexes of all defect spins in the layer
    """
    defectList(layer::AbstractIsingLayer) = defectList(defects(layer))

    """
    Returns wether layer has any defects
    """
    @inline ndefects(layer::AbstractIsingLayer) = layerdefects(defects(graph(layer)))[internal_idx(layer)]
    export ndefects
    @inline hasDefects(layer::AbstractIsingLayer) = ndefects(layer) > 0
    @inline setdefect(layer::AbstractIsingLayer, val, idx) = defects(graph(layer))[idxLToG(idx, layer)] = val
    @inline clamprange!(layer::AbstractIsingLayer, val, idxs) = clamprange!(defects(graph(layer)), val, idxLToG.(idxs, Ref(layer)))
###

iterator(layer::AbstractIsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# LayerTopology
@inline periodic(layer::AbstractIsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer::AbstractIsingLayer, periodic) = top!(layer, LayerTopology(top(layer); periodic))
@inline dist(idx1::Integer, idx2::Integer, layer::AbstractIsingLayer) = dist(idxToCoord(idx1, glength(layer))..., idxToCoord(idx2, size(layer,1))..., top(layer))
@inline dist(i1::Integer, j1::Integer, i2::Integer, j2::Integer, layer::AbstractIsingLayer) = dist(i1, j1, i2, j2, top(layer))
@inline idxToCoord(idx::Integer, layer::AbstractIsingLayer) = idxToCoord(idx, size(layer,1))

# Simulation stuff
Base.notify(layer::AbstractIsingLayer) = let _sim = sim(layer); notify(layerIdx(_sim)); end

export setPeriodic!


# Forward Graph Data
# @inline bfield(layer::IsingLayer) = @view bfield(graph(layer))[start(layer):endidx(layer)]

"""
Go from a local idx of layer to idx of the underlying graph
"""
@inline function idxLToG(idx::Integer, layer::IsingLayer)::Int32
    return Int32(start(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(i::Integer, j::Integer, layer::IsingLayer)::Int32
    return Int32(start(layer) + coordToIdx(i,j, glength(layer)) - 1)
end

idxLToG(tup::Tuple, layer) = idxLToG(tup[1], tup[2], layer)

"""
Go from graph idx to idx of layer
"""
@inline function idxGToL(idx::Integer, layer::IsingLayer)
    return Int32(idx + 1 - start(layer))
end


# Set the SType
"""
Set the simulation type through a layer
"""
setSType!(layer::AbstractIsingLayer, varargs...; refresh = true) = setSType!(graph(layer), varargs...; refresh = refresh)
