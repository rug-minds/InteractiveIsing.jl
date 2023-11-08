"""
ndefect not tracking correctly FIX
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
mutable struct IsingLayer{T, StateSet, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    # Reference to the graph holding it    
    graph::Union{Nothing, IsingGraphType}
    name::String
    # Internal idx of layer in shufflevec of graph
    internal_idx::Int32
    start::Int32
    size::Tuple{Int32,Int32}
    nstates::Int32

    coords::Coords{Tuple{Int32,Int32,Int32}}
    
    # Keeps track of the connected layers
    connections::Dict{Pair{Int32,Int32}, Any} 

    timers::Vector{PTimer}

    # defects::LayerDefects
    top::LayerTopology

    # DEFAULT INIT
    function IsingLayer{ST, SS}(g, name, internal_idx, start, size, nstates, coords, connections, timers, top) where {ST, SS}
        return new{ST, SS, typeof(g)}(g, name, internal_idx, start, size, nstates, coords, connections, timers, top)
    end


    function IsingLayer(
            StateType, 
            g::GraphType, 
            idx, 
            start, 
            length, 
            width; 
            set = convert.(eltype(g),(-1,1)), 
            name = "$(length)x$(width) Layer", 
            coords = Coords(nothing), 
            connections = Dict{Pair{Int32,Int32}, 
            WeightGenerator}(), 
            periodic::Union{Nothing,Bool} = true
        ) where GraphType <: IsingGraph
            
        lsize = tuple(Int32(length), Int32(width))
        set = convert.(eltype(g), set)
        layer = new{StateType, set, GraphType}(
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
            # PTimers
            Vector{PTimer}(),
        )
        layer.top = LayerTopology(layer, [1,0], [0,1]; periodic)
        finalizer(destructor, layer)
        return layer
    end
end
export IsingLayer


# TODO: Is this still needed?
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

function destructor(layer::IsingLayer)
    close.(timers(layer))
end


@setterGetter IsingLayer coords size layeridx graph
@setterGetter IsingLayerCopy coords size

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::IsingLayer{A,B}) where {A,B}
    showstr = "$A IsingLayer $(layeridx(layer)) with size $(size(layer)) and stateset $B\n"
    if coords(layer) != nothing
        showstr *= " at coordinates $(coords(layer))"
    end
    print(io, showstr, "\n")
    println(io, " with connections:")
    for key in keys(connections(layer))
        println(io, "\tConnected to layer $(key[2]) using ")
        println("\t", (connections(layer)[key]))
    end
    print(io, " and $(ndefect(layer)) defects")
    return
end

# SHOW THE TYPE
Base.show(io::IO, layertype::Type{IsingLayer{A,B}}) where {A,B} = print(io, "$A IsingLayer")


## ACCESSORS
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

function conns(l1::IsingLayer, l2::IsingLayer)
    return sp_adj(graph(l1))[graphidxs(l1), graphidxs(l2)]
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

"""
Resize a layer
Is this used?
"""
function Base.resize!(layer::IsingLayer, len, wid)
    println("This one is called")
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
        println("HIER")
        sp_adj(g, insertrowcol(g, _endidx+1:(_endidx+1 + extra_states)))
    else # extra_states < 0
        notidxs = graphidxs(layer)[end+extra_states+1:end]
        deleteat!(state(g), _startidx:_endidx)
        sp_adj(g, deleterowcol(g, notidxs))
    end
    return layer
end

"""
Get graph of layer
"""
@inline function graph(layer::IsingLayer{<:Any,<:Any, G})::G where G
    g = layer.graph::G
    return g
end

"""
Set graph of layer
"""
@inline graph(layer::IsingLayer, g::IsingGraph) = layer.graph = g 


# Get current layeridx through graph
@inline layeridx(layer::IsingLayer) = externalidx(layers(graph(layer)), layer.internal_idx)
@inline idx(layer::IsingLayer) = internal_idx(layer)

### COORDINATES OF LAYERS
@inline coords(layer::AbstractIsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::AbstractIsingLayer, val) = (layer.coords.cs = Int32.((val,val,val)))
export setcoords!

"""
Get adjacency of layer in layer coordinates
"""
@inline reladj(layer::AbstractIsingLayer) = adjGToL(layer.adj, layer)

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
bvec(layer::AbstractIsingLayer) = (@view bfield(graph(layer))[graphidxs(layer)])
clamps(layer::AbstractIsingLayer) = reshape((@view clamps(graph(layer))[graphidxs(layer)]), size(layer,1), size(layer,2))
clampsvec(layer::AbstractIsingLayer) = (@view clamps(graph(layer))[graphidxs(layer)])
export bfield, bvec
# Inherited from Graph
@inline nStates(layer::AbstractIsingLayer) = length(graphidxs(layer))
@inline sim(layer::AbstractIsingLayer) = sim(graph(layer))

##
function aliveidxs(layer::AbstractIsingLayer)
    ds = defects(graph(layer))
    preceding_defects = sum(ds.layerdefects[1:layer.internal_idx-1])
    these_defects = ds.layerdefects[layer.internal_idx]
    alivelist_range = (startidx(layer)-preceding_defects):(endidx(layer)-preceding_defects-these_defects)
    aliveList(ds)[alivelist_range]
end
export aliveidxs


### TIMERS
    pausetimers(layer) = close.(timers(layer))
    starttimers(layer) = start.(timers(layer))
    removetimers(layer) = begin close.(timers(layer)); layer.timers = Vector{PTimer}(); end

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
    @inline ndefect(layer::AbstractIsingLayer) = defects(graph(layer)).layerdefects[layer.internal_idx]
    @inline nalive(layer::AbstractIsingLayer) = nStates(layer) - ndefect(layer)
    export ndefect, nalive
    @inline hasDefects(layer::AbstractIsingLayer) = ndefect(layer) > 0
    @inline setdefect(layer::AbstractIsingLayer, val, idx) = defects(graph(layer))[idxLToG(idx, layer)] = val
    @inline clamprange!(layer::AbstractIsingLayer, val, idxs) = clamprange!(defects(graph(layer)), val, idxLToG.(idxs, Ref(layer)))
###

### RELOCATING
### Shift from placing 1 layer befor
"""
When shifting a layer by one index,
Copy over the state to the right position, except the adjacency matrix
"""
function relocate!(movable_layer::IsingLayer, causing_layer::IsingLayer, shift, copy::Bool = true)
    oldstate_view = state(movable_layer)
    movable_layer.start += shift*nStates(causing_layer)
    movable_layer.internal_idx += shift*1
    if copy  
        state(movable_layer) .= oldstate_view
    end
end

### GET INDEXES
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



### STATE SET
function changeset(l::IsingLayer{SType}, set) where SType
    _eltype = eltype(graph(l))
    newset = convert.(_eltype, set)
    g = graph(l)
    # newlayer = IsingLayer(SType, l.graph, l.internal_idx, l.start, l.size[1], l.size[2], name = l.name, coords = l.coords, connections = l.connections, rangebegin = set[1], rangeend = set[2])
    newlayer = IsingLayer{SType, newset}(g, l.name, l.internal_idx, l.start, l.size, l.nstates,  l.coords, l.connections, l.timers, l.top)
    newlayer.timers = l.timers

    return newlayer
end

function changeset!(l, set)
    _layers = layers(graph(l))
    _layeridx = layeridx(l)
    _layers[_layeridx] = changeset(l, set)
    notify(l)
    return _layers[_layeridx]
end
export changeset, changeset!

stateset(l::IsingLayer{<:Any, SS}) where SS = SS
stateset(::Type{IsingLayer{A, SS, B}}) where {A, SS, B} = SS
function extremiseDiscrete!(l::IsingLayer{ST, SS}) where {ST,SS}
    if ST == Discrete
        a = SS[1]
        b = SS[2]
        map!(x -> x >= (a+b)/2f0 ? b : a, state(l), state(l))
    end
    return state(l)
end
function extremise!(l::IsingLayer{ST, SS}) where {ST,SS}
    a = SS[1]
    b = SS[2]
    map!(x -> x >= (a+b)/2f0 ? b : a, state(l), state(l))
end

export changeset!, stateset

### TYPE STUFF
## DEFAULT NEW LAYER TYPE BASED ON GRAPH
default_ltype(g::IsingGraph{T}) where T = T == Int8 ? Discrete : Continuous 
@inline statetype(layer::IsingLayer{ST}) where {ST} = ST
@inline statetype(::Type{IsingLayer{ST,A,B}}) where {ST,A,B} = ST
setstatetype(l::IsingLayer{ST,SET,GT}, stype) where {ST,SET,GT} = IsingLayer{stype,SET}(l.graph, l.name, l.internal_idx, l.start, l.size, l.nstates, l.coords, l.connections, l.timers, l.top)

Base.eltype(l::IsingLayer{<:Any, <:Any, GT}) where GT = eltype(GT)

# ORDER LAYER TYPES BASED ON STATETYPE
Base.isless(::Type{IsingLayer{A,B,C}}, ::Type{IsingLayer{D,E,F}}) where {A,B,C,D,E,F} = isless(A,D)


export statetype, setstatetype

### GENERATING STATE
@inline Base.rand(layer::IsingLayer{StateType, StateSet}) where {StateType, StateSet} = sample_from_stateset(StateType, StateSet)
@inline Base.rand(layer::IsingLayer{StateType, StateSet}, num::Integer) where {StateType, StateSet} =  sample_from_stateset(StateType, StateSet, num)


@inline function initstate!(layer::IsingLayer)
    state(layer)[:] .= rand(layer, nStates(layer))
end