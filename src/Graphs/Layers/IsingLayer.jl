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
mutable struct IsingLayer{StateType, StateSet, Dim, Size, PtrRange, Top, Precision, AdjType} <: AbstractIsingLayer{StateType, Dim}
    # Reference to the graph holding it    
    # Can be nothing so that saving is easier
    graph::Union{Nothing, IsingGraph{Precision, AdjType}}
    name::String
    # Idx of the layer in the graph
    const idx::Int
    # TRAITS, Not used?
    traits::NamedTuple
    coords::Coords{Tuple{Int32,Int32,Int32}}
    # Keeps track of the connected layers
    connections::Dict{Pair{Int32,Int32}, Any} 
    # A layer can hold its own timers (WHY)
    const timers::Vector{PTimer}
    const top::Top   
end

function IsingLayer(
            g::Union{Nothing, IsingGraph},
            lsize,
            idx, 
            start::Int;
            stype = Discrete(), 
            precision = Float32,
            set = convert.(eltype(g),(-1,1)), 
            name = "Layer $idx", 
            # traits = (;StateType = StateType, StateSet = set, Indices = (start:(start+length*width-1)), Hamiltonians = (Ising,)),
            traits = (;),
            coords = Coords(nothing), 
            adjtype = SparseMatrixCSC{precision,Int32},
            wg::Union{WeightGenerator, Nothing} = nothing,
            connections = Dict{Pair{Integer,Integer}, WeightGenerator}(), 
            periodic::Union{Nothing,Bool,Tuple} = true,
            kwargs...
        )

        if !isnothing(wg)
            connections[idx=>idx] = wg
        end

        if stype isa Type
            stype = stype()
        end
    
        dims = length(lsize)
        isnothing(periodic) && (periodic = true)
        top = SquareTopology(lsize; periodic)
        graphidxs = start:(start+reduce(*, lsize)-1)
        set = convert.(eltype(g), set)

        layer = IsingLayer{stype, set, dims, lsize, graphidxs, typeof(top), precision, adjtype}(
            # Graph
            g,
            # Name
            name,
            # Layer idx
            idx,
            # Traits
            traits,
            #Coordinates
            coords,
            # Connections
            connections,
            # PTimers
            Vector{PTimer}(),
            # Topology
            top
        )
        return layer
end

get_weightgenerator(layer::IsingLayer) = get(layer.connections, layer.idx => layer.idx, nothing)
struct LayerProperties <: AbstractLayerProperties
    size::Tuple
    kwargs::NamedTuple
end

datalen(lp::LayerProperties) = prod(lp.size)

function Layer(dims...; kwargs...)
    LayerProperties(tuple(dims...), (;kwargs...))
end

export Layer

IsingLayer(g, idx, startidx, lp::LayerProperties) = IsingLayer(g, lp.size, idx, startidx; lp.kwargs...)

export IsingLayer

"""
Two IsingLayers are equivalent when
"""
equiv(i1::Type{IsingLayer{A,B,T1,Top1}}, i2::Type{IsingLayer{E,F,T2,Top2}}) where {A,B,E,F,T1,T2,Top1,Top2} = A == E && B == F

function layerparams(lt::Type{<:IsingLayer}, ::Val{S}) where S
    if S == :StateType
        return parameters(lt)[1]
    elseif S == :StateSet
        return parameters(lt)[2]
    elseif S == :Dim
        return parameters(lt)[3]
    elseif S == :Size
        return parameters(lt)[4]
    elseif S == :PtrRange
        return parameters(lt)[5]    
    elseif S == :T
        return parameters(lt)[6]
    elseif S == :Top
        return parameters(lt)[7]
    # elseif S == :GT
    #     return parameters(lt)[6]
    end
end

@inline layerparams(l::IsingLayer, s) = layerparams(typeof(l), Val(s))

function destructor(layer::IsingLayer)
    close.(timers(layer))
end


@setterGetter IsingLayer coords size idx graph
stateset(layer::IsingLayer) = layerparams(layer, :StateSet)

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::IsingLayer{A,B}) where {A,B}
    showstr = "$A IsingLayer $(layeridx(layer)) with size $(size(layer)) and stateset $(stateset(layer))\n"
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
@inline state(l::IsingLayer) = reshape((@view state(graph(l))[graphidxs(l)]), size(l)...)

@inline adj(l::IsingLayer) = @view adj(graph(l))[:, graphidxs(l)] 

@inline function set_adj!(layer::IsingLayer, wg::WeightGenerator, rcw)
    # connections(layer)[internal_idx(layer) => internal_idx(layer)] = wg
    connections(layer)[(layeridx(layer) => layeridx(layer))] = wg
    set_adj!(graph(layer), rcw)
    # notify(layer)
    return adj(graph(layer))
end

@inline function set_adj!(layer1::IsingLayer, layer2::IsingLayer, wg::WeightGenerator, rcw)
    connections(layer1)[internal_idx(layer1) => internal_idx(layer2)] = wg
    set_adj!(graph(layer1), rcw)
    notify(layer1)
    return adj(graph(layer1))
end
export state, adj

"""
Get the connections for an idx in the graph
"""
function conns(idx::Integer, g::IsingGraph)
    return adj(g)[:, idx]
end

"""
Get the connections for an idx in the layer, given in graphidxs
"""
function conns(idx::Integer, layer::IsingLayer)
    return adj(graph(layer))[:, idxLToG(idx,layer)]
end

function conns(idx::Integer, layer1::IsingLayer, layer2::IsingLayer)
    return adj(graph(layer1))[:, idxLToG(idx,layer1)][graphidxs(layer2)]
end

function conns(l1::IsingLayer, l2::IsingLayer)
    return adj(graph(l1))[graphidxs(l1), graphidxs(l2)]
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

@inline wg(layer::IsingLayer) = try connections(layer)[internal_idx(layer) => internal_idx(layer)]; catch; return "No weightfunc"; end
@inline wg(layer1::IsingLayer, layer2::IsingLayer) = try connections(layer1)[internal_idx(layer1) => internal_idx(layer2)]; catch; return "No weightfunc"; end

"""
Set graph of layer
"""
@inline graph(layer::IsingLayer) = layer.graph
@inline graph(layer::IsingLayer, g::IsingGraph) = layer.graph = g
# changegraph(l::IsingLayer, g) = IsingLayer{layerparams(l, :StateType), layerparams(l, :DIMS), layerparams(l, :T), layerparams(l, :Top)}(g, l.name, l.internal_idx, l.startidx, l.size, l.nstates, l.traits, l.coords, l.connections, l.timers, l.top)
addons(layer::IsingLayer) = graph(layer).addons
export addons

# Get current layeridx through graph
# @inline layeridx(layer::IsingLayer) = externalidx(layers(graph(layer)), layer.internal_idx)
@inline layeridx(layer::IsingLayer) = layer.idx

### COORDINATES OF LAYERS
@inline coords(layer::AbstractIsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer{T}; x = 0, y = 0, z = 0) where T = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::AbstractIsingLayer{T}, val) where T = (layer.coords.cs = Int32.((val,val,val)))

export setcoords!

"""
Get adjacency of layer in layer coordinates
"""
@inline reladj(layer::AbstractIsingLayer) = adjGToL(layer.adj, layer)

# Setters and getters
# @forwardfields IsingLayer IsingGraph g
@inline Base.size(layer::AbstractIsingLayer{T}) where {T} = layerparams(layer, :Size)
@inline Base.size(layer::AbstractIsingLayer, i) = size(layer)[i]
@inline glength(layer::AbstractIsingLayer) = size(layer,1)
@inline gwidth(layer::AbstractIsingLayer) = size(layer,2)
@inline gheight(layer::AbstractIsingLayer{T,3}) where T = size(layer,3)
# @inline dims(layer::AbstractIsingLayer) = length(size(layer))


@inline maxdist(layer::AbstractIsingLayer) = maxdist(layer, periodic(layer))
@inline maxdist(layer::AbstractIsingLayer, ::Type) = max(size(layer)...)
@inline function maxdist(layer::AbstractIsingLayer, ::Type{Periodic})
    l, w = size(layer)
    maxdist = dist(top(layer), 1,1, 1 + l÷2, 1 + w÷2)
    return maxdist
end
export maxdist

@inline coordToIdx(i,j,layer::AbstractIsingLayer) = coordToIdx(latmod(Int32(i), size(layer,1)), latmod(Int32(j), size(layer,2)), size(layer,1))
@inline idxToCoord(idx, layer::AbstractIsingLayer) = idxToCoord(Int32(idx), size(layer,1))
c2i = coordToIdx
i2c = idxToCoord

# @inline startidx(layer::AbstractIsingLayer) = start(layer)
# @inline endidx(layer::AbstractIsingLayer) = startidx(layer) + prod(size(layer)) - 1
# export endidx

@inline Base.getindex(layer::AbstractIsingLayer{T, 2}, idx) where T = state(layer)[idx]
@inline Base.getindex(layer::AbstractIsingLayer{T, 2}, i, j) where T = state(layer)[i,j]
@inline Base.setindex!(layer::AbstractIsingLayer{T, 2}, val, idx) where T = state(layer)[idx] = val
@inline Base.setindex!(layer::AbstractIsingLayer{T, 2}, val, i, j) where T = state(layer)[i,j] = val

@inline Base.in(idx::Integer, layer::IsingLayer) = idx ∈ graphidxs(layer)


"""
Range of idx of layer for underlying graph
"""
@inline graphidxs(layer::AbstractIsingLayer) = layerparams(layer, :PtrRange)
@inline startidx(layer::AbstractIsingLayer) = graphidxs(layer)[1]
@inline endidx(layer::AbstractIsingLayer) = graphidxs(layer)[end]
export graphidxs, startidx, endidx

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
    @inline function ndefect(layer::AbstractIsingLayer)
        if !isnothing(graph(layer)) 
            return 0
        end
        defects(graph(layer)).layerdefects[layeridx(layer)]
    end
    @inline nalive(layer::AbstractIsingLayer) = nStates(layer) - ndefect(layer)
    export ndefect, nalive
    @inline hasDefects(layer::AbstractIsingLayer) = ndefect(layer) > 0
    @inline setdefect(layer::AbstractIsingLayer, val, idx) = defects(graph(layer))[idxLToG(idx, layer)] = val
    @inline clamprange!(layer::AbstractIsingLayer, val, idxs) = setrange!(defects(graph(layer)), val, idxLToG.(idxs, Ref(layer)))
###

### RESIZING
"""
Resize a layer
Is this used?
"""
# function Base.resize!(layer::IsingLayer, len, wid)
#     g = graph(layer)
#     old_nstates = nStates(layer)
#     new_nstates = len*wid
#     extra_states = new_nstates - old_nstates
#     if extra_states == 0
#         return
#     end
#     _startidx = startidx(layer)
#     _endidx = endidx(layer)
#     if extra_states > 0
#         insert!(state(g), _endidx+1, rand(len*wid))
#         adj(g, insertrowcol(g, _endidx+1:(_endidx+1 + extra_states)))
#     else # extra_states < 0
#         notidxs = graphidxs(layer)[end+extra_states+1:end]
#         deleteat!(state(g), _startidx:_endidx)
#         adj(g, deleterowcol(g, notidxs))
#     end
#     return layer
# end

### RELOCATING
### Shift from placing 1 layer befor
"""
When shifting a layer by one index,
Copy over the state to the right position, except the adjacency matrix
"""
# function relocate!(movable_layer::IsingLayer, causing_layer::IsingLayer, shift, copy::Bool = true)
#     println("Moveable layer: ", movable_layer)
#     println("Causing layer: ", causing_layer)
#     oldstate_view = state(movable_layer)
#     movable_layer.startidx += shift*nStates(causing_layer)
#     movable_layer.internal_idx += shift*1
#     if copy  
#         state(movable_layer) .= oldstate_view
#     end
# end

#TODO: This is a patchwork fix, fix this better
# This is used to update the stateset in the type
# function remake_type(layer::IsingLayer)
#     pars =  typeof(layer).parameters
#     new_idxset = graphidxs(layer)
#     # use this one:
#     #function IsingLayer{ST, SS, IS, DIMS, T, Topology}(g, name, internal_idx, start, size, nstates, coords, connections, timers, top) where {ST, SS, IS, DIMS, T, Topology}
#     # return new{ST, SS, IS, DIMS, T, Topology}(g, name, internal_idx, start, size, nstates, coords, connections, timers, top)
#     return IsingLayer{pars[1], pars[2], new_idxset, pars[4], pars[5], pars[6]}(graph(layer), name(layer), internal_idx(layer), startidx(layer), size(layer), nstates(layer), layer.traits, layer.coords, connections(layer), timers(layer), top(layer))
# end

### GET INDEXES
iterator(layer::AbstractIsingLayer) = startidx(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# LayerTopology
@inline periodic(layer::AbstractIsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer::AbstractIsingLayer, periodic) = top!(layer, LayerTopology(top(layer); periodic))

# TODO: Change this
@inline dist(layer::AbstractIsingLayer, coords...) = dist(top(layer), coords...)
# @inline dist(i1::Integer, j1::Integer, i2::Integer, j2::Integer, k1, k2, layer::AbstractIsingLayer{T,3}) where T = sqrt((i1-i2)^2 + (j1-j2)^2 + (k1-k2)^2)
@inline idxToCoord(idx::Integer, layer::AbstractIsingLayer) = idxToCoord(idx, size(layer))

# Notify a change in the simulation
# TODO: Make this an observable in the graph that can be coupled with the one in the simulation
# Base.notify(layer::AbstractIsingLayer) = let _sim = sim(layer); if !isnothing(_sim); notify(layerIdx(_sim)) end; end

export setPeriodic!


# Forward Graph Data
# @inline bfield(layer::IsingLayer) = @view bfield(graph(layer))[start(layer):endidx(layer)]

"""
Go from a local idx of layer to idx of the underlying graph
"""
@inline function idxLToG(idx::Integer, layer::IsingLayer)
    return Int32(startidx(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(i::Integer, j::Integer, layer::IsingLayer)
    return Int32(startidx(layer) + coordToIdx(i,j, glength(layer)) - 1)
end

idxLToG(tup::Tuple, layer) = idxLToG(tup[1], tup[2], layer)

"""
Go from graph idx to idx of layer
"""
@inline function idxGToL(idx::Integer, layer::IsingLayer)
    return Int32(idx + 1 - startidx(layer))
end
export idxLToG, idxGToL



### STATE SET
function changeset(l::IsingLayer{SType}, set) where SType
    _eltype = eltype(graph(l))
    newset = convert.(_eltype, set)
    g = graph(l)
    # newlayer = IsingLayer(SType, l.graph, l.internal_idx, l.start, l.size[1], l.size[2], name = l.name, coords = l.coords, connections = l.connections, rangebegin = set[1], rangeend = set[2])
    newlayer = IsingLayer{SType, newset}(g, l.name, l.internal_idx, l.startidx, l.size, l.nstates,  l.coords, l.connections, l.timers, l.top)
    newlayer.timers = l.timers

    return newlayer
end

function changeset!(l, set)
    _layers = layers(graph(l))
    _layeridx = layeridx(l)
    _layers[_layeridx] = changeset(l, set)
    notify(graph(l))
    return _layers[_layeridx]
end
export changeset, changeset!

indexset(l::IsingLayer) = graphidxs(l)

function extremiseDiscrete!(l::IsingLayer{ST}) where ST
    if ST == Discrete
        a = stateset(l)[1]
        b = stateset(l)[end]
        map!(x -> x >= (a+b)/2f0 ? b : a, state(l), state(l))
    end
    return state(l)
end
function extremise!(l::IsingLayer{ST}) where ST
    a = stateset(l)[1]
    b = stateset(l)[end]
    map!(x -> x >= (a+b)/2f0 ? b : a, state(l), state(l))
end

mapToStateSet!(l::IsingLayer{ST}, dest, source) where {ST} = map!(x -> closestTo(l, x), dest, source)

function closestTo(l::IsingLayer{ST}, x) where {ST}
    if x < stateset(l)[1]
        return stateset(l)[1]
    elseif x > stateset(l)[end]
        return stateset(l)[end]
    end

    if ST == Discrete
        d1 = abs(x-stateset(l)[1])
        idx = 1
        for s in stateset(l)[2:end]
            d = abs(x-s)
            if d < d1
                d1 = d
                idx += 1
            else
                break
            end
        end
        return stateset(l)[idx]
    end

    return x
end

export changeset!, stateset

### TYPE STUFF
## DEFAULT NEW LAYER TYPE BASED ON GRAPH
default_ltype(g::IsingGraph{T}) where T = T == Int8 ? Discrete : Continuous 
@inline statetype(layer::IsingLayer{ST}) where {ST} = ST
@inline statetype(::Type{<:IsingLayer}) = layerparams(IsingLayer, Val(:StateType))
setstatetype(l::IsingLayer{ST}, stype) where {ST} = IsingLayer{stype}(l.graph, l.name, l.internal_idx, l.startidx, l.size, l.nstates, l.coords, l.connections, l.timers, l.top)

Base.eltype(l::IsingLayer) = eltype(graph(l))

# ORDER LAYER TYPES BASED ON STATETYPE
# TODO: HACKY
# # Make empty layers
# Base.isless(::Type{IsingLayer{A,B,C,D,T1}}, ::Type{IsingLayer{E,F,G,H,T2}}) where {A,B,C,D,E,F,G,H,T1,T2} = isless(A,D)
# Base.isless(::Type{IsingLayer{A,B}}, ::Type{IsingLayer{E,F,G,H,T}}) where {A,B,E,F,G,H,T} = isless(A,E)
# Base.isless(t1::Type{<:IsingLayer}, t2::Type{<:IsingLayer}) = isless(layerparams(t1, Val(:StateType)), layerparams(t2, Val(:StateType)))


export statetype, setstatetype

### GENERATING STATE
@inline Base.rand(layer::IsingLayer{StateType}) where {StateType} = sample_from_stateset(StateType, stateset(layer))
@inline Base.rand(layer::IsingLayer{StateType}, num::Integer) where {StateType} =  sample_from_stateset(StateType, stateset(layer), num)


@inline function initstate!(layer::IsingLayer)
    state(layer)[:] .= rand(layer, nStates(layer))
end
export initstate!