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
struct IsingLayerData{StateType, StateSet, Dim, Size, PtrRange, Top} <: AbstractLayerData{Dim}
    # Reference to the graph holding it    
    # Can be nothing so that saving is easier
    name::String
    # TRAITS, Not used?
    traits::NamedTuple
    coords::Coords{Tuple{Int32,Int32,Int32}}
    weightgenerator::Base.RefValue{Union{WeightGenerator, Nothing}}

    # Keeps track of the connected layers
    connections::Dict{Pair{Int32,Int32}, Any} 
    # A layer can hold its own timers (WHY)
    top::Top   
end

"""
Simple Constructor
"""
function IsingLayerData(size::Tuple, statetype, stateset, wg::Union{WeightGenerator, Nothing}, topology, coords = Coords(nothing))
    IsingLayerData{statetype, stateset, length(size), size, 1:prod(size), typeof(topology)}(
        "Layer 1",
        (;StateType = statetype, StateSet = stateset, Dim = length(size), Size = size, PtrRange = 1:prod(size), Top = typeof(topology)),
        coords,
        Base.RefValue{Union{WeightGenerator, Nothing}}(wg),
        Dict{Pair{Int32,Int32}, Any}(),
        topology
    )
end

"""
Offset the stateindex range, and set the stateset to be compatible with the precision
"""
function fix_layerdata(ild::IsingLayerData{ST,SS,Dim,Size,PtrRange,Top}, precision, offset) where {ST,SS,Dim,Size,PtrRange,Top}
    newset = convert.(precision, stateset(ild))
    newrange = (1+offset):(offset + prod(size(ild)))
    return IsingLayerData{ST, newset, Dim, Size, newrange, Top}(
        name(ild),
        traits(ild),
        ild.coords,
        ild.weightgenerator,
        connections(ild),
        top(ild)
    )
end


function IsingLayerData(
            lsize,
            idx, #TODO REMOVE
            start::Int;
            stype = Discrete(), 
            precision = Float32,
            set = convert.(precision, (-1.,1.)), 
            name = "Layer $idx", 
            # traits = (;StateType = StateType, StateSet = set, Indices = (start:(start+length*width-1)), Hamiltonians = (Ising,)),
            traits = (;),
            coords = Coords(nothing), 
            adjtype = SparseMatrixCSC{precision,Int32},
            wg::Union{WeightGenerator, Nothing} = nothing,
            connections = Dict{Pair{Int32,Int32}, Any}(), 
            periodic::Union{Nothing,Bool,Tuple} = true,
            kwargs...
        )
        if !isnothing(set) #TODO FIX THIS
            set = precision.(set)
        else
            set = convert.(precision, (-1.,1.)) 
        end
        
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

        layer = IsingLayerData{stype, set, dims, lsize, graphidxs, typeof(top)}(
            # Name
            name,
            # Traits
            traits,
            #Coordinates
            coords,
            # WeightGenerator
            Base.RefValue{Union{WeightGenerator, Nothing}}(wg),
            # Connections
            connections,

            # Topology
            top
        )
        return layer
end


@inline name(layer::IsingLayerData) = layer.name
@inline traits(layer::IsingLayerData) = layer.traits
@inline connections(layer::IsingLayerData) = layer.connections
@inline top(layer::IsingLayerData) = layer.top
@inline topology(layer::IsingLayerData) = top(layer)
@inline coords(layer::IsingLayerData) = layer.coords.cs
@inline setcoords!(layer::IsingLayerData; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::IsingLayerData, val) = (layer.coords.cs = Int32.((val,val,val)))
@inline Base.size(layer::IsingLayerData) = layerparams(layer, :Size)
@inline Base.length(layer::IsingLayerData) = prod(size(layer))

@inline function layerparams(lt::Type{<:IsingLayerData}, ::Val{S}) where S
    ps = parameters(lt)
    idx = nothing
    if S == :StateType
        idx = 1 
    elseif S == :StateSet
        idx = 2
    elseif S == :Dim
        idx = 3
    elseif S == :Size
        idx = 4
    elseif S == :PtrRange
        idx = 5    
    elseif S == :Tpt
        idx = 6
    end
    return ps[idx]
end

@inline layerparams(l::IsingLayerData, s::Symbol) = layerparams(typeof(l), Val(s))

@inline stateset(lt::Type{<:IsingLayerData{A,B}}) where {A,B} = B   
@inline stateset(l::IsingLayerData) = stateset(typeof(l))
@inline stateset_end(lt::Type{<:IsingLayerData{A,B}}) where {A,B} = last(B)
@inline stateset_end(l::IsingLayerData) = stateset_end(typeof(l))

@inline statetype(lt::Type{<:IsingLayerData{A}}) where {A} = A
@inline statetype(l::IsingLayerData) = statetype(typeof(l))

@inline range(lt::Type{<:IsingLayerData{A,B,C,D,E}}) where {A,B,C,D,E} = E
@inline range(l::IsingLayerData) = range(typeof(l))
@inline range_end(lt::Type{<:IsingLayerData{A,B,C,D,E}}) where {A,B,C,D,E} = last(E)
@inline range_end(l::IsingLayerData) = range_end(typeof(l))
@inline Base.parentindices(l::IsingLayerData) = (range(l),)

get_weightgenerator(layer::IsingLayerData) = layer.weightgenerator[]


"""
View like struct for layer data and isinggraph
"""
struct IsingLayer{StateType, Dim, Data, Graph} <: AbstractIsingLayer{StateType, Dim}
    data::Data
    graph::Graph
    idx::Int
end

@inline function IsingLayer(data::D, graph::G, idx::Integer) where {D<:IsingLayerData, G}
    IsingLayer{statetype(D), layerparams(D, Val(:Dim)), D, G}(data, graph, Int(idx))
end

@inline data(layer::IsingLayer) = layer.data
@inline graph(layer::IsingLayer) = layer.graph
@inline layeridx(layer::IsingLayer) = layer.idx
@inline internal_idx(layer::IsingLayer) = layeridx(layer)

@inline layerdatatype(::Type{<:IsingLayer{A,B,Data,Graph}}) where {A,B,Data,Graph} = Data
@inline layerdatatype(layer::IsingLayer) = layerdatatype(typeof(layer))

@inline function layerparams(lt::Type{<:IsingLayer}, ::Val{S}) where S
    layerparams(layerdatatype(lt), Val(S))
end
@inline layerparams(layer::IsingLayer, s::Symbol) = layerparams(data(layer), s)

@inline stateset(lt::Type{<:IsingLayer}) = stateset(layerdatatype(lt))
@inline stateset(layer::IsingLayer) = stateset(data(layer))
@inline stateset_end(lt::Type{<:IsingLayer}) = stateset_end(layerdatatype(lt))
@inline stateset_end(layer::IsingLayer) = stateset_end(data(layer))

@inline statetype(lt::Type{<:IsingLayer}) = statetype(layerdatatype(lt))
@inline statetype(layer::IsingLayer) = statetype(data(layer))

@inline range(lt::Type{<:IsingLayer}) = range(layerdatatype(lt))
@inline range(layer::IsingLayer) = range(data(layer))
@inline range_end(lt::Type{<:IsingLayer}) = range_end(layerdatatype(lt))
@inline range_end(layer::IsingLayer) = range_end(data(layer))
@inline Base.parentindices(layer::IsingLayer) = parentindices(data(layer))

@inline name(layer::IsingLayer) = name(data(layer))
@inline traits(layer::IsingLayer) = traits(data(layer))
@inline connections(layer::IsingLayer) = connections(data(layer))
@inline top(layer::IsingLayer) = top(data(layer))
@inline topology(layer::IsingLayer) = topology(data(layer))
@inline coords(layer::IsingLayer) = coords(data(layer))
@inline setcoords!(layer::IsingLayer; x = 0, y = 0, z = 0) = setcoords!(data(layer); x, y, z)
@inline setcoords!(layer::IsingLayer, val) = setcoords!(data(layer), val)
destructor(layer::IsingLayer) = destructor(data(layer))

# Create view
IsingLayer(g::AbstractIsingGraph, idx::Integer) = IsingLayer(getfield(g, :layers)[idx], g, idx)

Base.LinearIndices(l::IsingLayer) = LinearIndices(size(l))
Base.CartesianIndices(l::IsingLayer) = CartesianIndices(size(l))


get_weightgenerator(layer::IsingLayer) = get_weightgenerator(data(layer))
# struct LayerProperties <: AbstractLayerProperties
#     size::Tuple
#     kwargs::NamedTuple
# end

# datalen(lp::LayerProperties) = prod(lp.size)

# function Layer(dims...; kwargs...)
#     LayerProperties(tuple(dims...), (;kwargs...))
# end

# export Layer

# IsingLayer(g, idx, startidx, lp::LayerProperties) = IsingLayerData(lp.size, idx, startidx; lp.kwargs...)

export IsingLayer
export topology



## ACCESSORS
@inline function state(l::IsingLayer)
    et = eltype(l)
    gstate = getstate(graph(l))::Vector{et}
    v = @view gstate[graphidxs(l)]
    v = unsafe_wrap(Array, pointer(v), size(l))
end 

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
# @inline graph(layer::IsingLayer, g::IsingGraph) = layer.graph = g
# changegraph(l::IsingLayer, g) = IsingLayer{layerparams(l, :StateType), layerparams(l, :DIMS), layerparams(l, :T), layerparams(l, :Top)}(g, l.name, l.internal_idx, l.startidx, l.size, l.nstates, l.traits, l.coords, l.connections, l.timers, l.top)
addons(layer::IsingLayer) = graph(layer).addons
export addons

# Get current layeridx through graph
# @inline layeridx(layer::IsingLayer) = externalidx(layers(graph(layer)), layer.internal_idx)
### COORDINATES OF LAYERS
@inline coords(layer::AbstractIsingLayer) = coords(data(layer))
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer; x = 0, y = 0, z = 0) = setcoords!(data(layer); x, y, z)
@inline setcoords!(layer::AbstractIsingLayer, val) = setcoords!(data(layer), val)

export setcoords!

"""
Get adjacency of layer in layer coordinates
"""
@inline reladj(layer::AbstractIsingLayer) = adjGToL(adj(graph(layer)), layer)

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

# @inline coordToIdx(i,j,layer::AbstractIsingLayer) = coordToIdx(latmod(Int32(i), size(layer,1)), latmod(Int32(j), size(layer,2)), size(layer,1))
function coordToIdx(idxs::NTuple{N,T}, layer::AbstractIsingLayer) where {N,T}
    sl = size(layer)
    coordToIdx(ntuple(i -> let idx = idxs[i]; T(latmod(i, size(layer, idx))) end, length(idxs) ), sl)
end
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
    lidx = internal_idx(layer)
    preceding_defects = sum(ds.layerdefects[1:lidx-1])
    these_defects = ds.layerdefects[lidx]
    alivelist_range = (startidx(layer)-preceding_defects):(endidx(layer)-preceding_defects-these_defects)
    aliveList(ds)[alivelist_range]
end
export aliveidxs


### TIMERS
    # pausetimers(layer) = close.(timers(layer))
    # starttimers(layer) = start.(timers(layer))
    # removetimers(layer) = begin close.(timers(layer)); layer.timers = Vector{PTimer}(); end


### GET INDEXES
iterator(layer::AbstractIsingLayer) = startidx(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# AbstractLayerTopology
@inline periodic(layer::AbstractIsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer::AbstractIsingLayer, periodic) = top!(layer, AbstractLayerTopology(top(layer); periodic))

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
@inline function idxLToG(idx::Integer, layer::L) where L<:IsingLayer
    return Int32(startidx(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(i::Integer, j::Integer, layer::L) where L<:IsingLayer
    return Int32(startidx(layer) + coordToIdx(i,j, glength(layer)) - 1)
end

@inline idxLToG(tup::Tuple, layer::L) where L<:IsingLayer = idxLToG(tup[1], tup[2], layer)

"""
Go from graph idx to idx of layer
"""
@inline function idxGToL(idx::Integer, layer::L) where L<:IsingLayer
    return Int32(idx + 1 - startidx(layer))
end
export idxLToG, idxGToL

indexset(l::IsingLayer) = graphidxs(l)
indexset(lt::Type{<:IsingLayer}) = layerparams(lt, Val(:PtrRange))

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

Base.eltype(l::IsingLayer) = eltype(graph(l))

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

export statetype

@inline function initstate!(layer::IsingLayer)
    state(layer)[:] .= rand(layer, nStates(layer))
end
export initstate!


function Base.show(io::IO, layer::IsingLayer)
    print(io, statetype(layer), " IsingLayer ", layeridx(layer), " with size ", size(layer), " and stateset ", stateset(layer))
    layer_coords = coords(layer)
    if !isnothing(layer_coords)
        print(io, " at coordinates ", layer_coords)
    end
    print(io, "\n\n with connections:")
    for key in keys(connections(layer))
        print(io, "\n\tConnected to layer ", key[2], " using ")
        print(io, "\n\t", connections(layer)[key])
    end

    ndef = try
        ndefect(layer)
    catch
        nothing
    end
    if isnothing(ndef)
        print(io, "\n and 0 defects")
    else
        print(io, "\n and ", ndef, " defects")
    end
end

function Base.show(io::IO, layer::IsingLayerData)
    print(
        io,
        statetype(layer),
        " IsingLayerData size=",
        layerparams(layer, :Size),
        " stateset=",
        stateset(layer)
    )
    layer_coords = coords(layer)
    if !isnothing(layer_coords)
        print(io, " coords=", layer_coords)
    end
    print(io, " connections=", length(connections(layer)))
end

# SHOW THE TYPE
Base.show(io::IO, ::Type{IsingLayer{A,B,Data,Graph}}) where {A,B,Data,Graph} = print(io, "$A IsingLayer")
