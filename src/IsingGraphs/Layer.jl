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
mutable struct IsingLayer{T, IsingGraphType <: AbstractIsingGraph, StateReshape <: AbstractMatrix{T}, AdjReshape <: AbstractArray} <: AbstractIsingLayer{T}
    const graph::IsingGraphType
    layeridx::Int32
    state::StateReshape
    adj::AdjReshape
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}
    const d::LayerData
    defects::LayerDefects
    top::LayerTopology


    function IsingLayer(LayerType ,g::GraphType, idx, start, length, width; olddefects = 0, periodic::Union{Nothing,Bool} = true) where GraphType <: IsingGraph
        stateview = reshapeView(state(g), start, length, width)
        adjview = reshapeView(adj(g), start, length, width)
        statetype = typeof(stateview)
        adjtype = typeof(adjview)
        lsize = tuple(Int32(length), Int32(width))
        layer = new{LayerType, GraphType, statetype, adjtype}(
            # Graph
            g,
            # Layer idx
            Int32(idx),
            # View of state
            stateview,
            # View of adj
            adjview,
            # Start idx
            Int32(start),
            # Size
            lsize,
            # Number of states
            Int32(lsize[1]*lsize[2]),
            #Coordinates
            Coords(nothing),
            # Layer data
            LayerData(d(g), start, length, width)
        )
        layer.defects = LayerDefects(layer, olddefects)
        layer.top = LayerTopology(layer, [1,0], [0,1]; periodic)

        return layer
    end
end

mutable struct IsingLayerCopy{T, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    const graph::IsingGraphType
    layeridx::Int32
    state::Matrix{T}
    adj::Matrix{Vector{Tuple{Int32, Float32}}}
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}
    const d::LayerData
    defects::LayerDefects
    top::LayerTopology

    function IsingLayerCopy(layer::IsingLayer{A,B,C,D}) where {A,B,C,D}
        
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
            layer.d,
            layer.defects,
            layer.top
        )
    end
end


export IsingLayer

function regenerateViews(layer::IsingLayer)
    layer.state = reshapeView(state(graph(layer)), start(layer), glength(layer), gwidth(layer))
    layer.adj = reshapeView(adj(graph(layer)), start(layer), glength(layer), gwidth(layer))
end

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::AbstractIsingLayer)
    showstr = "IsingLayer $(layer.layeridx) with size $(size(layer))"
    if coords(layer) != nothing
        showstr *= " \t at coordinates $(coords(layer))"
    end
    print(io, showstr)
end

Base.show(io::IO, layertype::Type{<:AbstractIsingLayer}) = print(io, "IsingLayer")


IsingLayer(g, layer::AbstractIsingLayer) = IsingLayer(g, layeridx(layer), start(layer), glength(layer), gwidth(layer), olddefects = ndefects(layer))

@setterGetter IsingLayer coords size layeridx
@setterGetter IsingLayerCopy coords size


# Get current layeridx through graph
@inline layeridx(layer::IsingLayer) = layeridxs(graph(layer))[layer.layeridx]

@inline coords(layer::AbstractIsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::AbstractIsingLayer, val) = (layer.coords.cs = Int32.((val,val,val)))
export setcoords!

@inline reladj(layer::AbstractIsingLayer) = adjGToL(layer.adj, layer)
@forward IsingLayer LayerData d
@forward IsingLayer LayerDefects defects

@forward IsingLayerCopy LayerData d
@forward IsingLayerCopy LayerDefects defects

# Setters and getters
# @forward IsingLayer IsingGraph g
@inline size(layer::AbstractIsingLayer)::Tuple{Int32,Int32} = layer.size
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

@inline coordToIdx(i,j,layer::AbstractIsingLayer) = coordToIdx(latmod(i, glength(layer)), latmod(j, gwidth(layer)), glength(layer))

@inline dist(idx1::Integer, idx2::Integer, layer::AbstractIsingLayer) = dist(idxToCoord(idx1, glength(layer))..., idxToCoord(idx2, glength(layer))..., top(layer))



@inline getindex(layer::AbstractIsingLayer, idx) = state(layer)[idx]
@inline setindex!(layer::AbstractIsingLayer, val, idx) = state(layer)[idx] = val

@inline endidx(layer::AbstractIsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1

@inline graphidxs(layer::AbstractIsingLayer) = UnitRange{Int32}(start(layer):endidx(layer))
export graphidxs

# Inherited from Graph
@inline htype(layer::AbstractIsingLayer) = layer.graph.htype
@inline nStates(layer::AbstractIsingLayer) = length(state(layer))
@inline sim(layer::AbstractIsingLayer) = sim(graph(layer))

#forward alivelist
aliveList(layer::AbstractIsingLayer) = aliveList(defects(layer))
#forward defectList
defectList(layer::AbstractIsingLayer) = defectList(defects(layer))

@inline hasDefects(layer::AbstractIsingLayer) = ndefects(defects(layer)) > 0
@inline setdefect(layer::AbstractIsingLayer, val, idx) = defects(layer)[idx] = val

@inline startidx(layer::AbstractIsingLayer) = start(layer)
iterator(layer::AbstractIsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# LayerTopology
@inline periodic(layer::AbstractIsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer::AbstractIsingLayer, periodic) = top!(layer, LayerTopology(top(layer); periodic))
export setPeriodic!

editHType!(layer::AbstractIsingLayer, pairs...) = editHType!(layer.graph, pairs...)

# Forward Graph Data
# @inline mlist(layer::IsingLayer) = @view mlist(graph(layer))[start(layer):endidx(layer)]

"""
Go from a local idx of layer to idx of the underlying graph
"""
@inline function idxLToG(layer, idx::Integer)::Int32
    return Int32(start(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(layer, i, j)::Int32
    return Int32(start(layer) + coordToIdx(i,j, glength(layer)))
end
idxLToG(layer, tup::Tuple) = idxLToG(layer, tup[1], tup[2])

"""
Go from graph idx to idx of layer
"""
@inline function idxGToL(layer, idx)
    return Int32(idx + 1 - start(layer))
end

"""
Convert adjacency matrix of a layer to adjacency matrix with idxs of underlying graph
"""
function adjLToG(adj, layer)
    for entry in adj
        for idx in 1:length(entry)
            entry[idx] = tuple(idxLToG(layer, connIdx(entry[idx])), connW(entry[idx]))
        end
    end
end
# debug
export adjLToG

# What is this used for?
function adjGToL(adj::AbstractArray{T}, layer) where T
    newadj = Vector{T}(undef, length(adj))
    for adjidx in 1:length(adj)
        entry = adj[adjidx]
        newentry = typeof(entry)(undef, length(entry))
        for entryidx in 1:length(entry)
            conn = entry[entryidx]
            newentry[entryidx] = tuple(idxGToL(layer, connIdx(conn)), connW(conn))
        end
        newadj[adjidx] = newentry
    end
    return newadj
end
export adjGToL

function setAdj!(layer::IsingLayer, wf = defaultIsingWF)
    
    Threads.@threads for idx in eachindex(adj(layer))
    # for idx in eachindex(adj(layer))
        idxs_weights = getUniqueConnIdxs(wf, idx, glength(layer), gwidth(layer), wt(wf))
        
        for idx_weight in idxs_weights
            addWeight!(layer, idx, idx_weight[1], idx_weight[2])
        end
    end
end

# Set the SType
setSType!(layer::AbstractIsingLayer, varargs...; refresh = true) = setSType!(graph(layer), varargs...; refresh = refresh)
