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
export IsingLayer

function regenerateViews(layer::IsingLayer)
    layer.state = reshapeView(state(graph(layer)), start(layer), glength(layer), gwidth(layer))
    layer.adj = reshapeView(adj(graph(layer)), start(layer), glength(layer), gwidth(layer))
end

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::IsingLayer)
    showstr = "IsingLayer $(layer.layeridx) with size $(size(layer))"
    if coords(layer) != nothing
        showstr *= " \t at coordinates $(coords(layer))"
    end
    print(io, showstr)
end

Base.show(io::IO, layertype::Type{<:IsingLayer}) = print(io, "IsingLayer")


IsingLayer(g, layer::IsingLayer) = IsingLayer(g, layeridx(layer), start(layer), glength(layer), gwidth(layer), olddefects = ndefects(layer))

@setterGetter IsingLayer coords size
@inline coords(layer::IsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::IsingLayer; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::IsingLayer, val) = (layer.coords.cs = Int32.((val,val,val)))
export setcoords!

@inline reladj(layer::IsingLayer) = adjGToL(layer.adj, layer)
@forward IsingLayer LayerData d
@forward IsingLayer LayerDefects defects

# Setters and getters
# @forward IsingLayer IsingGraph g
@inline size(layer::IsingLayer)::Tuple{Int32,Int32} = layer.size
@inline glength(layer::IsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[1]::Int32
@inline gwidth(layer::IsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[2]::Int32
@inline endidx(layer::IsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1

@inline graphidxs(layer::IsingLayer) = UnitRange{Int32}(start(layer):endidx(layer))
export graphidxs

# Inherited from Graph
@inline htype(layer::IsingLayer) = layer.graph.htype
@inline nStates(layer::IsingLayer) = length(state(layer))
@inline sim(layer::IsingLayer) = sim(graph(layer))

#forward alivelist
aliveList(layer::IsingLayer) = aliveList(defects(layer))
#forward defectList
defectList(layer::IsingLayer) = defectList(defects(layer))

@inline hasDefects(layer::IsingLayer) = ndefects(defects(layer)) > 0
@inline setdefect(layer::IsingLayer, val, idx) = defects(layer)[idx] = val

@inline startidx(layer::IsingLayer) = start(layer)
iterator(layer::IsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# LayerTopology
@inline periodic(layer::IsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer, periodic) = top!(layer, LayerTopology(top(layer); periodic))
export setPeriodic!

editHType!(layer::IsingLayer, pairs...) = editHType!(layer.graph, pairs...)

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
setSType!(layer::IsingLayer, varargs...; refresh = true) = setSType!(graph(layer), varargs...; refresh = refresh)
