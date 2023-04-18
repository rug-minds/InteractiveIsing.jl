"""
ndefects not tracking correctly FIX
"""
mutable struct IsingLayer{T} <: AbstractIsingGraph{T}
    graph::IsingGraph{T}
    layeridx::Int32
    state::Base.ReshapedArray
    adj::Base.ReshapedArray
    start::Int32
    size::Tuple{Int32,Int32}
    d::LayerData
    defects::LayerDefects

    EmptyLayer(type) = new{type}() 

    IsingLayer(g::IsingGraph{T}, idx, start, length, width; olddefects = 0) where T =
    (
        layer = new{T}(
            # Graph
            g,
            # Layer idx
            Int32(idx),
            # View of state
            reshapeView(state(g), start, length, width),
            # View of adj
            reshapeView(adj(g), start, length, width),
            # Start idx
            Int32(start),
            # Size
            tuple(Int32(length), Int32(width)),
            # Layer data
            LayerData(d(g), start, length, width)
        );
        layer.defects = LayerDefects(layer, olddefects);
     

        return layer
    )
end

function regenerateViews(layer::IsingLayer)
    layer.state = reshapeView(state(graph(layer)), start(layer), glength(layer), gwidth(layer))
    layer.adj = reshapeView(adj(graph(layer)), start(layer), glength(layer), gwidth(layer))
end

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::IsingLayer)
    println(io, "IsingLayer $(layer.layeridx) with size $(size(layer))")
end

IsingLayer(g, layer::IsingLayer) = IsingLayer(g, layeridx(layer), start(layer), glength(layer), gwidth(layer), olddefects = ndefects(layer))

@setterGetter IsingLayer
@inline reladj(layer::IsingLayer) = adjGToL(layer.adj, layer)
@forward IsingLayer LayerData d
@forward IsingLayer LayerDefects defects
# Setters and getters
# @forward IsingLayer IsingGraph g
@inline glength(layer) = size(layer)[1]
@inline gwidth(layer) = size(layer)[2]
@inline endidx(layer::IsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1



@inline htype(layer::IsingLayer) = layer.graph.htype
@inline nStates(layer::IsingLayer) = length(state(layer))

#forward alivelist
aliveList(layer::IsingLayer) = aliveList(defects(layer))
#forward defectList
defectList(layer::IsingLayer) = defectList(defects(layer))

@inline hasDefects(layer::IsingLayer) = ndefects(defects(layer)) > 0
@inline setdefect(layer::IsingLayer, val, idx) = defects(layer)[idx] = val

startidx(layer::IsingLayer) = start(layer)
endidx(layer::IsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1
iterator(layer::IsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

Hamiltonians.editHType!(layer::IsingLayer, pairs...) = editHType!(layer.graph, pairs...)



"""
Go from a local idx of layer to idx of the underlying graph
"""
@inline function idxLToG(layer, idx)::Int32
    return Int32(start(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(layer, i, j)::Int32
    return Int32(start(layer) + coordToIdx(i,j, llength(layer)))
end

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
        idxs_weights = getUniqueConnIdxs(wf, idx, glength(layer), gwidth(layer))
        
        for idx_weight in idxs_weights
            addWeight!(layer, idx, idx_weight[1], idx_weight[2])
        end
    end
end


