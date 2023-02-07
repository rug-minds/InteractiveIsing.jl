struct IsingLayer{T} <: AbstractIsingGraph{T}
    graph::IsingGraph{T}
    state::Base.ReshapedArray
    adj::Base.ReshapedArray
    start::Int32
    size::Tuple{Int32,Int32}
    d::LayerData
end

@setterGetter IsingLayer
@inline reladj(layer::IsingLayer) = adjAbsToView(layer.adj, layer)
@forward IsingLayer LayerData d
# Setters and getters
# @forward IsingLayer IsingGraph g
@inline glength(layer) = size(layer)[1]
@inline gwidth(layer) = size(layer)[2]
@inline aliveList(layer::IsingLayer) = layer.graph.d.aliveList
@inline aliveList(layer::IsingLayer, newlist) = layer.graph.d.aliveList = newlist
@inline defectList(layer::IsingLayer) = layer.graph.d.defectList
@inline defectList(layer::IsingLayer, newlist) = layer.graph.d.defectList = newlist
@inline defectBools(layer::IsingLayer) = layer.graph.d.defectBools
@inline defects(layer::IsingLayer) = layer.graph.d.defects
@inline htype(layer::IsingLayer) = layer.graph.htype
@inline nStates(layer::IsingLayer) = length(state(layer))

@inline function upNDefects(layer::IsingLayer)::Nothing
    ndefects(layer, sum(defectBools(layer)[start(layer):(start(layer)+nStates(layer)-1)] ) )
    return
end

startidx(layer::IsingLayer) = start(layer)
endidx(layer::IsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1
iterator(layer::IsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

Hamiltonians.editHType!(layer::IsingLayer, pairs...) = editHType!(layer.graph, pairs...)

IsingLayer(g::IsingGraph, start, length, width) =
    IsingLayer(
        g,
        reshapeView(state(g), start, length, width),
        reshapeView(adj(g), start, length, width),
        Int32(start),
        tuple(Int32(length), Int32(width)),
        LayerData(d(g), start, length, width)
    )

@inline function viewToGIdx(layer, idx)::Int32
    return start(layer) + idx - 1
end

@inline function viewToGIdx(layer, i, j)::Int32
    return start(layer) + coordToIdx(i,j, llength(layer))
end

@inline function gIdxToView(layer, idx)
    return idx + 1 - start(layer)
end

function viewAdjToAbs!(adj, layer)
    for entry in adj
        for idx in 1:length(entry)
            entry[idx] = tuple(viewToGIdx(layer, connIdx(entry[idx])), connW(entry[idx]))
        end
    end
end
# debug
export viewAdjToAbs!

function adjAbsToView(adj::AbstractArray{T}, layer) where T
    newadj = Vector{T}(undef, length(adj))
    for adjidx in 1:length(adj)
        entry = adj[adjidx]
        newentry = typeof(entry)(undef, length(entry))
        for entryidx in 1:length(entry)
            conn = entry[entryidx]
            newentry[entryidx] = tuple(gIdxToView(layer, connIdx(conn)), connW(conn))
        end
        newadj[adjidx] = newentry
    end
    return newadj
end
export adjAbsToView

function fillAdjList!(layer::IsingLayer, length, width, weightFunc=defaultIsingWF)
    periodic = weightFunc.periodic
    NN = weightFunc.NN
    inv = weightFunc.invoke
    adjlist = adj(layer)
    
    for idx in 1:length*width
        adjlist[idx] = adjEntry(adjlist, length, width, idx, periodic, NN, inv)
    end
    
    viewAdjToAbs!(adjlist, layer)
  
end
