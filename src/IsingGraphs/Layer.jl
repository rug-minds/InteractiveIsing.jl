struct IsingLayer{T} <: AbstractIsingGraph{T}
    g::IsingGraph{T}
    state::Base.ReshapedArray
    adj::Base.ReshapedArray
    start::Int32
    size::Tuple{Int32,Int32}
    d::LayerData
end

@setterGetter IsingLayer
@forward IsingLayer LayerData d
# Setters and getters
# @forward IsingLayer IsingGraph g
@inline glength(layer) = size(layer)[1]
@inline gwidth(layer) = size(layer)[2]
@inline aliveList(layer::IsingLayer) = layer.g.d.aliveList
@inline aliveList(layer::IsingLayer, newlist) = layer.g.d.aliveList = newlist
@inline defectList(layer::IsingLayer) = layer.g.d.defectList
@inline defectList(layer::IsingLayer, newlist) = layer.g.d.defectList = newlist
@inline defectBools(layer::IsingLayer) = layer.g.d.defectBools
@inline defects(layer::IsingLayer) = layer.g.d.defects
@inline htype(layer::IsingLayer) = layer.g.htype

Hamiltonians.editHType!(layer::IsingLayer, pairs...) = editHType!(layer.g, pairs...)

IsingLayer(g::IsingGraph, start, length, width) =
    IsingLayer(
        g,
        reshapeView(state(g), start, length, width),
        reshapeView(adj(g), start, length, width),
        Int32(start),
        tuple(Int32(length), Int32(width)),
        LayerData(d(g), start, width, length)
    )

@inline function viewToGIdx(layer, idx)
    return start(layer) + idx - 1
end

@inline function viewToGIdx(layer, i, j)
    return start(layer) + coordToIdx(i,j, llength(layer))
end