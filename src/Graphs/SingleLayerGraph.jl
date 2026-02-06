# #=
# This File contains redirections of function from a graph to its first layer
# =#
export SingleLayerGraph, layer, state, size, topology, settopology!, genAdj!, setdist!

const SingleLayerGraph{T<:AbstractFloat, M<:AbstractMatrix{T}} = IsingGraph{T,M,<:Tuple{<:AbstractIsingLayer}}

layer(g::SingleLayerGraph) = g[1]
Base.CartesianIndices(g::SingleLayerGraph) = CartesianIndices(layer(g))
Base.LinearIndices(g::SingleLayerGraph) = LinearIndices(layer(g))
Base.size(g::SingleLayerGraph) = size(layer(g))
Base.size(g::SingleLayerGraph, dim::Int) = size(layer(g), dim)
eachaxis(g::SingleLayerGraph, dim::Int) = eachaxis(layer(g), dim)
Base.axes(g::SingleLayerGraph) = ntuple(i -> Base.OneTo(size(g, i)), ndims(g))
Base.axes(g::SingleLayerGraph, dim::Int) = Base.OneTo(size(g, dim))

state(g::SingleLayerGraph) = reshape(g.state, size(layer(g)))

genAdj!(g::SingleLayerGraph, wg; kwargs...) = genAdj!(layer(g), wg; kwargs...)
topology(g::SingleLayerGraph) = topology(layer(g))
settopology!(g::SingleLayerGraph, top::AbstractLayerTopology) = layer(g).top = top
setdist!(g::SingleLayerGraph, ds::NTuple{N,Float64}) where N = setdist!(topology(layer(g)), ds)
setdist!(g::SingleLayerGraph, ds...) = setdist!(topology(layer(g)), ds)





