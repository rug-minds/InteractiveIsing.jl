export AbstractIsingGraph, IsingGraph, CIsingGraph, coordToIdx, idxToCoord, ising_it, setSpins!, addDefects!, remDefects!, addDefect!, remDefect!, 
    connIdx, connW, initSqAdj, HFunc, HWeightedFunc, HMagFunc, HWMagFunc, setGHFunc!, gwidth, glength

# Aliases
const Edge = Pair{Int32,Int32}
const Vert = Int32
const Weight = Float32
const Conn = Tuple{Vert, Weight}

"""
Makes a reshaped view of a vector so that a part of memory can be interpreted as holding a matrix of dimensions length*width
"""
@inline function reshapeView(vec, start, length, width)
    return reshape((@view vec[start:(start + width*length - 1)]), length, width)
end
export reshapeView

export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end

@ForwardDeclare IsingGraph "IsingGraphs"
@ForwardDeclare IsingLayer "IsingGraphs"

include("Data.jl")
include("GraphDefects.jl")
include("Graph.jl")

include("LayerDefects.jl")
include("LayerTopology.jl")

include("Layer.jl")
include("SetEls.jl")

include("LayerConnFunc.jl")

include("User.jl")