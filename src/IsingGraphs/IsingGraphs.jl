export AbstractIsingGraph, IsingGraph, CIsingGraph, reInitGraph!, coordToIdx, idxToCoord, ising_it, setSpins!, setSpin!, addDefects!, remDefects!, addDefect!, remDefect!, 
    connIdx, connW, initSqAdj, HFunc, HWeightedFunc, HMagFunc, HWMagFunc, setGHFunc!

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

include("Data.jl")
include("Graph.jl")
include("Layer.jl")