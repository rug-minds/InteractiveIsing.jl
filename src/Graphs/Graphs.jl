export AbstractIsingGraph, IsingGraph, CIsingGraph, coordToIdx, idxToCoord, index_set, setSpins!, addDefects!, remDefects!, addDefect!, remDefect!, 
    connIdx, connW, initSqAdj, HFunc, HWeightedFunc, HMagFunc, HWMagFunc, setGHFunc!, gwidth, glength,
    StatePartition, partition_index, partition_value, partition_dispatch,
    VectorSpinGraph, AbstractSpinGraph, AbstractVectorSpinGraph, vector_unit_norm

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

include("Utils.jl")
include("GraphDefects.jl")
include("GraphDefectsNew.jl")
include("IsingGraph.jl")
include("VectorSpinGraph.jl")


include("Layers/Layers.jl")
include("StatePartition.jl")
include("SingleLayerGraph.jl")
include("InitialState.jl")
include("Constructors.jl")
include("VectorSpinConstructors.jl")


include("User.jl")
include("Saving.jl")
