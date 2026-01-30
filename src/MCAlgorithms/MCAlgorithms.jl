using LoopVectorization, SparseArrays, MacroTools
export build_H

abstract type MCAlgorithm <: ProcessAlgorithm end




include("Utils.jl")
include("Algorithms/Algorithms.jl")
include("Hamiltonians/Hamiltonians.jl")

