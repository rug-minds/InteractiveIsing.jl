using LoopVectorization, SparseArrays, MacroTools
export build_H

abstract type IsingMCAlgorithm <: ProcessAlgorithm end




include("Utils.jl")
include("Algorithms/Algorithms.jl")
include("Hamiltonians/Hamiltonians.jl")

