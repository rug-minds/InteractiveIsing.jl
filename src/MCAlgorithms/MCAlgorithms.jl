# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: ProcessAlgorithm end
    
  
    

    include("Utils.jl")
    include("Hamiltonians/Hamiltonians.jl")
    include("Algorithms/Algorithms.jl")
# end