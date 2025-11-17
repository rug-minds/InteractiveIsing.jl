# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: ProcessAlgorithm end
    
  
    

    include("Utils.jl")
    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")

# end