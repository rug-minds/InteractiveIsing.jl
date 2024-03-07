module MCAlgorithms  
    abstract type MCAlgorithm end
    abstract type Hamiltonian end

    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")
end