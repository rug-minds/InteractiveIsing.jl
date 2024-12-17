# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type Algorithm <: Function end
    abstract type MCAlgorithm <: Algorithm end
    abstract type Hamiltonian end
    # struct Hamiltonian 
    #     description::Any
    # end
    abstract type ConcreteHamiltonian end

    const defined_derived = Dict{Type, Type}()


    """
        Fallback required derived hamiltonian for an algorithm, 
        indicating that that the algorithm did not define a required derived hamiltonian.
    """
    function requires(::Type{<:MCAlgorithm})
        throw(ArgumentError("This algorithm did not define a required derived Hamiltonian."))
    end

    # function prepare(::Type{Algo}, g; kwargs...) where {Algo <: MCAlgorithm}
    #     args_algo = (;_prepare(Algo, g; kwargs...)...)

    #     return args_algo
    # end

    function prepare(::Type{Algo}, @specialize(args)) where {Algo <: MCAlgorithm}
        return (;_prepare(Algo, args)...)
    end

  
    

    include("Utils.jl")
    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")
# end