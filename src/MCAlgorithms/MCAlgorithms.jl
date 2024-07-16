# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: Function end
    abstract type Hamiltonian end
    # struct Hamiltonian 
    #     description::Any
    # end
    abstract type DerivedHamiltonian end

    const defined_derived = Dict{Type, Type}()

    include("ParamVal.jl")


    """
        Fallback required derived hamiltonian for an algorithm, 
        indicating that that the algorithm did not define a required derived hamiltonian.
    """
    function requires(::Type{<:MCAlgorithm})
        throw(ArgumentError("This algorithm did not define a required derived Hamiltonian."))
    end

    function prepare(::Type{Algo}, g; kwargs...) where {Algo <: MCAlgorithm}
        args_algo = (;_prepare(Algo, g; kwargs...)...)

        return args_algo
    end


  
    

    include("Utils.jl")
    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")
# end