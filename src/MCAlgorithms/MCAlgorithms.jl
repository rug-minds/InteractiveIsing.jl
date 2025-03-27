# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: ProcessAlgorithm end
    abstract type Hamiltonian end

    """
    For a Hamiltonian, return all fieldnames that are a ParamVal
    """
    @generated function paramnames(h::Hamiltonian)
        _fieldnames = fieldnames(h)
        _fieldtypes = fieldtypes(h)
        paramval_params = tuple((fieldnames[i] for i in eachindex(fieldnames) if fieldtypes[i] <: ParamVal)...)
        return (:($paramval_params))
    end

    function update!(::Hamiltonian, args)
        return nothing
    end

    abstract type ConcreteHamiltonian end

    const defined_derived = Dict{Type, Type}()


    """
        Fallback required derived hamiltonian for an algorithm, 
        indicating that that the algorithm did not define a required derived hamiltonian.
    """
    function requires(::Type{<:MCAlgorithm})
        throw(ArgumentError("This algorithm did not define a required derived Hamiltonian."))
    end

    # function Processes.prepare(::Type{Algo}, g; kwargs...) where {Algo <: MCAlgorithm}
    #     args_algo = (;_prepare(Algo, g; kwargs...)...)

    #     return args_algo
    # end

    # function Processes.prepare(::Algo, @specialize(args)) where {Algo <: MCAlgorithm}
    #     return (;_prepare(Algo, args)...)
    # end

  
    

    include("Utils.jl")
    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")
# end