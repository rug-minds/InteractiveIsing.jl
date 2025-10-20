# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: ProcessAlgorithm end
    abstract type Hamiltonian end
    abstract type AbstractHamiltonianTerms{HS} <: Hamiltonian end
    getHS(::Type{<:AbstractHamiltonianTerms{HS}}) where {HS} = HS
    getHS(::AbstractHamiltonianTerms{HS}) where {HS} = HS
    getHS(h::Type{<:Hamiltonian}) = (h,)

    function _paramnames(h::Type{<:Hamiltonian}, all_names = Symbol[])
        for h in getHS(h)
            if h <: AbstractHamiltonianTerms
                _paramnames(h, all_names)
            else
                fnames = fieldnames(h)
                ftypes = fieldtypes(h)
                for (idx, name) in enumerate(fnames)
                    if ftypes[idx] <: ParamVal
                        push!(all_names, name)
                    end
                end
            end
        end
    end

    """
    For a Hamiltonian, return all fieldnames that are a ParamVal
    """
    @generated function paramnames(h::Union{Hamiltonian, AbstractHamiltonianTerms})
        h = getHS(h)
        all_names = Symbol[]
        _fieldnames = fieldnames.(h)
        _fieldtypes = fieldtypes.(h)
        for i in eachindex(_fieldnames)
            _paramnames(h[i], all_names)
        end
        paramval_params = (all_names...,)
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