using MacroTools
# RuntimeGeneratedFunctions.init(@__MODULE__)
"""
An hamiltonia is a struct that holds parameters
"""
abstract type Hamiltonian end

Base.merge(nt::NamedTuple, h::Hamiltonian) = (;nt..., pairs(h)...)
"""
Pairs gives an iterator over the ParamVal fields of the Hamiltonian
"""
@generated function Base.pairs(h::Hamiltonian)
    fnames = fieldnames(h)
    ftypes = fieldtypes(h)
    names_types = zip(fnames, ftypes)
    pvals_names = (x[1] for x in names_types if x[2] <: ParamVal) 
    tuple_exp = Expr(:tuple, Expr(:parameters, (Expr(:kw, name, :(getproperty(h, $(QuoteNode(name))))) for name in pvals_names)...))
    return :(pairs($(tuple_exp)))
end




abstract type AbstractHamiltonianTerms{HS} <: Hamiltonian end
Base.pairs(ht::AbstractHamiltonianTerms) = pairs(merge(pairs.(getfield(ht, :hs))...))
getHS(::Type{<:AbstractHamiltonianTerms{HS}}) where {HS} = HS
getHS(::AbstractHamiltonianTerms{HS}) where {HS} = HS
getHS(h::Type{<:Hamiltonian}) = (h,)

H_types(t::Type{<:Hamiltonian}) = tuple(getHS(t).parameters...)
H_types(h::Hamiltonian) = H_types(typeof(h))

"""
Collect all pairs from all hamiltonian terms
"""
@generated function Base.pairs(ht::AbstractHamiltonianTerms) 
    hs = H_types(ht)
    # Get all ParamVal fields from all Hamiltonian terms
    all_kvs = []
    for (idx, ham_type) in enumerate(hs)
        fnames = fieldnames(ham_type)
        ftypes = fieldtypes(ham_type)
        for (fname, ftype) in zip(fnames, ftypes)
            if ftype <: ParamVal
                # Access via ht.hs[idx].fieldname
                push!(all_kvs, Expr(:kw, fname, :(getproperty(getfield(getfield(ht, :hs), $idx), $(QuoteNode(fname))))))
            end
        end
    end
    tuple_exp = Expr(:tuple, Expr(:parameters, all_kvs...))
    return :(pairs($(tuple_exp)))
end

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


include("HamiltonianTerms.jl")



export HamiltonianTerms
include("DeltaRule.jl")
include("Quadratic.jl")
include("Quartic.jl")
include("MagField.jl")
include("Ising.jl")
include("ConstantHam.jl")
include("IsingOLD.jl")
include("GaussianBernoulli.jl")
include("Clamping.jl")
include("DepolarisationField.jl")
include("DeltaH.jl")

    