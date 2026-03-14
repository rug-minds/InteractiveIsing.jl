"""
An hamiltonian is a struct that holds parameters
"""

@inline Base.merge(nt::NamedTuple, h::Hamiltonian) = (;nt..., pairs(h)...)
"""
Pairs gives an iterator over the ParamTensor fields of the Hamiltonian
"""
@generated function Base.pairs(h::Hamiltonian)
    fnames = fieldnames(h)
    ftypes = fieldtypes(h)
    names_types = zip(fnames, ftypes)
    pvals_names = (x[1] for x in names_types if x[2] <: ParamTensor) 
    tuple_exp = Expr(:tuple, Expr(:parameters, (Expr(:kw, name, :(getproperty(h, $(QuoteNode(name))))) for name in pvals_names)...))
    return :(pairs($(tuple_exp)))
end

H_types(t::Type{<:Hamiltonian}) = tuple(getHS(t).parameters...)
H_types(h::Hamiltonian) = H_types(typeof(h))

function _paramnames(h::Type{<:Hamiltonian}, all_keys = Symbol[])
    for h in getHS(h)
        if h <: AbstractHamiltonianTerms
            _paramnames(h, all_keys)
        else
            fnames = fieldnames(h)
            ftypes = fieldtypes(h)
            for (idx, name) in enumerate(fnames)
                if ftypes[idx] <: ParamTensor
                    push!(all_keys, name)
                end
            end
        end
    end
end

"""
For a Hamiltonian, return all fieldnames that are a ParamTensor
"""
@generated function paramnames(h::Union{Hamiltonian, AbstractHamiltonianTerms})
    h = getHS(h)
    all_keys = Symbol[]
    _fieldnames = fieldnames.(h)
    _fieldtypes = fieldtypes.(h)
    for i in eachindex(_fieldnames)
        _paramnames(h[i], all_keys)
    end
    paramtensor_params = (all_keys...,)
    return (:($paramtensor_params))
end


function update!(::Hamiltonian, args)
    return nothing
end


"""
    Fallback required derived hamiltonian for an algorithm, 
    indicating that that the algorithm did not define a required derived hamiltonian.
"""
function requires(::Type{<:MCAlgorithm})
    throw(ArgumentError("This algorithm did not define a required derived Hamiltonian."))
end
