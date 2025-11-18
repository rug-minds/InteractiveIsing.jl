struct HamiltonianTerms{Hs <: Tuple{Vararg{Hamiltonian}}} <: AbstractHamiltonianTerms{Hs}
    hs::Hs
end

"""
From the type initialize
"""
function HamiltonianTerms{HS}(g::AbstractIsingGraph) where HS <: Tuple{Vararg{Hamiltonian}}
    HamiltonianTerms{HS}(ntuple(i->HS.parameters[i](g), length(HS.parameters)))
end

function changeterm(hts, newham)
    hams = hamiltonians(hts)
    newhams = tuple((Base.typename(typeof(newham))  == Base.typename(typeof(h)) ? newham : h for h in hams)...)
    return HamiltonianTerms{typeof(newhams)}(newhams)
end

HamiltonianTerms(hs::Type{<:Hamiltonian}...) = HamiltonianTerms{Tuple{hs...}}
HamiltonianTerms(hs::Hamiltonian...) = HamiltonianTerms{Tuple{typeof.(hs)...}}(hs)

hamiltonians(hts::HamiltonianTerms) = getfield(hts, :hs)
numhamiltonians(hts::HamiltonianTerms) = length(hamiltonians(hts))
numhamiltonians(hts::Type{<:HamiltonianTerms}) = length(hts.parameters[1].parameters)

htypes(hts::Type{<:HamiltonianTerms}) = htypes(hts.parameters[1])

Base.:+(h1::Hamiltonian, h2::Hamiltonian) = HamiltonianTerms((h1, h2))
Base.:+(h1::Hamiltonian, h2::HamiltonianTerms) = HamiltonianTerms((h1, hamiltonians(h2)...))
Base.:+(h1::HamiltonianTerms, h2::Hamiltonian) = HamiltonianTerms((hamiltonians(h1)..., h2))

# @generated function paramnames(h::Union{Type{<:HamiltonianTerms{Hs}}, HamiltonianTerms{Hs}}) where Hs
#     names = Symbol[]
#     # _paramnames = tuple(Iterators.Flatten(fieldnames.(Hs.parameters)...)...)
#     for h in Hs.parameters
#         push!(names, fieldnames(h)...)
#     end
#     return :($(tuple(names...)))
# end

function setparam(ham::Hamiltonian, field, paramval)
    fnames = paramnames(ham)
    found = findfirst(x->x==field, fnames)
    if isnothing(found)
        error("Field $field not found in Hamiltonian $ham")
    end
    newfields = (i == found ? paramval : ham.fieldnames[i] for i in eachindex(fnames))  # Regenerate all the subhamiltonians
    Base.typename(typeof(ham)).wrapper(newfields...)                                    # Create a new Hamiltonian type
end
export setparam

function paramactivation!(g::AbstractIsingGraph, param::Symbol, activation::Bool)
    g.hamiltonian = paramactivation(g.hamiltonian, param, activation)
    refresh(g)
end

function h_param(g::AbstractIsingGraph, param::Symbol)
    return getproperty(g.hamiltonian, param)
end
export paramactivation!, h_param


function paramactivation(ham::Hamiltonian, param::Symbol, activation::Bool)
    initialparam = getproperty(ham, param)
    if activation
        return activateparam(ham, param)
    else
        return deactivateparam(ham, param)
    end
end

function activateparam(ham::Hamiltonian, param::Symbol)
    initialparam = getproperty(ham, param)
    setparam(ham, param, activate(initialparam))
end

function activateparam(hts::HamiltonianTerms, param::Symbol)
    newham = activateparam(gethamiltonian(hts, param), param)
    changeterm(hts, newham)
end

function deactivateparam(ham::Hamiltonian, param::Symbol)
    initialparam = getproperty(ham, param)
    setparam(ham, param, deactivate(initialparam))
end

function deactivateparam(hts::HamiltonianTerms, param::Symbol)
    newham = deactivateparam(gethamiltonian(hts, param), param)
    changeterm(hts, newham)
end

function sethomogeneousparam(ham::Hamiltonian, param::Symbol)
    initialparam = getproperty(ham, param)
    newparam = HomogeneousParam(initialparam.val, length(initialparam.val); description = initialparam.description)
    setparam(ham, param, newparam)
end

function sethomogeneousparam(hts::HamiltonianTerms, param::Symbol)
    newham = sethomogeneousparam(gethamiltonian(hts, param), param)
    changeterm(hts, newham)
end

export deactivateparam, sethomogeneousparam, activateparam

struct LazyTermField{HTS, substruct, paramname} end


ltf_exp = nothing
@generated function (ltf::LazyTermField{HTS, substruct, paramname})(hts) where {HTS, substruct, paramname} 
    global ltf_exp = :(getfield(hts, :hs)[$substruct].$(paramname))
    return ltf_exp
end

"""
From the set of Hamiltonians, directly get a paramval from an underlying Hamiltonian
"""
# function Base.getproperty(h::HamiltonianTerms, paramname::Symbol)
#     # constant propagation flag:
#     if paramname == :hamiltonians
#         return getfield(h, :hs)
#     end
#     getparam(h, Val(paramname))
# end
@inline @generated function Base.getproperty(h::HamiltonianTerms{HS}, paramname::Symbol) where HS
    args = (;)
    for (hidx, H) in enumerate(HS.parameters)
        for (fidx, fieldname) in enumerate(fieldnames(H))
            args = (;args..., fieldname => LazyTermField{typeof(h), hidx, fieldname}())
        end
    end
    return :(getfield($args, paramname)(h))
end

"""
Get a param
"""
@generated function getparam(h::HamiltonianTerms{Hs}, paramnameval::Val{paramname}) where {Hs, paramname}
    for (hidx, H) in enumerate(Hs.parameters)
        for (fidx, fieldname) in enumerate(fieldnames(H))
            if paramname == fieldname
                ft = fieldtypes(H)[fidx]
                return :(getfield(h,:hs)[$hidx].$(paramname)::$(ft))
            end
        end
    end
    error("Parameter $paramname not found in any of the Hamitonians")
end

"""
Get a hamiltonian from a type
"""
function gethamiltonian(hts::HamiltonianTerms, t::Type)
    for h in hamiltonians(hts)
        if typeof(h) <: t
            return h
        end
    end
    error("Type $t not found in any of the Hamiltonians")
end

"""
Get the hamiltonian from a parameter name
"""
function gethamiltonian(hts::HamiltonianTerms, t::Symbol)
    for h in hamiltonians(hts)
        if t in fieldnames(typeof(h))
            return h
        end
    end
    error("Type $t not found in any of the Hamiltonians")
end
export gethamiltonian

# Fallback for fieldnames for a hamltonian
Base.fieldnames(::Hamiltonian) = tuple()

"""
Iterating over terms forwards to the hamiltonians
"""
Base.iterate(hts::HamiltonianTerms, state = 1) = iterate(getfield(hts, :hs), state)

Base.broadcastable(c::HamiltonianTerms) = getfield(c, :hs)

"""
Get a hamiltonian from the set of hamiltonians
"""
Base.getindex(hts::HamiltonianTerms, idx::Int) = getfield(hts, :hs)[idx]


update!(algo, a,b) = nothing
"""
If updating functions are defined, update
"""
update!_expr = quote end
@inline @generated function update!(algo, hts::HamiltonianTerms{Hs}, args) where Hs
    # names = paramnames(hts)
    num_h = numhamiltonians(hts)
    global update!_expr = quote
        $([:(update!(algo, hamiltonians(hts)[$i]::$(Hs.parameters[i]), args)) for i in 1:num_h]...)
    end
    return update!_expr
end

@inline function init!(hts::HamiltonianTerms{Hs}, g) where Hs
    @inline init!.(hamiltonians(hts), Ref(g))
    return hts
end

init!(hts::Any, g) = hts
export init!


# Hamiltonian
function deltaH(hts::HamiltonianTerms)
    return reduce(+, deltaH.(hts))
end
