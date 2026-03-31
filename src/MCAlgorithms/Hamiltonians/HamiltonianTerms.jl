struct HamiltonianTerms{Hs <: Tuple{Vararg{Hamiltonian}}} <: AbstractHamiltonianTerms{Hs}
    hs::Hs
end

"""
From the type initialize
"""
function HamiltonianTerms{HS}(g::AbstractIsingGraph) where HS <: Tuple{Vararg{Hamiltonian}}
    HamiltonianTerms{HS}(ntuple(i->HS.parameters[i](g), length(HS.parameters)))
end

HamiltonianTerms(hs::Type{<:Hamiltonian}...) = HamiltonianTerms{Tuple{hs...}}
HamiltonianTerms(hs::Hamiltonian...) = HamiltonianTerms{Tuple{typeof.(hs)...}}(hs)

# BASE EXTENSIONS
Base.:+(h1::Hamiltonian, h2::Hamiltonian) = HamiltonianTerms((h1, h2))
Base.:+(h1::Hamiltonian, h2::HamiltonianTerms) = HamiltonianTerms((h1, hamiltonians(h2)...))
Base.:+(h1::HamiltonianTerms, h2::Hamiltonian) = HamiltonianTerms((hamiltonians(h1)..., h2))

# Allow bare Types in + chains: Ising() + Quartic + Sextic
Base.:+(h1::Hamiltonian, ::Type{H}) where {H<:Hamiltonian} = h1 + H()
Base.:+(::Type{H}, h2::Hamiltonian) where {H<:Hamiltonian} = H() + h2
Base.:+(h1::HamiltonianTerms, ::Type{H}) where {H<:Hamiltonian} = h1 + H()
Base.:+(::Type{H}, h2::HamiltonianTerms) where {H<:Hamiltonian} = H() + h2
Base.:+(::Type{H1}, ::Type{H2}) where {H1<:Hamiltonian, H2<:Hamiltonian} = H1() + H2()


function changeterm(hts, newham)
    hams = hamiltonians(hts)
    newhams = tuple((Base.typename(typeof(newham))  == Base.typename(typeof(h)) ? newham : h for h in hams)...)
    return HamiltonianTerms{typeof(newhams)}(newhams)
end

@inline Base.map(f, hts::HamiltonianTerms) = @inline map(f, hamiltonians(hts))

@inline function Base.getproperty(h::HamiltonianTerms{HS}, paramname::Symbol) where HS
    paramname === :hs && return getfield(h, :hs)

    foundidxs = Int[]
    for (hidx, H) in enumerate(HS.parameters)
        if paramname in fieldnames(H)
            push!(foundidxs, hidx)
        end
    end

    if length(foundidxs) > 1
        error("Property $(paramname) exists in multiple Hamiltonian terms at indices $(foundidxs). Please first get the proper hamiltonian through normal indexing and then get the property.")
    end
    isempty(foundidxs) && return getfield(h, paramname)
    return getfield(getfield(h, :hs)[foundidxs[1]], paramname)
end

@generated function Base.getindex(h::HamiltonianTerms{HS}, ::Type{H}) where {HS, H}
    hams = HS.parameters
    findall(x-> x <: H, hams) |> (idxs -> isempty(idxs) ? :(error("Hamiltonian type $H not found in HamiltonianTerms")) : :(getfield(h, :hs)[$(idxs[1])]))
end

hamiltonians(hts::HamiltonianTerms) = getfield(hts, :hs)
numhamiltonians(hts::HamiltonianTerms) = length(hamiltonians(hts))
numhamiltonians(hts::Type{<:HamiltonianTerms}) = length(hts.parameters[1].parameters)

htypes(hts::Type{<:HamiltonianTerms}) = htypes(hts.parameters[1])



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

Base.@constprop :aggressive @inline getparam(hts::HamiltonianTerms, paramname::Symbol) =
    getparam(hts, Val(paramname))

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

"""
Get a param from a specific Hamiltonian term by term type and field name.

For the generated fast path, call with `Val(:fieldname)`.
"""
@generated function getparam(hts::HamiltonianTerms{Hs}, ::Type{H}, ::Val{fieldname}) where {Hs, H, fieldname}
    matching_idxs = Int[]
    for (hidx, Hterm) in enumerate(Hs.parameters)
        if Hterm <: H
            push!(matching_idxs, hidx)
        end
    end

    if isempty(matching_idxs)
        return :(error($(string("Hamiltonian type ", H, " not found in HamiltonianTerms"))))
    end

    if length(matching_idxs) > 1
        return :(error($(string(
            "Hamiltonian type ",
            H,
            " is ambiguous in HamiltonianTerms; matches indices ",
            matching_idxs,
            ". Please disambiguate before requesting field ",
            fieldname,
            "."
        ))))
    end

    hidx = matching_idxs[1]
    Hterm = Hs.parameters[hidx]
    for (fidx, fname) in enumerate(fieldnames(Hterm))
        if fname == fieldname
            ftype = fieldtypes(Hterm)[fidx]
            return :(getfield(getfield(hts, :hs)[$hidx], $(QuoteNode(fieldname)))::$(ftype))
        end
    end

    return :(error($(string("Field ", fieldname, " not found in Hamiltonian type ", Hterm))))
end

Base.@constprop :aggressive @inline getparam(hts::HamiltonianTerms, ::Type{H}, fieldname::Symbol) where {H} =
    getparam(hts, H, Val(fieldname))

@inline gethamiltonianfield(hts::HamiltonianTerms, ::Type{H}, fieldname::Val) where {H} =
    getparam(hts, H, fieldname)

Base.@constprop :aggressive @inline gethamiltonianfield(hts::HamiltonianTerms, ::Type{H}, fieldname::Symbol) where {H} =
    getparam(hts, H, Val(fieldname))

export getparam, gethamiltonianfield

# @inline function _merge_parameter_derivatives(current::NamedTuple, new::NamedTuple)
#     duplicate_keys = [k for k in keys(new) if k in keys(current)]
#     isempty(duplicate_keys) || error("Parameter derivative contains duplicate fields $(duplicate_keys).")
#     return merge(current, new)
# end

# @inline function parameter_derivative(hts::AbstractHamiltonianTerms, state::AbstractIsingGraph, args...)
#     total = NamedTuple()
#     for hterm in hts
#         applicable(parameter_derivative, hterm, state, args...) || continue
#         total = _merge_parameter_derivatives(total, parameter_derivative(hterm, state, args...))
#     end
#     return total
# end

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


update!(algo, a, state, proposal) = nothing
"""
If updating functions are defined, update
"""
update!_expr = quote end
@inline @generated function update!(algo, hts::HamiltonianTerms{Hs}, state, proposal) where Hs
    # names = paramnames(hts)
    num_h = numhamiltonians(hts)
    global update!_expr = quote
        $([:(@inline update!(algo, hamiltonians(hts)[$i]::$(Hs.parameters[i]), state, proposal)) for i in 1:num_h]...)
    end
    return update!_expr
end

@inline update!(::LangevinDynamics, hts::HamiltonianTerms{Hs}, state::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, state, proposal)
@inline update!(::KineticMC, hts::HamiltonianTerms{Hs}, state::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, state, proposal)


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
