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

function _template_parameter_names(::Type{H}) where {H}
    hasfield(H, :parameters) || return ()

    P = fieldtype(H, :parameters)
    P <: Parameters || return ()

    entries_type = P.parameters[1]
    entries_type <: NamedTuple || return ()
    return fieldnames(entries_type)
end

_hamiltonian_property_names(::Type{H}) where {H} = propertynames_type(H)

propertynames_type(::Type{H}) where {H<:HamiltonianTerm} =
    _hamiltonianterm_propertynames(H)

@generated function _hamiltonianterms_propertynames(::Type{HTS}) where {HTS<:HamiltonianTerms}
    hs_type = HTS.parameters[1]
    fields = fieldnames(HTS)
    names = Symbol[fields...]
    seen = Set{Symbol}(fields)
    ambiguous = Set{Symbol}()

    for H in hs_type.parameters
        for name in propertynames_type(H)
            name in fields && continue
            if name in seen
                push!(ambiguous, name)
            else
                push!(seen, name)
                push!(names, name)
            end
        end
    end

    unique_names = Tuple(name for name in names if !(name in ambiguous))
    return :($(QuoteNode(unique_names)))
end

Base.propertynames(hts::HamiltonianTerms; private = false) =
    _hamiltonianterms_propertynames(typeof(hts))

@inline function Base.getproperty(h::HamiltonianTerms{HS}, paramname::Symbol) where HS
    paramname === :hs && return getfield(h, :hs)

    foundidxs = Int[]
    hs = getfield(h, :hs)
    for (hidx, hterm) in enumerate(hs)
        if paramname in propertynames(hterm)
            push!(foundidxs, hidx)
        end
    end

    if length(foundidxs) > 1
        error("Property $(paramname) exists in multiple Hamiltonian terms at indices $(foundidxs). Please first get the proper hamiltonian through normal indexing and then get the property.")
    end
    isempty(foundidxs) && return getfield(h, paramname)
    return getproperty(hs[foundidxs[1]], paramname)
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
        if paramname in _template_parameter_names(H)
            return :(getproperty(getfield(h, :hs)[$hidx], $(QuoteNode(paramname))))
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
        if t in propertynames(h)
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

    if fieldname in _template_parameter_names(Hterm)
        return :(getproperty(getfield(getfield(hts, :hs)[$hidx], :parameters), $(QuoteNode(fieldname))))
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

# @inline function parameter_derivative(hts::AbstractHamiltonianTerms, model::AbstractIsingGraph, args...)
#     total = NamedTuple()
#     for hterm in hts
#         applicable(parameter_derivative, hterm, model, args...) || continue
#         total = _merge_parameter_derivatives(total, parameter_derivative(hterm, model, args...))
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

"""
    update!(algo, hterm, model, proposal)

Default Hamiltonian update hook.

Hamiltonian terms that maintain cached state can specialize this method. Terms
without cached state fall through to `nothing`.
"""
update!(algo, a, model, proposal) = nothing

"""
    update!(algo, hterm::Hamiltonian, model, proposal::MultiSpinProposal)

Fallback cache update for one Hamiltonian term after an accepted multi-spin
move.

Rejected proposals do not update anything. Accepted proposals are decomposed
with `subproposals(proposal)` and forwarded to the term's single-spin
`update!` method in proposal order.
"""
@inline function update!(algo, hterm::Hamiltonian, model::AbstractIsingGraph, proposal::MultiSpinProposal)
    isaccepted(proposal) || return nothing
    for fp in subproposals(proposal)
        @inline update!(algo, hterm, model, fp)
    end
    return nothing
end

"""
    update!(algo, hts::HamiltonianTerms, model, proposal::MultiSpinProposal)

Fallback cache update for a set of Hamiltonian terms after an accepted
multi-spin move.

The proposal is decomposed into single-spin `FlipProposal`s and each
subproposal is passed through the usual `HamiltonianTerms` update path. This
preserves the same sequential behavior as applying the flips one at a time.
"""
@inline function update!(algo, hts::HamiltonianTerms, model::AbstractIsingGraph, proposal::MultiSpinProposal)
    isaccepted(proposal) || return nothing
    for fp in subproposals(proposal)
        @inline update!(algo, hts, model, fp)
    end
    return nothing
end

"""
    update!(::Metropolis, hts::HamiltonianTerms, model, proposal::MultiSpinProposal)

Metropolis-specific multi-spin update fallback.

This method removes dispatch ambiguity with the generated `HamiltonianTerms`
single-proposal update path. It has the same sequential decomposition semantics
as the generic multi-spin fallback.
"""
@inline function update!(algo::Metropolis, hts::HamiltonianTerms, model::AbstractIsingGraph, proposal::MultiSpinProposal)
    isaccepted(proposal) || return nothing
    for fp in subproposals(proposal)
        @inline update!(algo, hts, model, fp)
    end
    return nothing
end

"""
    update!(algo, hts::HamiltonianTerms, model, proposal)

Generated update dispatcher for Hamiltonian term collections.

For a single proposal, this unrolls the update over all terms in `hts`. Terms
without a specialized update method fall through to the default no-op method
above.
"""
update!_expr = quote end
@inline @generated function update!(algo, hts::HamiltonianTerms{Hs}, model, proposal) where Hs
    # names = paramnames(hts)
    num_h = numhamiltonians(hts)
    global update!_expr = quote
        $([:(@inline update!(algo, hamiltonians(hts)[$i]::$(Hs.parameters[i]), model, proposal)) for i in 1:num_h]...)
    end
    return update!_expr
end

@inline update!(::LocalLangevin, hts::HamiltonianTerms{Hs}, model::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, model, proposal)
@inline update!(::LocalLangevin, hts::HamiltonianTerms, model::AbstractIsingGraph, proposal::MultiSpinProposal) = update!(Metropolis(), hts, model, proposal)
@inline update!(::GlobalLangevin, hts::HamiltonianTerms{Hs}, model::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, model, proposal)
@inline update!(::GlobalLangevin, hts::HamiltonianTerms, model::AbstractIsingGraph, proposal::MultiSpinProposal) = update!(Metropolis(), hts, model, proposal)
@inline update!(::BlockLangevin, hts::HamiltonianTerms{Hs}, model::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, model, proposal)
@inline update!(::BlockLangevin, hts::HamiltonianTerms, model::AbstractIsingGraph, proposal::MultiSpinProposal) = update!(Metropolis(), hts, model, proposal)
@inline update!(::KineticMC, hts::HamiltonianTerms{Hs}, model::AbstractIsingGraph, proposal::FlipProposal) where {Hs} = update!(Metropolis(), hts, model, proposal)


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
