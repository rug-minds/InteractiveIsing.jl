# Generated functions for ParameterRefs
# These are separated to avoid circular dependencies in generated function calls

# ParameterRef generated functions
@generated function get_ref(p, args::DataType)
    return :(args.$(ref_symb(p)))
end

@generated function Base.get(pr::ParameterRef, args)
    ref = struct_ref_exp(pr)
    exp = quote $(ref...) end
    return exp
end

get_ref_exp = nothing
@generated function get_ref(p, args::Union{<:NamedTuple, <:Base.Pairs})
    symb = type_apply(ref_symb, p) ## TODO: What does this do
    # refs = refmap(Val(symb))
    # global get_ref_exp = :($(build_getproperty_chain(:args, refs)))
    global get_ref_exp = :(getproperty(args, $(QuoteNode(symb))))
    return get_ref_exp
end

@inline @generated function (pr::ParameterRef)(args::AS, idxs) where AS
    global paramref_type_exp = generate_block(pr(), args, idxs, rem_lnn = false)
    return paramref_type_exp
end

# RefMult generated functions
@generated function get_prefs(p::RefMult{Refs}) where Refs
    ret = _get_prefs(p())
    return :($ret)
end

@generated function ispure(rm::RefMult{Refs}) where {Refs}
    pure = _ispure(rm())
    return :($pure)
end

@generated function ref_indices(rc::RefMult{Refs}) where Refs
    t = _ref_indices(rc())
    return :($t)
end

@generated function indices_set(ref1::AbstractParameterRef, ref2::AbstractParameterRef, filled_indices = nothing)
    idcs1 = _ref_indices(ref1())
    idcs2 = _ref_indices(ref2())
    _union = tuple(union(idcs1, idcs2)...)
    if !(filled_indices <: Nothing) # If indices are filled, they are esentially not there
        _union = tuple((setdiff(_union, getval(filled_indices)))...)
    end
    # Return each symbol without duplicates
    return :($_union)
end

@generated function contract_indices(ref1, ref2, filled_indices = nothing)
    idcs1 = _ref_indices(ref1())
    idcs2 = _ref_indices(ref2())
    _intersect = tuple(intersect(idcs1, idcs2)...)
    if !(filled_indices <: Nothing) && !(filled_indices == @NamedTuple{}) # If indices are filled, they are esentially not there
        _intersect = tuple((setdiff(_intersect, getval(filled_indices)))...)
    end
    return :($_intersect)
end

# RefReduce generated functions
@generated function ref_indices(rr::RefReduce)
    t = _ref_indices(rr())
    return :($t)
end

@generated function ispure(rr::RefReduce)
    pure = _ispure(rr())
    return :($pure)
end

# @inline @generated function (rr::RefReduce)(args::NT, idxs) where NT
#     rr = simplify(rr)
#     global refreduce_type_args = [rr, args, idxs]
#     global refreduce_type_exp = generate_block(rr(), args, idxs)
#     return refreduce_type_exp
# end
