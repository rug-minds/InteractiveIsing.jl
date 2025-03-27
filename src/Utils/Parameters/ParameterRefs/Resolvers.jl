###
# These function transform a parameterref method call into a number
# I.e. fills and/or contractions of parameterrefs
###

## Reductions
# performance
@inline (rc::RefMult)(@specialize(args); @specialize(idxs...)) = @inline refmult_type(rc, args, (;idxs...))
refmult_type_exp = nothing
@inline @generated function refmult_type(@specialize(rc::RefMult{Refs}), args, idxs) where Refs
    # fill_indices = idxs.parameters[4].parameters[1] # These are the indices to be filled with a definite value
    fill_indices = fieldnames(idxs)


    # The indices that are not filled are summed over
    sum_indices = tuple(setdiff(ref_indices(rc()), fill_indices)...) # These are the indices to be contracted
    exp = nothing
    if simplemult(rc())
        if isempty(sum_indices)
            exp = :(fill_ref(rc, idxs, args, $(reftype(rc(), args))))
            @goto wrapf
        end
        if isempty(fill_indices)
            exp = :(simple_contraction(rc, args))
            @goto wrapf
        end
    end

    leftref = first(Refs)[1]
    rightref = last(Refs)[1]
    lrcontract = contract_indices(leftref, rightref, Val(fill_indices))
    if isempty(lrcontract)  ## Left and right are not contracting
        exp = :(rc[1](args; idxs...) * rc[2](args; idxs...))
        @goto wrapf
    end
    lefttype = reftype(leftref, args)
    righttype = reftype(rightref, args)

    exp = :(reduce_contraction($leftref, $rightref, idxs, args, $(Val.(lrcontract)), $lefttype, $righttype))

    @label wrapf
    exp = expr_F_wrap(rc(), exp) # Wrap the whole result in a function if present
    exp = quote
        # $(unpack_keyword_expr(idxs, :idxs))
        @inline $exp
    end
    global refmult_type_exp = exp
    return exp
end

reduce_contraction_exp = nothing
@inline @generated function reduce_contraction(vlike, mlike, idxs, @specialize(args), c_symbvals , ::VecLike, ::SparseMatrixRef)
    global reduce_contraction_exp = quote
        @inline j = idxs[:j]
        vector = $(struct_ref_exp(vlike)...)
        sp_matrix = $(struct_ref_exp(mlike)...)
        cumsum = zero((@inline promote_eltype(vector, sp_matrix)))
        # cumsum = zero(Float32)
        
        @turbo for ptr in nzrange(sp_matrix, j)
            i = sp_matrix.rowval[ptr]
            wij = sp_matrix.nzval[ptr]
            cumsum += wij * vector[i]
        end
        return cumsum
    end
    return reduce_contraction_exp
end

@inline @generated function reduce_contraction(vleft, vright, idxs, args, contract_symb , ::VecLike, ::VecLike)
    fill_keys = idxs.parameters[4].parameters[1]
    c_symb = contract_symb.parameters[1] |> getval
    global reduce_contraction_exp = quote
        left_vec = $(struct_ref_exp(vleft)...)
        right_vec = $(struct_ref_exp(vright)...)
        (;$(fill_keys...)) = idxs # get the sum index
        cumsum = zero(promote_eltype($(struct_ref_exp(vleft)...), $(struct_ref_exp(vright)...)))
        # cumsum = zero(Float32)
        @simd for $c_symb in eachindex(right_vec)
            # left = $(expand_exp(vleft))
            left = $(expand_exp(vleft))
            right = $(expand_exp(vright))
            cumsum += left * right
        end
        return cumsum
    end
    return reduce_contraction_exp
end

fill_ref_exp = nothing
@generated function fill_ref(rm::RefMult, idxs, args, ::VecLike)
    idxs_key = idxs.parameters[4].parameters[1][1]

    global fill_ref_exp = quote
        $idxs_key = idxs[$(QuoteNode(idxs_key))] # get the sum index
        # cumsum = zero(promote_eltype($(struct_ref_exp.(vec_refs(rc(), args))...)))
        cumsum = $(expand_exp(rm()))
        return cumsum
    end
    return fill_ref_exp
end

simple_contraction_exp = nothing
@generated function simple_contraction(refs, ::VecLike)
    global simple_contraction_exp = quote
        cumsum = zero(promote_eltype($(struct_ref_exp.(refs)...)))
        @turbo for i in eachindex($(struct_ref_exp.(refs[1])))
            cumsum += $(expand_exp(refs))
        end
        return cumsum
    end
    return simple_contraction_exp
end