

reduce_contraction_exp = nothing
@inline @generated function reduce_contraction(vlike, mlike, idxs, args, c_symbvals , ::VecLike, ::SparseMatrixRef)
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