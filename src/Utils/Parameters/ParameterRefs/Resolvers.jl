###
# These function transform a parameterref method call into a number
# I.e. fills and/or contractions of parameterrefs
###
const last_exptree = Ref{GenExpressionTree}(GenExpressionTree())

## Reductions
# performance
@inline (rc::RefMult)(args::NT; genid = TreeID(rc), idxs...) where NT = @inline refmult_type(rc, args, (;idxs...); genid)
refmult_type_exp = nothing
@inline @generated function refmult_type(rc::RefMult{Refs}, args, idxs; genid) where Refs
    exptree = GenExpressionTree(genid(), :refmult_type)

    # fill_indices = idxs.parameters[4].parameters[1] # These are the indices to be filled with a definite value
    fill_indices = fieldnames(idxs)

    # The indices that are not filled are summed over
    sum_indices = tuple(setdiff(ref_indices(rc()), fill_indices)...) # These are the indices to be contracted
    exp = nothing
    if simplemult(rc())
        if isempty(sum_indices)
            exp = :(fill_ref(rc, idxs, args, $(reftype(rc(), args)), genid = $(nextID(genid()))))
            @goto wrapf
        end
        if isempty(fill_indices)
            exp = :(simple_contraction(rc, args; genid = $(nextID(genid()))))
            @goto wrapf
        end
    end

    leftref = first(Refs)[1]
    rightref = last(Refs)[1]
    lrcontract = contract_indices(leftref, rightref, Val(fill_indices))
    if isempty(lrcontract)  ## Left and right are not contracting
        # exp = :((@inline rc[1](args; genid = $(nid), idxs...)) * (@inline rc[2](args; genid = $(nid), idxs...)))
        exp = :(($(rc()[1])(args; genid = $(nextID(genid())), idxs...)) * ($(rc()[2])(args; genid = $(nextID(genid())), idxs...)))

        # exp = :($(expand_exp(rc()[1]))*$(expand_exp(rc()[2])))
        @goto wrapf #Wrap the function around it
    end
    lefttype = reftype(leftref, args)
    righttype = reftype(rightref, args)

    exp = :(reduce_contraction($leftref, $rightref, idxs, args, $(Val.(lrcontract)), $lefttype, $righttype; genid = $(nextID(genid()))))

    @label wrapf
    exp = expr_F_wrap(rc(), exp) # Wrap the whole result in a function if present
    global refmult_type_exp = quote
        # $(unpack_keyword_expr(idxs, :idxs))
        @inline $exp
    end
    # global refmult_type_exp = exp
    exp = refmult_type_exp
    setexpr!(exptree, exp)
    global last_exptree[] = mergetree(last_exptree[], exptree)
    return exp
end



reduce_contraction_exp = nothing
@inline @generated function reduce_contraction(vlike, mlike, idxs, @specialize(args), c_symbvals , ::VecLike, ::SparseMatrixRef; genid)
    exptree = GenExpressionTree(genid(), :reduce_contraction_vlike_sparsematrixref)
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
    setexpr!(exptree, reduce_contraction_exp)
    # error("Tree: $exptree, id: $(genid())")

    global last_exptree[] = mergetree(last_exptree[], exptree)
    return reduce_contraction_exp
end

@inline @generated function reduce_contraction(vleft, vright, idxs, args, contract_symb , ::VecLike, ::VecLike; genid)
    exptree = GenExpressionTree(genid(), :reduce_contraction_vlike_vlike)
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
    setexpr!(exptree, reduce_contraction_exp)
    global last_exptree[] = mergetree(last_exptree[], exptree)
    return reduce_contraction_exp
end

fill_ref_exp = nothing
@generated function fill_ref(rm::RefMult, idxs, args, ::VecLike; genid)
    exptree = GenExpressionTree(genid(), :fill_ref)
    idxs_key = idxs.parameters[4].parameters[1][1]

    global fill_ref_exp = quote
        $idxs_key = idxs[$(QuoteNode(idxs_key))] # get the sum index
        # cumsum = zero(promote_eltype($(struct_ref_exp.(vec_refs(rc(), args))...)))
        cumsum = $(expand_exp(rm()))
        return cumsum
    end

    setexpr!(exptree, fill_ref_exp)
    global last_exptree[] = mergetree(last_exptree[], exptree)
    return fill_ref_exp
end

simple_contraction_exp = nothing
@generated function simple_contraction(refs, ::VecLike; genid)
    exptree = GenExpressionTree(genid(), :simple_contraction)
    
    global simple_contraction_exp = quote
        cumsum = zero(promote_eltype($(struct_ref_exp.(refs)...)))
        @turbo for i in eachindex($(struct_ref_exp.(refs[1])))
            cumsum += $(expand_exp(refs))
        end
        return cumsum
    end

    setexpr!(exptree, simple_contraction_exp)
    global last_exptree[] = mergetree(last_exptree[], exptree)
    return simple_contraction_exp
end