
refmult_type_exp = nothing
@inline (rc::RefMult{Refs})(args::NT; idxs...) where {Refs,NT} = @inline (rc)(args, (;idxs...))
@inline @generated function (rc::RefMult{Refs})(args::NT, idxs) where {Refs,NT}
    global refmult_type_exp = generate_block(rc(), args, idxs)
    return refmult_type_exp    
end

refreduce_type_exp = nothing
refreduce_type_args = nothing
"""
Inline a refreduce like: ref1(args; idxs...) +/- ref2(args; idxs...) ...
"""
@inline (rr::RefReduce)(args::NT; idxs...) where NT = @inline (rr)(args, (;idxs...))

@inline @generated function (rr::RefReduce)(args::NT, idxs) where NT
    rr = simplify(rr)
    global refreduce_type_args = [rr, args, idxs]
    global refreduce_type_exp = generate_block(rr(), args, idxs)
    return refreduce_type_exp
end