### Reduce
"""
Combines object with the same indices into a single object
These are objects that are added or subtracted
"""
struct RefReduce{Refs, reduce_fs, F, D} <: AbstractParameterRef 
    data::D
end

RefReduce{A,B,C,D}() where {A,B,C,D} = RefReduce{A,B,C,Nothing}(nothing)

RefReduce(refs::Tuple, reduce_fs, data = nothing; func = tuple()) = RefReduce{refs, reduce_fs, func ,typeof(data)}(data)
# ref_indices(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = tuple(union(ref_indices.(Refs)...)...)
# ref_indices(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = tuple(union(ref_indices.(Refs)...)...)
get_prefs(rr::RefReduce{Refs}) where Refs = (Refs)
get_reduce_fs(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = reduce_fs
getF(rr::RefReduce{R,rf,F}) where {R,rf,F} = F

function return_type(rr::RefReduce{Refs, reduce_fs, F, D}, args) where {Refs, reduce_fs, F, D}
    promote_type(return_type.(Refs, Ref(args))...)
end

Base.length(rr::RefReduce) = length(get_prefs(rr))

Base.:+(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefReduce((p1, p2), (+,))
Base.:-(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefReduce((p1, p2), (-,))
Base.:+(p1::AbstractParameterRef, p2::RefReduce) = RefReduce((p1, get_prefs(p2)...), (+, get_reduce_fs(p2)...))
Base.:-(p1::AbstractParameterRef, p2::RefReduce) = RefReduce((p1, get_prefs(p2)...), (-, get_reduce_fs(p2)...))
Base.:+(p1::RefReduce, p2::AbstractParameterRef) = RefReduce((get_prefs(p1)..., p2), (get_reduce_fs(p1)...,+))
Base.:-(p1::RefReduce, p2::AbstractParameterRef) = RefReduce((get_prefs(p1)..., p2), (get_reduce_fs(p1)...,-))

Base.:-(p::RefReduce) = RefReduce(get_prefs(p), get_reduce_fs(p), p.data, func = tuple(-))
Base.:^(p::RefReduce, pow::Real) = RefReduce(get_prefs(p), get_reduce_fs(p), p.data, func = (^, pow))

@generated function ref_indices(rr::RefReduce)
    t = tuple(union(ref_indices.(get_prefs(rr()))...)...)
    return :($t)
end
issparse(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = all(issparse.(Refs))

@generated function ispure(rr::RefReduce)
    pure = true
    first_indexset = ref_indices(gethead(get_prefs(rr())))
    for ref in get_prefs(rr())
        idxs = ref_indices(ref)
        pure = ispure(ref) && idxs ∈ first_indexset
        if !pure
            break
        end
    end
    # return :(Val($pure))
    return :($pure)
end

function expand_exp(rr::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs}
    refs = collect(Refs)  # Convert tuple to array for easier manipulation
    ops = collect(reduce_fs)
    
    # Start with the first expression
    expr = expand_exp(refs[1])
    
    # Build the expression by applying operators in sequence
    for i in 1:length(ops)
        expr = Expr(:call, ops[i], expr, expand_exp(refs[i+1]))
    end
    
    return expr
end


refreduce_resolve_exp = nothing
function resolve_exp(rr::RefReduce, indorsymb)
    refs = get_prefs(rr)
    ops = get_reduce_fs(rr)
    
    # Start with the first expression
    expr = expand_exp(refs[1])
    
    # Build the expression by applying operators in sequence
    for i in 1:length(ops)
        expr = Expr(:call, ops[i], expr, resolve_exp(refs[i+1], indorsymb))
    end
    
    global refreduce_resolve_exp = expr
    return expr
end

"""
instead of inlining the operators like (-)(r1,r2), inline them like r1-r2+...
"""
function expand_exp_full(rr::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs}
    _expand_exp_full(rr, 1)
end

function _expand_exp_full(rr::RefReduce{Refs, reduce_fs}, idx) where {Refs, reduce_fs}
    if idx>length(Refs)
        return :()    
    end
    return quote $(Refs[idx]) $(reduce_fs[idx]) $(_expand_exp_full(rr, idx+1)...) end
end

function struct_ref_exp(rm::RefReduce{Refs}) where Refs
    return tuple(Iterators.flatten(struct_ref_exp.(Refs))...)
end

function expand_to_calls(rm::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs}
    refs = collect(Refs)  # Convert tuple to array for easier manipulation
    ops = collect(reduce_fs)
    
    ref2expr = ref -> Expr(:call, ref, :($(Expr(:parameters, :(idxs...)))), :args)
    # ref2expr = ref -> Expr(:call, ref, :($(Expr(:parameters, :($(Expr(:kw, :genid, :(nextID(genid()))))), :(idxs...)))), :args)

    expr = ref2expr(Refs[1])
    
    # Build the expression by applying operators in sequence
    for i in 1:length(ops)
        expr = Expr(:call, ops[i], expr, ref2expr(Refs[i+1]))
    end
    
    
    return expr
end

@inline function (rr::RefReduce)(args::NT; genid = nothing, idxs...) where NT
    @inline refreduce_type(rr, args, (;idxs...); genid)
end

refreduce_type_exp = nothing
refreduce_type_args = nothing

"""
Inline a refreduce like: ref1(args; idxs...) +/- ref2(args; idxs...) ...
"""
@inline @generated function refreduce_type(rr, @specialize(args), idxs; genid = TreeID(rr))
    exptree = GenExpressionTree(genid(), :refreduce_type)
    global refreduce_type_args = [rr, args, idxs]
    global refreduce_type_exp = quote
        $(unpack_keyword_expr(idxs, :idxs))
        $(expand_to_calls(rr()))
    end
    setexpr!(exptree, refreduce_type_exp)
    # error("Tree: $exptree, id: $(genid())")
    global last_exptree[] = mergetree(last_exptree[], exptree)
    return refreduce_type_exp
end