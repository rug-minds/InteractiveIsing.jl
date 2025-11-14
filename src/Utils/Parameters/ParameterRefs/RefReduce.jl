### Reduce
"""
Combines object with the same indices into a single object
These are objects that are added or subtracted
"""
struct RefReduce{Refs, reduce_fs, F, D} <: AbstractParameterRef 
    data::D
end

RefReduce{A,B,C,D}() where {A,B,C,D} = RefReduce{A,B,C,Nothing}(nothing)

RefReduce(refs::Tuple, reduce_fs, data = nothing; func = identity) = RefReduce{refs, reduce_fs, func ,typeof(data)}(data)
get_prefs(rr::RefReduce{Refs}) where Refs = (Refs)
get_reduce_fs(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = reduce_fs
get_fs(rr::RefReduce) = get_reduce_fs(rr)

getF(rr::RefReduce{R,rf,F}) where {R,rf,F} = F
setF(rr::RefReduce{Refs, reduce_fs, F}, func) where {Refs, reduce_fs, F} = RefReduce(Refs, reduce_fs, rr.data, func = func)

Base.getindex(rr::RefReduce, idx) = get_prefs(rr)[idx]
Base.lastindex(rr::RefReduce) = length(get_prefs(rr))

function return_type(rr::RefReduce{Refs, reduce_fs, F, D}, args) where {Refs, reduce_fs, F, D}
    promote_type(return_type.(Refs, Ref(args))...)
end

Base.length(rr::RefReduce) = length(get_prefs(rr))

Base.:+(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefReduce((p1, p2), (+,))
Base.:-(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefReduce((p1, p2), (-,))

# Already a ref reduce
# Plusses come first
function Base.:+(p1::AbstractParameterRef, p2::RefReduce)
    if getF(p2) == identity
        return RefReduce((p1, get_prefs(p2)...), (+, get_reduce_fs(p2)...))
    else # If there's a function, then keep it blocked
        return RefReduce((p2, p1), tuple(+))
    end
end
Base.:+(p1::RefReduce, p2::AbstractParameterRef) = (+)(p2, p1)

function Base.:-(p1::RefReduce, p2::AbstractParameterRef)
    if getF(p1) == identity 
        return RefReduce((get_prefs(p1)..., p2), (get_reduce_fs(p1)..., -))
    else # If there's a function, then keep it blocked
        return RefReduce((p1, p2), tuple(-))
    end
end
Base.:-(p1::AbstractParameterRef, p2::RefReduce) = (-)(p2, p1)


# Base.:+(p1::RefReduce, p2::AbstractParameterRef) = RefReduce((p2, get_prefs(p1)...), (+, get_reduce_fs(p1)...))

Base.:-(p::RefReduce) = RefReduce(get_prefs(p), get_reduce_fs(p), p.data, func = -)
# Base.:^(p::RefReduce, pow::Real) = RefReduce(get_prefs(p), get_reduce_fs(p), p.data, func = (^, pow))

function num_plusses(rr::RefReduce)
    found = findfirst(get_reduce_fs(rr) .== -)
    isnothing(found) && (found = length(get_reduce_fs(rr)) + 1)
    found
end

function num_minuses(rr::RefReduce)
    found = findfirst(get_reduce_fs(rr) .== -)
    isnothing(found) && (found = length(get_reduce_fs(rr)) + 1)
    length(get_reduce_fs(rr)) - found + 1
end

function get_plusses(rr::RefReduce)
    tuple(rr[1:num_plusses(rr)]...)
end

function get_minuses(rr::RefReduce)
    tuple(rr[num_plusses(rr)+1:end]...) 
end

_ref_indices(rr::RefReduce) = tuple(union(_ref_indices.(_get_prefs(rr))...)...)

issparse(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = all(issparse.(Refs))

function _ispure(rr::RefReduce)
    pure = true
    all_indices_flattened = Set(Iterators.flatten(_ref_indices.(_get_prefs(rr))))
    for ref in _get_prefs(rr)
        idxs = _ref_indices(ref)
        pure = _ispure(ref) && isempty(setdiff(all_indices_flattened, idxs)) #Underlying is pure, and all have same indices
        if !pure
            break
        end
    end
    return pure
end

"""
Only consists of simple refs
"""

### EXPRESSION STUFF

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

