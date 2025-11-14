"""
Take (exp1, exp2, ...) and (op1, op2, ...)
return op_end(ex_end-1, ex_end) |> op_end-1(ex_end-2, ans) |> etc
"""
function operator_reduce_exp(exps, ops)
    wrapf = (exp1, exp2, op) -> Expr(:call, op, exp1, exp2)
    # start from the right and recursively wrap the exps in the calls
    exp = wrapf(exps[end-1], exps[end], ops[end])
    for i in length(exps)-2:-1:1
        exp = wrapf(exps[i], exp, ops[i])
    end
    return exp
end

function operator_reduce_exp_single(exps, ops)
    return Expr(:call, :+, exps[1], (ops[i-1] == (+) ? exps[i] : Expr(:call, :-, exps[i]) for i in 2:length(exps))...)
end

"""
Get the expression that unpacks the keyword arguments with var name `name`
    (;k1, k2, ...) = name
"""
function unpack_keyword_expr(kwtuple::NamedTuple, name::Symbol)
    if isempty(kwtuple)
        return Expr(:block)
    end
    return :((;$(keys(kwtuple)...)) = $name)
end
"""
For a keyword tuple type get the expression that unpacks the keyword arguments
    (;k1, k2, ...) = name
"""
function unpack_keyword_expr(kwtuple::Type{<:NamedTuple}, name::Symbol)
    if isempty(fieldnames(kwtuple))
        return Expr(:block)
    end
    keys = fieldnames(kwtuple)
    return :((;$(keys...)) = $name)
end

function unpack_keyword_expr(keys::Tuple, name::Symbol)
    if isempty(keys)
        return Expr(:block)
    end
    return :((;$(keys...)) = $name)
end

function vec_assignments_exp(refs...)
    total_exp = Expr(:block)
    for (refi,ref) in enumerate(refs)
        assign_exp = Expr(:(=), Symbol(:vec, refi), ref)
        push!(total_exp.args, assign_exp)
    end
    return total_exp
end



### NEW

function zero_assignment(args...; precision = nothing)
    if !isnothing_generated(precision)
        return :($precision(0))
    else
        return :(zero(promote_eltype($(args...))))
    end
end

function isloopconstant(apr::AbstractParameterRef, args, filled_idxs = nothing)
    lidxs = loop_idxs(apr, filled_idxs)
    return isempty(lidxs) || loopconstant(dereftype(apr, args))
end

struct AssignmentExprPrepare{PR}
    pref::PR
    name::Symbol
    expr::Expr
end

pref(aep::AssignmentExprPrepare) = aep.pref
name(aep::AssignmentExprPrepare) = aep.name
getexpr(aep::AssignmentExprPrepare) = aep.expr



Base.iterate(a::AssignmentExprPrepare, state = 1) = state == 2 ? nothing : (a, state+1)

function get_assignments(apr::AbstractParameterRef, args, filled_idxs = nothing)
    if isa(apr, ParameterRef)
        lidxs = loop_idxs(apr, filled_idxs)
        if loopconstant(dereftype(apr, args)) # If it's value-like then it was already assigned
            return AssignmentExprPrepare(apr, gensym(ref_symb(apr)), Expr(:call, :getindex, struct_ref_exp(apr)...))
        elseif isempty(lidxs) # If there's no loop idxs, it can safely be assigned
            return AssignmentExprPrepare(apr, gensym(ref_symb(apr)), Expr(:call, :getindex, struct_ref_exp(apr)..., ref_indices(apr)...))
        else 
            return AssignmentExprPrepare(apr, gensym(ref_symb(apr)), struct_ref_exp(apr)...)
        end
    end
    prefs = get_prefs(apr)
    # return tuple(get_assignments.(prefs)...)
    return tuple(Iterators.Flatten(get_assignments.(prefs, Ref(args), Ref(filled_idxs)))...)
end

function value_exp(apr::AbstractParameterRef, args, filled_idxs = nothing)
    debugmode = args isa NamedTuple
    debugmode && println("Value_exp for ", apr)
    if isa(apr, ParameterRef)
        lidxs = loop_idxs(apr, filled_idxs)
        debugmode && println("\tLidxs: ", lidxs)
        if isloopconstant(apr, args, filled_idxs) # If it's value-like then it was already assigned
            debugmode && println("\tLoop constant")
            return wrapF(apr, itag())
        else
            debugmode && println("\tNot loop constant")
            return wrapF(apr, Expr(:ref, itag(), ref_indices(apr)...))
        end
    end
    if isa(apr, RefMult)
        return wrapF(apr, Expr(:call, nameof(get_mult_f(apr)), (value_exp(p, args, filled_idxs) for p in get_prefs(apr))...))
    end
    
    if isa(apr, RefReduce)
        fs = get_reduce_fs(apr)
        return wrapF(apr, Expr(:call, :+, value_exp(get_prefs(apr)[1], args, filled_idxs),
        (   fs[i] == :+ ?
            value_exp(get_prefs(apr)[i+1], args, filled_idxs) :
            Expr(:call, :-, value_exp(get_prefs(apr)[i+1], args, filled_idxs))
            for i in 1:length(get_prefs(apr))-1)...))
    end
end

using OrderedCollections

function assignment_map(apr::AbstractParameterRef, args, filled_idxs = nothing)
    flat_prefs = flatprefs(apr)
    assignments = get_assignments(apr, args, filled_idxs)
    # symbs = full_symb.(flat_prefs)
    bare_refs = remF.(flat_prefs)
    
    # d = Dict{ParameterRef, Tuple}(k => v for (k,v) in zip(flat_prefs, assignments))
    # nt = (;zip(symbs, assignments)...)
    d = OrderedDict(zip(bare_refs, assignments))
    return d
end

function keyselect(od, keys::Union{NTuple{N, K}, Vector{K}}) where {K, N}
    return OrderedDict(zip(keys, getindex.(Ref(od), keys)))
    # return nt[keys]
end

get_assignment(od, ref) = od[remF(ref)] |> getexpr
# get_assignment(t, refsymb::Symbol) = t[refsymb] |> getexpr
get_name(od, ref) = od[ref] |> name
# get_name(t, refsymb::Symbol) = t[refsymb] |> name
# get_assignment_exp(t, refsymb::Symbol) = Expr(:(=), refsymb, get_assignment(t, refsymb)[2])
get_names(od) = name.(values(od))
# get_names(t) = name.(values(t)) 

# submap(t, ref::AbstractParameterRef) = keyselect(t, full_symb.(flatprefs(ref)))
submap(od::OrderedDict, ref::AbstractParameterRef) = keyselect(od, remF.(flatprefs(ref)))
# first_veclike_name(t, args, filled_idxs) = name(values(t)[findfirst(p -> !isloopconstant(p, args, filled_idxs), pref.(values(t)))])
first_veclike_name(od, args, filled_idxs) = name(values(od)[findfirst(p -> !isloopconstant(p, args, filled_idxs), pref.(values(od)))])

# function get_axes_exp(t, args, filled_idxs, index_symb = nothing)
#     veclike_name = first_veclike_name(t, args, filled_idxs)
#     return Expr(:call, :axes, veclike_name, 1)
# end

function get_axes_exp(od, args, filled_idxs, index_symb = nothing)
    veclike_name = first_veclike_name(od, args, filled_idxs)
    return Expr(:call, :axes, veclike_name, 1)
end

function unique_assignments(od)
    [Expr(:(=), get_name(od, ref), get_assignment(od, ref)) for ref in keys(od)]
end

itag(tag = nothing) = remove_line_number_nodes(:(@$($tag)))


