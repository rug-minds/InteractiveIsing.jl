struct ExprIdx{T<:Tuple}
    idxs::T
end

ExprIdx(idxs...) = ExprIdx(idxs)
ExprIdx(eidxs::ExprIdx, idxs...) = ExprIdx(eidxs.idxs..., idxs...)

Base.length(idxs::ExprIdx) = length(idxs.idxs)
Base.lastindex(idxs::ExprIdx) = length(idxs)
Base.getindex(idxs::ExprIdx, i) = idxs.idxs[i]


gethead(eidxs::ExprIdx) = eidxs.idxs[1]
gettail(eidxs::ExprIdx) = ExprIdx(eidxs.idxs[2:end])


get_last_exp(e::Expr, idxs::ExprIdx) = get_last_exp(e.args[gethead(idxs)], gettail(idxs))
get_last_exp(e::Expr, idxs::ExprIdx{Tuple{Int}}) = e

get_lastargs(e::Expr, idxs::ExprIdx) = get_lastargs(e.args[gethead(idxs)], gettail(idxs))
get_lastargs(e::Expr, idxs::ExprIdx{Tuple{Int}}) = e.args

Base.getindex(e::Expr, idxs::ExprIdx) = getindex(e.args[gethead(idxs)], gettail(idxs))
Base.getindex(e::Expr, idxs::ExprIdx{Tuple{Int}}) = getindex(e.args, gethead(idxs))

Base.setindex!(e::Expr, val, idxs::ExprIdx) = get_lastargs(e, idxs)[idxs[end]] = val

"""
Walk through an expression tree and return all parts of the expression where func evaluates to true
    Func gets head and args (head, args) -> Bool
"""
function expmatch(func, expr::Expr)
    exps = []
    _expmatch(func, expr, exps)
    return exps
end

function _expmatch(func, expr, exps)
    if expr isa Expr
        if func(expr.head, expr.args)
            push!(exps, expr)
        end
    else
        if func(typeof(expr), expr)
            push!(exps, expr)
        end
    end

    if expr isa Expr
        for arg in expr.args
            _expmatch(func, arg, exps)
        end
    end
end
export expmatch

"""
Walk through an expression tree and return references to args and an idx (args, idx) where func evaluates to true
    Func gets head and arg -> Bool
"""
function argmatch(func, expr::Expr)
    matches = []
    _argmatch(func, expr, matches)
    return matches
end

function _argmatch(func, expr, matches)
    for (arg_idx, arg) in enumerate(expr.args)
        if func(arg)
            push!(matches, (expr.args, arg_idx))
        end
        if arg isa Expr
            _argmatch(func, arg, matches)
        end
    end
end


"""
Walk through all symbols in an expression and replace them by the function func
"""
function symbwalk!(func, expr::Expr)
    _symbwalk!(func, expr)
end

function symbwalk(func, expr::Expr)
    copyexp = deepcopy(expr)
    _symbwalk!(func, copyexp)
    return copyexp
end
function _symbwalk!(func, expr::Expr)
    for arg_idx in eachindex(expr.args)
        if expr.args[arg_idx] isa Expr
            _symbwalk!(func, expr.args[arg_idx])
        else
            returnexp = func(expr.args[arg_idx])
            expr.args[arg_idx] = returnexp
        end
    end
    return expr
end

"""
Walk through the expression tree but only apply the function to the expression nodes
"""
function expwalk!(func, expr::Expr)
    _expwalk!(func, expr)
end

function expwalk(func, expr::Expr)
    copyexp = deepcopy(expr)
    _expwalk(func, copyexp)
    return copyexp
end

function _expwalk!(func, expr::Expr)
    for arg_idx in eachindex(expr.args)
        if expr.args[arg_idx] isa Expr
            _expwalk!(func, expr.args[arg_idx])

            returnexp = func(expr.args[arg_idx])
            expr.args[arg_idx] = returnexp
        end
    end
    return
end
export expwalk

function get_function_line(exp)
    expmatch((head, args) -> head == :function, exp)[1].args[1]
end
export get_function_line

"""
Remove all the linenumbernodes
"""
function printexp(exp)
    copyexp = deepcopy(exp)
    _printexp(copyexp)
    println(copyexp)
    return nothing
end

function _printexp(exp)
    if exp isa Expr
        for a_idx in reverse(eachindex(exp.args))
            if exp.args[a_idx] isa LineNumberNode
                deleteat!(exp.args, a_idx)
            elseif exp.args[a_idx] isa Expr
                _printexp(exp.args[a_idx])
            end
        end
    end
    return nothing
end
export printexp

"""
Remove args from the total expression tree where the function f evaluates to true
"""
function remove_args(f, exp)
    expr = deepcopy(exp)
    _remove_args!(f, expr)
    return expr
end

"""
Remove args from the total expression tree where the function f evaluates to true
"""
function remove_args!(f, exp)
    for arg_idx in reverse(eachindex(exp.args))
        if exp.args[arg_idx] isa Expr
            remove_args!(f, exp.args[arg_idx])
        elseif f(exp.args[arg_idx])
            deleteat!(exp.args, arg_idx)
        end
    end
    return exp
end

"""
Remove all LineNumberNodes from the expression
"""
function remove_line_number_nodes(exp)
    expr = deepcopy(exp)
    remove_line_number_nodes!(expr)
    return expr
end

rlnn(x) = remove_line_number_nodes(x)
export rlnn

# """
# Remove all LineNumberNodes from the expression
# """
# function remove_line_number_nodes!(exp)
#     f = (x) -> x isa LineNumberNode
#     remove_args!(f, exp)
#     return expq
# end
ismacrosymbol(s::Symbol) = startswith(string(s), "@")

"""
Remove all LineNumberNodes from the expression
"""
function remove_line_number_nodes!(exp)
    if !(exp isa Expr)
        return
    end
    if exp.head == :macrocall && ismacrosymbol(exp.args[1]) && exp.args[1] != Symbol("@\$") # Special handling for @$
        filter!(arg -> !(arg isa LineNumberNode), @view exp.args[3:end])
        remove_line_number_nodes!.(exp.args[3:end])

    else
        filter!(arg -> !(arg isa LineNumberNode), exp.args)
        remove_line_number_nodes!.(exp.args)
    end
end

"""
Concatenate expressions
"""
function expcat(exps...)
    e = Expr(:block)
    for exp in exps
        if exp isa Expr && exp.head == :block
            for arg in exp.args
                push!(e.args, arg)
            end
        else
            push!(e.args, exp)
        end
    end
    remove_line_number_nodes!(e)
    return e
end

function findall_in_exp(func, exp)
    found = Any[]
    for (arg_idx, arg) in enumerate(exp.args)
        _findall_exp(func, arg, found, arg_idx)
    end
    return found
end

function _findall_exp(func, exp, found, level_idxs...)
    if func(exp)
        push!(found, level_idxs)
    end
    if exp isa Expr
        for (arg_idx, arg) in enumerate(exp.args)
            _findall_exp(func, arg, found, level_idxs..., arg_idx)
        end
    end
end

"""
Replace the args of an expression at level idxs with replace
"""
function replace_args!(exp, idxs, replace)
    enter_args(exp, idxs[1:end-1]).args[idxs[end]] = replace
end

function clean_interpolate_symb!(exp)
    if !isa(exp, Expr)
        return
    else
        if exp.head == :macrocall && ismacrosymbol(exp.args[1])
            filter!(arg -> !(arg isa LineNumberNode), exp.args)
            remove_line_number_nodes!.(exp.args)
        end
    end
end

function find_interpolate_symb(exp, idxs::ExprIdx = ExprIdx(); tag = nothing)
    
    if exp isa Expr
        if exp.head == :macrocall
            if exp.args[1] == Symbol("@\$")
                if isnothing(tag)
                    return tuple(idxs)
                else
                    if any(x -> x == tag, exp.args[2:end])
                        return tuple(idxs)
                else
                    return nothing
                end
            end 
        end
    end

        returns = (find_interpolate_symb(exp.args[arg_idx], ExprIdx(idxs, arg_idx)) for arg_idx in eachindex(exp.args))
        t = tuple(Iterators.flatten(filter(x -> !isnothing(x), collect(returns)))...)
        return t
    end
    return nothing
end

function interpolate_symb_tag(exp, idxs::ExprIdx)
    # as = get_lastargs(exp, idxs)
    # println("Interpolate symb tag")
    # println("exp: ", exp)
    # println("Typeof exp: ", typeof(exp))
    # println("exp.head: ", exp.head)
    # println("exp.args: ", exp.args)
    # println("idxs: ", idxs)
    e = exp[idxs]
    as = e.args
    if length(as) <= 2
        return Symbol()
    else
        return as[3]
    end
end

inner_tag(x) = x isa Pair ? x.first : Symbol()

"""
Expressions with @\$ will be replaced by the inner expressions
    without wrapping the interpolated exp in a block
"""
function interpolate!(outer, inners...)
    clean_interpolate_symb!(outer)
    remove_line_number_nodes!.(inners)
    # Find @$
    # found = findall_in_exp(x -> x == Symbol("@\$"), outer)

    found = [find_interpolate_symb(outer)...]
    if found[1] == ExprIdx() # If it's the top, just replace it with the first inner
        return inners[1]
    end

    inners = sort(inners, by = inner_tag)
    inner_tags = [inner_tag(x) for x in inners]

    outer_tags = [interpolate_symb_tag(outer, idxs) for idxs in found]
    # println("outer_tags: ", outer_tags)
    
    # sort!(outer_tags)
    # println("Found: ",found)

    # @assert !(length(found) < length(inners)) "Not enough @\$ to interpolate"
    # @assert !(length(found) > length(inners)) "Too many @\$ to interpolate"

    # for (idx, inner) in reverse(collect(enumerate(inners)))
    #     replace_args!(outer, found[idx][1:end-1], inner)
    # end
    for (inner_i, itag) in enumerate(inner_tags)
        outer_tag_idx = findfirst(x -> x == itag, outer_tags)
        if !isnothing(outer_tag_idx)
            idxs = found[outer_tag_idx]
            inner_exp = inners[inner_i]
            if inner_exp isa Expr && inner_exp.head == :block && length(inner_exp.args) == 1 
                #if block around a single other exp, remove the outer block
                inner_exp = inner_exp.args[1]
            end
            outer[idxs] = inners[inner_i]
            deleteat!(found, outer_tag_idx)
        end
    end

    return outer
end

interpolate(outer, inners...) = interpolate!(deepcopy(outer), inners...)
export interpolate, interpolate!