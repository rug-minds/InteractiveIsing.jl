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

# """
# Remove all LineNumberNodes from the expression
# """
# function remove_line_number_nodes!(exp)
#     f = (x) -> x isa LineNumberNode
#     remove_args!(f, exp)
#     return exp
# end
"""
Remove all LineNumberNodes from the expression
"""
function remove_line_number_nodes!(exp)
    if !(exp isa Expr)
        return exp
    end
    
    # Special handling for @turbo macro
    if exp.head == :macrocall && exp.args[1] == Symbol("@turbo")
        turbo_args = exp.args[3]  # Save the for loop
        return Expr(:macrocall, Symbol("@turbo"), nothing, remove_line_number_nodes!(turbo_args))
    end
    
    # Handle regular block expressions
    if exp.head == :block
        new_args = filter(arg -> begin
            if arg isa LineNumberNode
                return false
            elseif arg isa Expr
                return true
            else
                return true
            end
        end, exp.args)
        
        # Process nested expressions
        for (i, arg) in enumerate(new_args)
            if arg isa Expr
                new_args[i] = remove_line_number_nodes!(arg)
            end
        end
        
        exp.args = new_args
    else
        # Handle other expression types
        for (i, arg) in enumerate(exp.args)
            if arg isa Expr
                exp.args[i] = remove_line_number_nodes!(arg)
            end
        end
    end
    
    return exp
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


# This doesn't work like this
function interpolate!(outer, inners...)
    remove_line_number_nodes!(outer)
    # Find @$
    found = argmatch(x -> x == Symbol("@\$"), outer)
    @assert !(length(found) < length(inners)) "Not enough @\$ to interpolate"
    @assert !(length(found) > length(inners)) "Too many @\$ to interpolate"

    for (idx, inner) in enumerate(inners)
        # Replace @$ with inner
        if inner isa Expr && inner.head == :block
            for innerargs in inner.args
                splice!(found[idx][1], found[idx][2]:found[idx][2]-1, [innerargs])
            end
            # Delete the macrosymbol
            deleteat!(found[idx][1], found[idx][2])
            # change head from macrocall to block
            found[idx][1][found[idx][2]].head = :block
        else
            splice!(found[idx][1], found[idx][2]:found[idx][2], [inner])
            found[idx][1][found[idx][2]].head = :block
        end
    end

    return outer
end

interpolate(outer, inners...) = interpolate!(deepcopy(outer), inners...)
export interpolate, interpolate!