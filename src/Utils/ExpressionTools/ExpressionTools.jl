"""
Walk through an expression tree and return all parts of the expression where func evaluates to true
    Func gets head and args (head, args) -> Bool
"""
function expwalk(func, expr::Expr)
    exps = []
    _expwalk(func, expr, exps)
    return exps
end

function _expwalk(func, expr, exps)
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
            _expwalk(func, arg, exps)
        end
    end
end
export expwalk

"""
Walk through all symbols in an expression and replace them by the function func
"""
symbwalk!(func, expr::Expr) = _symbwalk!(func, deepcopy(expr))
function _symbwalk!(func, expr::Expr)
    for arg_idx in eachindex(expr.args)
        if expr.args[arg_idx] isa Expr
            _symbwalk!(func, expr.args[arg_idx])
        else
            expr.args[arg_idx] = func(expr.args[arg_idx])
        end
    end
    return expr
end