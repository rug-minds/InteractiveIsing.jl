"""
Gets the argument names for a method
"""
function method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

"""
Gets the keywords args for a macro
From the args given
"""
function prunekwargs(args...)
    @nospecialize
    firstarg = first(args)
    if isa(firstarg, Expr) && firstarg.head == :parameters
        return prunekwargs(BenchmarkTools.drop(args, 1)..., firstarg.args...)
    else
        params = collect(args)
        for ex in params
            if isa(ex, Expr) && ex.head == :(=)
                ex.head = :kw
            end
        end
        return params
    end
end