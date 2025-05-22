function index_names(nt::Union{Type{<:NamedTuple}, NamedTuple, Tuple})
    if nt isa NamedTuple
        return tuple(keys(nt)...)
    elseif nt isa Tuple
        return nt
    elseif nt <: NamedTuple
        return tuple(fieldnames(nt)...)
    else
        throw(ArgumentError("index_names only works on NamedTuples, Tuples and Nothings"))
    end
end
index_names(nt::Nothing) = nothing
### Macro putting in prefs
"""
Find symbols of the form s_ij...
"""
function is_paramref(symb)
    string = String(symb)
    found = findfirst(x -> x == '_', string)
    if !isnothing(found)
        return true
    end
    return false
end

"""
Find where there are symbols of the form s_ij...
"""
function find_paramref(ex)
    symbs = []
    indexes = []
    for (idx, subex) in enumerate(ex.args)
        _find_paramref(subex, (idx,), symbs, indexes)
    end
    return symbs, indexes
end

function _find_paramref(ex, this_idxs, symbs, indexes)
    if ex isa Symbol
        if is_paramref(ex)
            push!(symbs, ex)
            push!(indexes, this_idxs)
        end
    elseif ex isa Expr
        for (this_idx, arg) in enumerate(ex.args)
            _find_paramref(arg, (this_idxs..., this_idx), symbs, indexes)
        end
    end
end

"""
Substitute in a ParameterRef
"""
function substitute_paramref(ex, indexes, symb)
    replace_symb(ex, ParameterRef(symb), indexes)
end

ParameterRefs_ex = nothing
macro ParameterRefs(ex)
    if @capture(ex, function fname_(a__) body_ end)
        symbs = find_paramref(body)
        for (symb, indexes) in zip(symbs[1], symbs[2])
            body = substitute_paramref(body, indexes, symb)
        end
        ex = quote function $fname($(a...)) $body end end
        global ParameterRefs_ex = ex
        return esc(ex)
    else
        if ex isa Symbol
            ex = Expr(:block, ex)
        end
        symbs = find_paramref(ex)
        for (symb, indexes) in zip(symbs[1], symbs[2])
            ex = substitute_paramref(ex, indexes, symb)
        end
        global ParameterRefs_ex = ex
        return esc(ex)
    end 
end

var"@PR" = var"@ParameterRefs"

export @ParameterRefs, @PR


#####

# build_getproperty_chain(symbs::Tuple) = build_getfield_chain(first(symbs), Base.tail(symbs))
"""
From a base struct and then a list of symbols
    build the expression to get the properties, i.e.
    getproperty(getproperty((...(base_expr), symbs[1]...), symbs[end-1]) symbs[end])
"""
function build_getproperty_chain(base_expr, symbols::Tuple)
    if isempty(symbols)
        return base_expr
    end
    return build_getproperty_chain(:(getproperty($base_expr, $(QuoteNode(first(symbols))))), Base.tail(symbols))
end


##
function expr_F_wrap(exp, func::Function)
    if typof(func) == identity
        return exp
    end
    return Expr(:call, func, exp)
end