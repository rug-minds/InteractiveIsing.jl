struct ParameterRef{Symb, indexsymb} end

# function ::(ParameterRef{S, idxs})(params) where {S, idxs}
#     params[S]
# end

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

function ParameterRef(symb)
    sstr = String(symb)
    u_found = findfirst(x -> x == '_', sstr)
    get_symb = sstr[1:u_found-1]
    get_index = sstr[u_found+1:end]
    return ParameterRef{Symbol(get_symb), Symbol.(tuple(get_index...))}()
end

function find_paramref(ex)
    symbs = []
    indexes = []
    println("Looking at $ex")
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

function substitute_paramval(ex, indexes, symb)
    replace_symb(ex, ParameterRef(symb), indexes)
end

macro ParameterRefs(ex)
    symbs = find_paramref(ex)
    for (symb, indexes) in zip(symbs[1], symbs[2])
        ex = substitute_paramval(ex, indexes, symb)
    end
    println(ex)
end


d_ex = quote function Δi_H(::Ising)
        contractions = :(w_ij*s_j)
        multiplications = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i))
        return (;contractions, multiplications)
    end
end

export d_ex, @ParameterRefs
ev =  @ParameterRefs function ΔiH(::Ising)
    contractions = :(w_ij*s_j)
    multiplications = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i))
    return (;contractions, multiplications)
end
