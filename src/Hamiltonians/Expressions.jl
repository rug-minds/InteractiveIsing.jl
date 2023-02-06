"""
Finds the expression that fits with the given symbol, value of the symbol,
and the factor type
"""
function findExpr(symb, val, type)
    for factor in factors
        if factor.symb == symb && factor.val == val && factor.type == type
            return factor.expr
        end
    end
    return ""
end

"""
Makes expression in form of string out of a set of symbols, values, 
and a predicate that tells the function wether the expression is part of
a loop over neighbors or not
"""
function buildExpr(type, symbs, vals)
    str = string()

    for (idx, symb) in enumerate(symbs)
        term = findExpr(symb, vals[idx], type)
        if term != ""
            if str != ""
                str *= " + "
            end
            # str *= ("(@inbounds " * term * ")")
            str *= term
        end
        
    end
    return str
end

buildExpr(type, htype::HType{Symbs,Vals}) where {Symbs, Vals} = buildExpr(type, Symbs, Vals)
buildExpr(type, htype::Type{HType{Symbs,Vals}}) where {Symbs, Vals} = buildExpr(type, Symbs, Vals)

export buildExpr