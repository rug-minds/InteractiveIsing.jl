
"""
Wrap an expression in a turbo for loop
"""
function wrap_turbo_collect(exp, iterator, symb)
    turboexp = quote
        @turbo for $(symb) in ($(iterator))
            $exp
        end
    end
    
    # interpolate!(turboexp, exp) |> remove_line_number_nodes
end


#### LOOPS
"""
Wrap an expression in a for loop
"""
function for_wrap(exp, iterator, symb)
    expr = quote
        for $symb in $iterator
            $exp
        end
    end
end

"""
Wrap an exp that should be reduced in nested loops
in a turbo nested loop
"""
function nested_turbo_wrap(exp, iterators, symbs)
    for i in eachindex(symbs)
        if i == length(iterators)
            exp = wrap_turbo_collect(exp, iterators[i], symbs[i])
        else
            exp = for_wrap(exp, iterators[i], symbs[i])
        end
    end
    return exp
end

function nested_for_wrap(exp, iterators, symbs)
    for i in eachindex(symbs)
            exp = for_wrap(exp, iterators[i], symbs[i])
    end
    return exp
end