function deletekey(ps::Base.Pairs, key)
    _keys = collect(keys(ps))
    filter!(x -> x != key, _keys)
    if !isempty(_keys)
        return pairs(ps[_keys...])
    else
        pairs((;))
    end
end

function deletekeys(ps::Base.Pairs, ks...)
    newargs = filter(pair -> !(pair.first in ks), ps)
    return pairs((;newargs...))
end

"""
Merge 2 kwargs. If a key is present in both, the value from the top is used
"""
function mergekwargs(bottom, top)
    if isnothing(bottom) && isnothing(top)
        return pairs((;))
    elseif isnothing(top) || (!isnothing(bottom) && isempty(top))
        return bottom
    elseif isnothing(bottom) || (!isnothing(top) && isempty(bottom))
        return top
    end
    topkeys = collect(Any, keys(top))
    topvalues = collect(Any, values(top))
    for key in keys(bottom)
        if !(key in topkeys)
            push!(topkeys, key)
            push!(topvalues, bottom[key])
        end
    end
    return pairs((;(topkeys .=> topvalues)...))
end

"""
Replace kwargs in bottom that are also specified in top,
Keys in top that are not in bottom are ignored
"""
function replacekwargs(bottom, top)
    if isnothing(bottom) && isnothing(top)
        return pairs((;))
    #If top is empty, return bottom
    elseif isnothing(top) || isempty(top)
        return bottom
    end
    bottomkeys = collect(Any, keys(bottom))
    bottomvalues = collect(Any, values(bottom))
    for topkey in keys(top)
        if (idx = findfirst(x -> x == topkey, bottomkeys)) !== nothing
            bottomvalues[idx] = top[topkey]
        else
            println("Unsupported key $topkey ignored")
        end
    end
    return pairs((;(bottomkeys .=> bottomvalues)...))
end
export mergekwargs, replacekwargs