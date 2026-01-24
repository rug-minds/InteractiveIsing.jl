@inline function tuple_searchsortedlast(t::Tuple, x)
    lo = 1
    hi = length(t)
    while lo <= hi
        mid = (lo + hi) >>> 1
        if t[mid] <= x
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return hi
end

@inline function tuple_searchsortedfirst(t::Tuple, x)
    lo = 1
    hi = length(t)
    while lo <= hi
        mid = (lo + hi) >>> 1
        if t[mid] < x
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return lo
end
