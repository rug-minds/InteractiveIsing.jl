function deletekey!(ps::Base.Pairs, key)
    _keys = collect(keys(ps))
    filter!(x -> x != key, _keys)
    return pairs(ps[_keys...])
end

function deletekeys!(ps::Base.Pairs, ks...)
    _keys = collect(keys(ps))
    filter!(x -> !(x in ks), _keys)
    return pairs(ps[_keys...])
end