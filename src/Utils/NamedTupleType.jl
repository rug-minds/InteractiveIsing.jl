function gettype(nt::Type{<:NamedTuple}, symb::Symbol)
    idx = findfirst(x -> x == symb, fieldnames(nt))
    return nt.types[idx]
end

function gettype(nt::Type{<:NamedTuple}, idx::Integer)
    return nt.types[idx]
end

gettype(nt::NamedTuple, symb::Symbol) = gettype(typeof(nt), symb)

getval(::Type{Val{T}}) where T = T 