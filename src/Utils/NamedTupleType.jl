function gettype(nt::Type{<:NamedTuple}, symb::Symbol)
    idx = findfirst(x -> x == symb, fieldnames(nt))
    return nt.types[idx]
end

function gettype(nt::Type{<:NamedTuple}, idx::Integer)
    return nt.types[idx]
end

gettype(nt::NamedTuple, symb::Symbol) = gettype(typeof(nt), symb)

"""
Get the val from a val type
"""
getval(::Type{Val{T}}) where T = T

function searchkey(nt, symb::Symbol; fallback)
    if haskey(nt, symb)
        return nt[symb]
    else
        return fallback
    end
end

function searchdeletekey(nt, symb::Symbol; fallback)
    if haskey(nt, symb)
        return nt[symb], delete!(nt, symb)
    else
        return fallback
    end
end