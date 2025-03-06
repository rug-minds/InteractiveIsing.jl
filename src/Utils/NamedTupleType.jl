@generated function gettype(::Type{NT}, ::Val{field}) where {NT<:NamedTuple, field}
    idx = findfirst(==(field), fieldnames(NT))
    return :($(NT.types[idx]))
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