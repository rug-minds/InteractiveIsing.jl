@inline @generated function gettype(::Type{NT}, ::Val{field}) where {NT, field}
    idx = findfirst(x -> x == field, fieldnames(NT))
    if idx === nothing
        error("Field $field not found in type $NT, with fields $(fieldnames(NT))")
    end
    return :($(fieldtypes(NT)[idx]))
end

@inline gettype(nt, field::Symbol) = @inline gettype(nt, Val(field))

@inline function gettype(::Type{NT}, fields::Tuple) where NT
    if isempty(fields)
        return NT
    end
    
    return gettype(gettype(NT, first(fields)), Base.tail(fields))
end

# gettype(nt::Type, s::Val{s}) where s = gettype(nt, Val(s))

function gettype(nt::Type{<:NamedTuple}, idx::Integer)
    return nt.types[idx]
end

gettype(nt::NamedTuple, symb::Symbol) = gettype(typeof(nt), symb)


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