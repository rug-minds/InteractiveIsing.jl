@inline @generated function gettype(::Type{NT}, ::Val{field}) where {NT, field}
    idx = findfirst(x -> x == field, fieldnames(NT))
    if idx === nothing
        error("Field $field not found in type $NT, with fields $(fieldnames(NT))")
    end
    return :($(fieldtypes(NT)[idx]))
end

@inline gettype(nt, field::Symbol) = @inline gettype(nt, Val(field))
# @inline gettype(nt::Type, field::Symbol) = @inline gettype(nt, Val(field))


"""
Get consecutive types from structs with nested properties
"""
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



function _gettype_recursive(nt::Type, field::Symbol)
    for (i, fn) in enumerate(fieldnames(nt))
        if fn == field
            return fieldtypes(nt)[i]
        end
    end

    if !isempty(fieldtypes(nt))
        for ft in fieldtypes(nt)
            res = _gettype_recursive(ft, field)
            if res !== nothing
                return res
            end
        end
    end

    return nothing
end


"""
For nested structs, find the type of a field.
this will find the first occurence of the field, and return the type of that.
"""
gettype_recursive(nt::Type, field::Symbol) = gettype_recursive(nt, Val(field))
@inline @generated function gettype_recursive(::Type{NT}, ::Val{field}) where {NT, field}
    t = _gettype_recursive(NT, field)
    return :($t)
end


"""
Get consecutive types from structs with nested properties
"""
@inline function gettype_recursive(::Type{NT}, fields::Tuple) where NT
    if isempty(fields)
        return NT
    end
    return gettype_recursive(gettype_recursive(NT, first(fields)), Base.tail(fields))
end


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

Base.isempty(nt::Type{<:NamedTuple}) = isempty(fieldnames(nt))

# Base.keys(nt::NamedTuple) = fieldnames(typeof(nt))
Base.keys(ntt::Type{<:NamedTuple}) = fieldnames(ntt)
Base.keys(::Nothing) = tuple()
Base.keys(::Type{Nothing}) = tuple()