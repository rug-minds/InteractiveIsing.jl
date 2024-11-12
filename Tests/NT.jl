function get_field_type(::Type{T}, field::Val{S}) where {T <: NamedTuple,S}
    names = fieldnames(T)
    types = fieldtypes(T)
    
    index = findfirst(==(S), names)
    
    return index === nothing ? throw(ArgumentError("Field $field not found")) : types[index]
end

function get_field_type(@specialize(nt::NamedTuple), field)
    return get_field_type(typeof(nt), field)
end

get_field_type(nt, field::Symbol) = get_field_type(nt, Val{field}())

function nttest(a, b , @specialize(nt))
    if get_field_type(nt, :a) == Int64
        return a+b
    elseif get_field_type(nt, :a) == Float64
        return a*b
    end
end
function test1()
    nt = (;a = 1)
    @code_warntype nttest(1, 2, nt)
    # nttest(1, 2, nt)
end

function test2()
    nt = (;a = 1, b = 2)
    @code_warntype nttest(1, 2, nt)
end
