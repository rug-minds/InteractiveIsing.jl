function tuple_type_property(propf, t::Type{T}) where T<:Tuple
    _tuple_type_property(propf, (propf(typehead(t)),), typetail(t))
end

tuple_type_property(propf, t::Tuple) = tuple_type_property(propf, typeof(t))

function _tuple_type_property(propf, acc, tail)
    if tail == Tuple{}
        return acc
    else
        return _tuple_type_property(propf, (acc..., propf(typehead(tail))), typetail(tail))
    end
end

@inline function typehead(t::Type{T}) where T<:Tuple
    Base.tuple_type_head(T)
end

@inline typehead(::Type{Tuple{}}) = nothing

@inline function typeheadval(t::Type{T}) where T<:Tuple
    Val(typehead(t))
end

@inline typeheadval(::Type{Tuple{}}) = nothing

@inline function typetail(t::Type{T}) where T<:Tuple
    Base.tuple_type_tail(T)
end

@inline typetail(t::Type{Tuple{}}) = nothing

@inline function headval(t::Tuple)
    Val(Base.first(t))
end

@inline headval(::Tuple{}) = nothing

@inline gethead(t::Tuple) = Base.first(t)
@inline gethead(::Tuple{}) = nothing

@inline gettail(t::Tuple) = Base.tail(t)
@inline gettail(::Tuple{}) = nothing


getval(::Val{V}) where V = V
getval(::Type{Val{V}}) where V = V
