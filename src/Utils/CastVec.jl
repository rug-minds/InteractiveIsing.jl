struct CastVec{In,Out} <: AbstractVector{Out}
    data::Vector{In}
end
Base.getindex(v::CastVec{In,Out}, i) where {In,Out} = Out(v.data[i])
Base.setindex!(v::CastVec{In,Out}, val, i) where {In,Out} = v.data[i] = In(val)
Base.length(v::CastVec) = length(v.data)
Base.size(v::CastVec) = size(v.data)
Base.eltype(v::CastVec) = eltype(v.data)
Base.IteratorSize(::Type{CastVec}) = Base.HasLength()
Base.iterate(v::CastVec, i=1) = i > length(v) ? nothing : (v[i], i+1)
CastVec(t::Type, data) = CastVec{eltype(data), t}(data)