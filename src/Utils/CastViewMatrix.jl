struct CastViewMatrix{In, Out} <: AbstractMatrix{Out}
    data::Vector{In}
    idxs::UnitRange{Int}
    size::Tuple{Int,Int}
end

Base.getindex(m::CastViewMatrix{In,Out}, i::Integer) where {In,Out} = Out(m.data[m.idxs[i]])
Base.getindex(m::CastViewMatrix{In,Out}, other) where {In,Out} = Out.(m.data[m.idxs[other]])

Base.getindex(m::CastViewMatrix{In,Out}, i, j) where {In,Out} = Out(m.data[m.idxs[j+(i-1)*size(m,2)]])
Base.setindex!(m::CastViewMatrix{In,Out}, val, i, j) where {In,Out} = m.data[m.idxs[j+(i-1)*size(m,2)]] = In(val)
Base.size(m::CastViewMatrix) = m.size
Base.size(m::CastViewMatrix, i) = m.size[i]
Base.length(m::CastViewMatrix) = length(m.idxs)
Base.eltype(m::CastViewMatrix{In, Out}) where {In,Out} = Out
Base.IteratorSize(::Type{CastViewMatrix}) = Base.HasLength()
Base.iterate(m::CastViewMatrix, i=1) = i > length(m) ? nothing : (m[i], i+1)
CastViewMatrix(t::DataType, data, range ,len, wid) = CastViewMatrix{eltype(data), t}(data, range, (len, wid))