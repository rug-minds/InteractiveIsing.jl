export SparseVal
"""
Hacky way to define a ref that can be indexed but gives a constant value
"""
mutable struct SparseVal{T, IT <: Integer} <: AbstractSparseArray{T, IT, 1}
    Val::T
    idx::IT
    const len::IT
end

function Base.show(io::IO, ::MIME"text/plain", sp::SparseVal)
    println(io, "$(sp.len)-element SparseVal{$(eltype(sp))}")
    println(io, "[$(sp.idx)] = $(sp.Val) ($(sp.len))")
end


@inline Base.ndims(::SparseVal) = 1
@inline Base.size(::SparseVal) = (1,)
@inline Base.length(sp::SparseVal) = sp.len
@inline Base.getindex(sp::SparseVal{T}, i::Integer) where T = i == sp.idx ? sp.Val : zero(T)
@inline Base.getindex(sp::SparseVal{T}) where T = sp.Val
@inline Base.setindex!(sp::SparseVal{T}, v, i::Integer) where T = begin sp.Val = v; sp.idx = i end
@inline Base.setindex!(sp::SparseVal{T}, v::T) where T = sp.Val = v

@inline Base.axes(sp::SparseVal) = (Base.OneTo(sp.len),)
@inline Base.axes(sp::SparseVal, i) = axes(sp::SparseVal)[i]
@inline Base.eltype(::SparseVal{T}) where T = T
@inline Base.convert(::Type{T}, n::SparseVal) where T = T(n.Val)
@inline Base.:-(n1::SparseVal, n::T) where T = T(n1.Val - n)
@inline Base.:-(n::T, n1::SparseVal) where T = T(n - n1.Val)
@inline Base.reduce(::typeof(+), sp::SparseVal) = sp.Val
@inline SparseArrays.rowvals(sp::SparseVal) = sp.idx
@inline SparseArrays.nonzeros(sp::SparseVal) = sp.Val
@inline SparseArrays.nzrange(sp::SparseVal, i::Integer = 1) = sp.idx
@inline SparseArrays.nnz(sp::SparseVal) = 1
@inline nzval(sp::SparseVal) = sp.Val
@inline SparseArrays.isassigned(sp::SparseVal, i::Integer) = i == sp.idx

@inline loopconstant(::SparseVal) = true
@inline loopconstant(::Type{<:SparseVal}) = true
@inline unroll_exp(::Union{Type{<:SparseVal}, SparseVal}, vecname, exp_f = identity) = :($(exp_f(:($(vecname)[]))))



@inline Base.:+(n1::SparseVal, n::T) where T = T(n1.Val + n)
@inline Base.:+(n::T, n1::SparseVal) where T = T(n + n1.Val)

@inline Base.:*(n::T, n1::SparseVal) where T = SparseVal(n*n1.Val)
@inline Base.:*(n1::SparseVal, n::T) where T = SparseVal(n*n1.Val)