using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase

using VectorizationBase: AbstractSIMD, VecUnroll, vload
const _SAI = LayoutPointers.StaticArrayInterface

@inline function column_contraction(i, v::AbstractArray{T}, sp::SparseMatrixCSC{T}; transform::F = identity, transform_weight::FW = identity) where {T, F, FW}
    tot = zero(T)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @turbo for ptr in nzrange(sp, i)
        j = rowval[ptr]
        wij = nzval[ptr]
        val = getindex(v, j)
        transformed = transform(val)
        transformed_weight = transform_weight(wij)
        tot += transformed_weight * transformed
    end
    return tot
end

@inline function column_contraction_t(i, v::AbstractArray{T}, sp::SparseMatrixCSC{T}; transform::F = identity, transform_weight::FW = identity) where {T, F, FW}
    tot = zero(T)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @turbo for ptr in nzrange(sp, i)
        j = rowval[ptr]
        wij = nzval[ptr]
        # val = getindex(v, j)
        val = 2f0
        transformed = transform(val)
        transformed_weight = transform_weight(wij)
        tot += transformed_weight * transformed
    end
    return tot
end



# @generated function column_contraction(i::I, vectors_then_matrices...) where I<:Integer

# end