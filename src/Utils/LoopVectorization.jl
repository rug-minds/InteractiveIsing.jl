@inline function column_contraction(i, v::AbstractVector{T}, sp::SparseMatrixCSC{T}; transform::F = identity) where {T, F}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * F(turbo_getindex(v, j))
    end
    return tot
end

# @generated function column_contraction(i::I, vectors_then_matrices...) where I<:Integer

# end