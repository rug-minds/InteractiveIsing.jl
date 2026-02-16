using LoopVectorization
using Tullio
using SparseArrays
using BenchmarkTools
using Random
using Finch
adj = g.adj
state = g.state
rng = Random.MersenneTwister()
state .= rand(rng, size(state)...)

function normal_nzrange(state::S, sparse_adj, i) where S
    s = zero(eltype(state))
    
    @inbounds for ptr in nzrange(sparse_adj, i)
        j = sparse_adj.rowval[ptr]
        wij = sparse_adj.nzval[ptr]
        s += wij*state[j]
    end
    return s
end

function normal_view(state::S, sparse_adj, i) where S
    s = zero(eltype(state))
    row = @view sparse_adj[i, :]
    @fastmath @simd for j in findnz(row)[1]
        wij = row[j]
        s += wij*state[j]
    end
    return s
end

function LV_adj(state::S, sparse_adj, i) where S
    s = zero(eltype(state))
    @turbo for ptr in nzrange(sparse_adj, i)
        j = sparse_adj.rowval[ptr]
        wij = sparse_adj.nzval[ptr]
        s += wij*state[j]
    end
    return s
end

function Tullio_adj(state::S, sparse_adj, i) where S
    s = zero(eltype(state))
    @tullio s += sparse_adj[1,j]*state[j] 
    return s
end

LV_adj(state, adj, 1)
Tullio_adj(state, adj, 1)


@benchmark normal_nzrange(state, adj, 1)
@benchmark normal_view(state, adj, 1)
@benchmark LV_adj(state, adj, 1)
@benchmark Tullio_adj(state, adj, 1)



### FINCH TESTS


"""
Convert a CSC adjacency matrix into a Finch sparse tensor.
"""
to_finch_sparse(adj::SparseMatrixCSC) = Finch.Tensor(adj)
to_finch_sparse(g::AbstractIsingGraph) = to_finch_sparse(adj(g))

"""
Compute the i-th row contraction Σ_j w[i,j] * s[j] using Finch.
"""
Base.@propagate_inbounds function finch_row_contraction(state::AbstractVector{T}, wadj_finch, i::Integer) where {T}
    s = Finch.Tensor(state)
    acc = Finch.Scalar(zero(T))
    Finch.@finch for j = _
        acc[] += wadj_finch[i, j] * s[j]
    end
    return acc[]
end

finch_row_contraction(state::AbstractArray{T}, wadj_finch, i::Integer) where {T} =
    finch_row_contraction(vec(state), wadj_finch, i)

function finch_row_contraction(state::AbstractArray{T}, wadj::SparseMatrixCSC, i::Integer) where {T}
    finch_row_contraction(state, to_finch_sparse(wadj), i)
end

"""
Compute the i-th row term of s_i * w_ij * s_j using Finch.
"""
function finch_row_sws(state::AbstractArray{T}, wadj_finch, i::Integer) where {T}
    s = vec(state)
    @boundscheck checkbounds(s, i)
    return s[i] * finch_row_contraction(s, wadj_finch, i)
end

function finch_row_sws(state::AbstractArray{T}, wadj::SparseMatrixCSC, i::Integer) where {T}
    finch_row_sws(state, to_finch_sparse(wadj), i)
end


f_adj = to_finch_sparse(adj)

finch_row_contraction(state, f_adj, 1)
@benchmark finch_row_contraction(state, f_adj, 1)