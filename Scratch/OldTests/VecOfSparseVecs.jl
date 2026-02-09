using SparseArrays, LoopVectorization
function sparse_arr2vecs(arr::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    vecs = Vector{SparseVector{Tv,Ti}}(undef, size(arr, 2))
    for i in axes(arr, 2)
        vecs[i] = arr[:,i]
    end
    return vecs
end

function ΔEarrs(g, oldstate, newstate, gstate::Vector{T}, sparsevecs, idx, gstype, StateT) where T
    efac = zero(T)
    vec = sparsevecs[idx]
    ptrs = vec.nzind
    @turbo for ptr in eachindex(ptrs)
        efac += -gstate[vec.nzind[ptr]]*vec.nzval[ptr]
    end
    return (newstate-oldstate)*efac
end
ΔEarrs(g,adj,idx)= ΔEarrs(g, state(g), state(g), state(g), adj, idx, stype(g), Discrete)  