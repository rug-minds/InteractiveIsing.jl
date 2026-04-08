using SparseArrays

"""
Directed sparse adjacency with mirrored CSC and CSR storage.

- `sp` stores CSC for fast column-wise (incoming) iteration.
- `sp_csr` stores CSR for fast row-wise (outgoing) iteration.
- Optional `indexmap` enables O(1) value updates for fixed topology.
  The mapped value is `(csc_ptr, csr_ptr)`.
"""
struct DirectedAdjacency{Tv, Ti, D, I, C} <: AbstractSparseMatrix{Tv, Ti}
    sp::SparseMatrixCSC{Tv, Ti}
    sp_csr::C
    diag::D
    indexmap::I
end

fastwrite(::Union{DirectedAdjacency{Tv, Ti, D, I, C}, Type{<:DirectedAdjacency{Tv, Ti, D, I, C}}}) where {Tv, Ti, D, I, C} = !(I <: Nothing)
separate_diagonal(::Union{DirectedAdjacency{Tv, Ti, D, I, C}, Type{<:DirectedAdjacency{Tv, Ti, D, I, C}}}) where {Tv, Ti, D, I, C} = !(D <: Nothing)

# CSR accessors that tolerate minor API variations across SparseMatricesCSR versions.
@inline function _csr_rowptr(csr)
    if hasproperty(csr, :rowptr)
        return getproperty(csr, :rowptr)
    elseif isdefined(SparseMatricesCSR, :getrowptr)
        return SparseMatricesCSR.getrowptr(csr)
    else
        error("Could not resolve CSR row pointer storage")
    end
end

@inline function _csr_colval(csr)
    if hasproperty(csr, :colval)
        return getproperty(csr, :colval)
    elseif hasproperty(csr, :colvals)
        return getproperty(csr, :colvals)
    elseif isdefined(SparseMatricesCSR, :getcolval)
        return SparseMatricesCSR.getcolval(csr)
    else
        error("Could not resolve CSR column-index storage")
    end
end

@inline function _csr_nzval(csr)
    if hasproperty(csr, :nzval)
        return getproperty(csr, :nzval)
    elseif isdefined(SparseMatricesCSR, :getnzval)
        return SparseMatricesCSR.getnzval(csr)
    else
        error("Could not resolve CSR nzval storage")
    end
end

@inline _to_csr(sp::SparseMatrixCSC) = SparseMatricesCSR.SparseMatrixCSR(sp)

function DirectedAdjacency(sp::SparseMatrixCSC{Tv, Ti}, diag::D = nothing; fastwrite = false) where {Tv, Ti, D}
    sp_csr = _to_csr(sp)
    indexmap = fastwrite ? map_topology(sp, sp_csr) : nothing
    return DirectedAdjacency(sp, sp_csr, diag, indexmap)
end

"""
Copy topology from another DirectedAdjacency, with optional new diagonal and fastwrite mode.
"""
function DirectedAdjacency(da::DirectedAdjacency, diag::D = nothing; fastwrite = false) where {D}
    sp_copy = copy(da.sp)
    sp_csr_copy = copy(da.sp_csr)
    indexmap_copy = if fastwrite
        fastwrite(da) ? copy(da.indexmap) : map_topology(sp_copy, sp_csr_copy)
    else
        nothing
    end
    diag = isnothing(diag) ? (separate_diagonal(da) ? copy(da.diag) : nothing) : diag
    return DirectedAdjacency(sp_copy, sp_csr_copy, diag, indexmap_copy)
end

"""
Build map: (row=>col) -> (ptr_in_csc_nzval, ptr_in_csr_nzval)
"""
function map_topology(sp::SparseMatrixCSC{Tv, Ti}, sp_csr) where {Tv, Ti}
    csc_map = Dict{Pair{Ti, Ti}, Ti}()
    for col in 1:size(sp, 2)
        for ptr in sp.colptr[col]:(sp.colptr[col + 1] - 1)
            row = sp.rowval[ptr]
            csc_map[row => col] = Ti(ptr)
        end
    end

    csr_rowptr = _csr_rowptr(sp_csr)
    csr_colval = _csr_colval(sp_csr)

    csr_map = Dict{Pair{Ti, Ti}, Ti}()
    for row in 1:size(sp, 1)
        for ptr in csr_rowptr[row]:(csr_rowptr[row + 1] - 1)
            col = csr_colval[ptr]
            csr_map[Ti(row) => Ti(col)] = Ti(ptr)
        end
    end

    indexmap = Dict{Pair{Ti, Ti}, Tuple{Ti, Ti}}()
    for (pair, csc_ptr) in csc_map
        if haskey(csr_map, pair)
            indexmap[pair] = (csc_ptr, csr_map[pair])
        end
    end
    return indexmap
end

function remap!(adj::DirectedAdjacency)
    indexmap = adj.indexmap
    empty!(indexmap)
    newmap = map_topology(adj.sp, adj.sp_csr)
    for (pair, ptrs) in newmap
        indexmap[pair] = ptrs
    end
    return adj
end

@inline function Base.setindex!(A::DA, value, row::Ti, col::Ti) where {DA <: DirectedAdjacency, Ti}
    if separate_diagonal(A) && row == col
        A.diag[row] = value
        return A
    end
    @inline _setindex!(Val(fastwrite(A)), A, value, row, col)
    return A
end

@inline function _setindex!(::Val{true}, A::DirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    pair = row => col
    if haskey(A.indexmap, pair)
        csc_ptr, csr_ptr = A.indexmap[pair]
        A.sp.nzval[csc_ptr] = value
        _csr_nzval(A.sp_csr)[csr_ptr] = value
    else
        # Topology changed: update both sparse stores, then remap pointers.
        A.sp[row, col] = value
        A.sp_csr[row, col] = value
        remap!(A)
        @warn "Adding new edge to a fastwrite DirectedAdjacency is slow due to remapping. Consider preallocating all edges or using non-fastwrite mode for dynamic topologies."
    end
end

@inline function _setindex!(::Val{false}, A::DirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    A.sp[row, col] = value
    A.sp_csr[row, col] = value
end

@inline function Base.getindex(A::DirectedAdjacency, row::Ti, col::Ti) where {Ti}
    if separate_diagonal(A) && row == col
        return A.diag[row]
    end
    return getindex(A.sp, row, col)
end

# Fast iteration helper for outgoing edges (CSR row traversal).
@inline function outnzrange(A::DirectedAdjacency, row::Int)
    rp = _csr_rowptr(A.sp_csr)
    return rp[row]:(rp[row + 1] - 1)
end

# Forward sparse-like API through CSC storage by default.
Base.size(A::DirectedAdjacency) = size(A.sp)
SparseArrays.nnz(A::DirectedAdjacency) = nnz(A.sp)
SparseArrays.rowvals(A::DirectedAdjacency) = rowvals(A.sp)
SparseArrays.getcolptr(A::DirectedAdjacency) = SparseArrays.getcolptr(A.sp)
SparseArrays.getrowval(A::DirectedAdjacency) = SparseArrays.getrowval(A.sp)
SparseArrays.getnzval(A::DirectedAdjacency) = SparseArrays.getnzval(A.sp)
SparseArrays.sparse(A::DirectedAdjacency) = A.sp
SparseArrays.nzrange(A::DirectedAdjacency, col::Int) = nzrange(A.sp, col)
SparseArrays.nonzeros(A::DirectedAdjacency) = nonzeros(A.sp)
csr_rowptr(A::DirectedAdjacency) = _csr_rowptr(A.sp_csr)
csr_colval(A::DirectedAdjacency) = _csr_colval(A.sp_csr)
csr_nzval(A::DirectedAdjacency) = _csr_nzval(A.sp_csr)

function Base.copy(A::DirectedAdjacency)
    indexmap = fastwrite(A) ? copy(A.indexmap) : nothing
    diag = separate_diagonal(A) ? copy(A.diag) : nothing
    DirectedAdjacency(copy(A.sp), copy(A.sp_csr), diag, indexmap)
end
