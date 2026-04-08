"""
Sparse Adjacency matrix that has fast access for fixed connectivity
    Changing connectivity is slow, but changing weights is fast
    Useful for models where the topology is fixed but weights are learned
    E.g. Ising models with fixed interactions but learnable coupling strengths
"""
struct UndirectedAdjacency{Tv, Ti, SP, D, I} <: AbstractSparseMatrix{Tv, Ti}
    sp::SP
    diag::D
    indexmap::I
end

fastwrite(::Union{UndirectedAdjacency{Tv, Ti, SP, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, SP, D, I}}}) where {Tv, Ti, SP, D, I} = !(I <: Nothing)
separate_diagonal(::Union{UndirectedAdjacency{Tv, Ti, SP, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, SP, D, I}}}) where {Tv, Ti, SP, D, I} = !(D <: Nothing)

function UndirectedAdjacency(sp::AbstractSparseMatrix{Tv, Ti}, diag::D = nothing; fastwrite = false) where {Tv, Ti, D}
    indexmap = fastwrite ? map_topology(sp) : nothing
    return UndirectedAdjacency{Tv, Ti, typeof(sp), D, typeof(indexmap)}(sp, diag, indexmap)
end

@inline _copy_diag(diag) = copy(diag)
@inline _copy_diag(::Nothing) = nothing

@inline function _same_sparse_topology(sp1::SparseMatrixCSC, sp2::SparseMatrixCSC)
    size(sp1) == size(sp2) || return false
    sp1.colptr == sp2.colptr || return false
    sp1.rowval == sp2.rowval || return false
    return true
end

"""
Create a new UndirectedAdjacency from an existing one, optionally replacing the sparse
connections and/or the diagonal while inheriting any omitted part from the original.

If `check_topology` is true, a provided sparse matrix must have the same sparsity
pattern as the original adjacency.
"""
function reconstruct(ua::UndirectedAdjacency; sp = nothing, diag = nothing, fastwrite = fastwrite(ua), check_topology = true)
    sp_new = if isnothing(sp)
        copy(ua.sp)
    else
        copy(sparse(sp))
    end

    size(sp_new) == size(ua.sp) || throw(DimensionMismatch("New sparse adjacency must have size $(size(ua.sp)), got $(size(sp_new))"))
    if !isnothing(sp) && check_topology && !_same_sparse_topology(ua.sp, sp_new)
        throw(ArgumentError("New sparse adjacency must have the same nonzero topology as the original adjacency"))
    end

    diag_new = if isnothing(diag)
        separate_diagonal(ua) ? _copy_diag(ua.diag) : nothing
    else
        _copy_diag(diag)
    end

    if !isnothing(diag_new) && length(diag_new) != size(ua.sp, 1)
        throw(DimensionMismatch("New diagonal must have length $(size(ua.sp, 1)), got $(length(diag_new))"))
    end

    return UndirectedAdjacency(sp_new, diag_new; fastwrite)
end

"""
From another UndirectedAdjacency, create a new one with provided sparse connections and
optionally a new diagonal, while inheriting omitted data from the original adjacency.

This preserves the existing constructor API by using a non-conflicting positional sparse
matrix argument instead of a keyword-only overload.
"""
@inline function UndirectedAdjacency(ua::UndirectedAdjacency, sp::AbstractSparseMatrix; diag = nothing, fastwrite = fastwrite(ua), check_topology = true)
    return reconstruct(ua; sp, diag, fastwrite, check_topology)
end

"""
From another UndirectedAdjacency, create a new one with the same topology but new weights and/or diagonal
    If diag is not provided, it will be copied from the original if it exists, otherwise set to nothing
    If fastwrite is true, a new indexmap will be created for the new adjacency, otherwise it will be set to nothing
"""
function UndirectedAdjacency(ua::UndirectedAdjacency{Tv, Ti, SP, D, I}, diag::D = nothing; fastwrite = false) where {Tv, Ti, SP, D, I}
    sp_copy = copy(ua.sp)
    indexmap_copy = fastwrite ? copy(ua.indexmap) : nothing
    diag = isnothing(diag) ? (separate_diagonal(ua) ? copy(ua.diag) : nothing) : diag
    return UndirectedAdjacency{Tv, Ti, SP, D, I}(sp_copy, diag, indexmap_copy)
end

"""
Give the columns rows and values of a new sparse matrix with the same topology as an existing UndirectedAdjacency, create a new UndirectedAdjacency with the new weights
    This is useful for efficiently updating the weights of an existing adjacency without changing the topology
    If fastwrite is true, a new indexmap will be created for the new adjacency, otherwise it will be set to nothing
"""
function UndirectedAdjacency(ua::UndirectedAdjacency{Tv, Ti, SP, D, I}, cols, rows, vals) where {Tv, Ti, SP, D, I}
    sp_new = sparse(rows, cols, vals, size(ua.sp)...)
    indexmap_new = fastwrite(ua) ? map_topology(sp_new) : nothing
    return UndirectedAdjacency{Tv, Ti, SP, D, I}(sp_new, ua.diag, indexmap_new)
end

function UndirectedAdjacency(rows, cols, vals, m, n; diag = nothing, fastwrite = false)
    sp = sparse(rows, cols, vals, m, n)
    indexmap = fastwrite ? map_topology(sp) : nothing
    return UndirectedAdjacency{eltype(sp.nzval), eltype(sp.rowval), typeof(sp), typeof(diag), typeof(indexmap)}(sp, diag, indexmap)
end

function UndirectedAdjacency(m::Integer, n::Integer; diag = nothing, fastwrite = false)
    sp = spzeros(m, n)
    indexmap = fastwrite ? map_topology(sp) : nothing
    return UndirectedAdjacency{eltype(sp.nzval), eltype(sp.rowval), typeof(sp), typeof(diag), typeof(indexmap)}(sp, diag, indexmap)
end

"""
Map the topology of a sparse matrix to a dictionary for fast access
    Returns a Dict mapping (row, col) pairs to the index in the sparse matrix's storage
    This allows O(1) access to the nonzero entries of the sparse matrix
    Useful for models where we need to frequently update weights based on fixed topology
"""
function map_topology(sp::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    indexmap = Dict{Pair{Ti, Ti}, Ti}()
    for col in 1:size(sp, 2)
        for row in sp.rowval[sp.colptr[col]:(sp.colptr[col + 1] - 1)]
            pair = row => col
            indexmap[pair] = length(indexmap) + 1
        end
    end
    return indexmap
end

function remap!(adj::UndirectedAdjacency)
    indexmap = adj.indexmap
    sp = adj.sp
    empty!(indexmap)
    for col in 1:size(sp, 2)
        for row in sp.rowval[sp.colptr[col]:(sp.colptr[col + 1] - 1)]
            pair = row => col
            indexmap[pair] = length(indexmap) + 1
        end
    end
    return sp
end
