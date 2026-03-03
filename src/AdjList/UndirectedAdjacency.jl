"""
Sparse Adjacency matrix that has fast access for fixed connectivity
    Changing connectivity is slow, but changing weights is fast
    Useful for models where the topology is fixed but weights are learned
    E.g. Ising models with fixed interactions but learnable coupling strengths
"""
struct UndirectedAdjacency{Tv, Ti, D, I} <: AbstractSparseMatrix{Tv, Ti}
    sp::SparseMatrixCSC{Tv, Ti}
    diag::D
    # indexmap::Dict{Pair{Ti, Ti}, Ti}
    indexmap::I
end

fastwrite(::Union{UndirectedAdjacency{Tv, Ti, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, D, I}}}) where {Tv, Ti, D, I} = !(I <: Nothing)
separate_diagonal(::Union{UndirectedAdjacency{Tv, Ti, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, D, I}}}) where {Tv, Ti, D, I} = !(D <: Nothing)

function UndirectedAdjacency(sp::SparseMatrixCSC{Tv, Ti}, diag::D = nothing; fastwrite = false) where {Tv, Ti, D}
    indexmap = fastwrite ? map_topology(sp) : nothing
    return UndirectedAdjacency(sp, diag, indexmap)
end

"""
From another UndirectedAdjacency, create a new one with the same topology but new weights and/or diagonal
    If diag is not provided, it will be copied from the original if it exists, otherwise set to nothing
    If fastwrite is true, a new indexmap will be created for the new adjacency, otherwise it will be set to nothing
"""
function UndirectedAdjacency(ua::UndirectedAdjacency, diag::D = nothing; fastwrite = false) where {D}
    sp_copy = copy(ua.sp)
    indexmap_copy = fastwrite ? copy(ua.indexmap) : nothing
    diag = isnothing(diag) ? (separate_diagonal(ua) ? copy(ua.diag) : nothing) : diag
    return UndirectedAdjacency(sp_copy, diag, indexmap_copy)
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
@inline function Base.setindex!(A::UA, value, row::Ti, col::Ti) where {UA <: UndirectedAdjacency, Ti}
    if separate_diagonal(A) && row == col
        A.diag[row] = value
        return A
    end
    @inline _setindex!(Val(fastwrite(A)), A, value, row, col)
    return A
end

@inline function _setindex!(fastwrite::Val{true}, A::UndirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    pair = row => col
    if haskey(A.indexmap, pair)
        idx = A.indexmap[pair]
        A.sp.nzval[idx] = value
        second_pair = col => row
        idx_sym = A.indexmap[second_pair]
        A.sp.nzval[idx_sym] = value # Ensure symmetry
    else
        # error("Attempting to set a value for a non-existent edge in the sparse adjacency matrix")
        A.sp[row, col] = value
        A.sp[col, row] = value # Ensure symmetry
        remap!(A) # Update the indexmap after modifying the sparse matrix
        @warn "Adding new edge to a fastwrite UndirectedAdjacency is slow due to remapping. Consider preallocating all edges or using a non-fastwrite adjacency for dynamic topologies."
    end
end

@inline function _setindex!(fastwrite::Val{false}, A::UndirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    A.sp[row, col] = value
    A.sp[col, row] = value # Ensure symmetry
end

@inline function Base.getindex(A::UndirectedAdjacency, row::Ti, col::Ti) where {Ti}
    if separate_diagonal(A) && row == col
        return A.diag[row]
    end
    getindex(A.sp, row, col) # This will return zero for non-existent edges due to the sparse matrix structure  
end

#Forward sparse methods:
Base.size(A::UndirectedAdjacency) = size(A.sp)
SparseArrays.nnz(A::UndirectedAdjacency) = nnz(A.sp)
SparseArrays.rowvals(A::UndirectedAdjacency) = rowvals(A.sp)
SparseArrays.getcolptr(A::UndirectedAdjacency) = SparseArrays.getcolptr(A.sp)
SparseArrays.getrowval(A::UndirectedAdjacency) = SparseArrays.getrowval(A.sp)
SparseArrays.getnzval(A::UndirectedAdjacency) = SparseArrays.getnzval(A.sp)
SparseArrays.sparse(A::UndirectedAdjacency) = A.sp
SparseArrays.nzrange(A::UndirectedAdjacency, col::Int) = nzrange(A.sp, col)
SparseArrays.nonzeros(A::UndirectedAdjacency) = nonzeros(A.sp)
Base.transpose(A::UndirectedAdjacency) = A # Undirected adjacency is symmetric, so transpose is the same
Base.adjoint(A::UndirectedAdjacency) = A # Undirected adjacency is symmetric, so adjoint is the same
function Base.copy(A::UndirectedAdjacency)
    indexmap = fastwrite(A) ? copy(A.indexmap) : nothing
    diag = separate_diagonal(A) ? copy(A.diag) : nothing
    UndirectedAdjacency(copy(A.sp), diag, indexmap)
end

@inline function weighted_neighbors_sum(node_idx, adj::UA, nodes::AV) where {UA <: UndirectedAdjacency, AV<:AbstractVector}
    @inline column_contraction(node_idx, nodes, adj.sp)
end



