"""
    connection_iterator(adj::SparseMatrixCSC, include_diag = false)

Return an iterator over `(row, col, weight)` triples stored in `adj`. Diagonal
entries are omitted unless `include_diag` is true.
"""
function connection_iterator(adj::SP, include_diag::B = false) where {SP <: SparseMatrixCSC, B <: Bool}
    if include_diag
        # Keep the raw CSC storage order: columns outside, stored pointers inside.
        return (
            (SparseArrays.rowvals(adj)[ptr], col, SparseArrays.nonzeros(adj)[ptr])
            for col in axes(adj, 2)
            for ptr in SparseArrays.nzrange(adj, col)
        )
    end

    return (
        (SparseArrays.rowvals(adj)[ptr], col, SparseArrays.nonzeros(adj)[ptr])
        for col in axes(adj, 2)
        for ptr in SparseArrays.nzrange(adj, col)
        if SparseArrays.rowvals(adj)[ptr] != col
    )
end

"""
    index_pairs_iterator(adj::SparseMatrixCSC, include_diag = false)

Return an iterator over `(row, col)` pairs stored in `adj`. Diagonal entries
are omitted unless `include_diag` is true.
"""
function index_pairs_iterator(adj::SP, include_diag::B = false) where {SP <: SparseMatrixCSC, B <: Bool}
    if include_diag
        # Keep pair iteration aligned with `connection_iterator` storage order.
        return (
            (SparseArrays.rowvals(adj)[ptr], col)
            for col in axes(adj, 2)
            for ptr in SparseArrays.nzrange(adj, col)
        )
    end

    return (
        (SparseArrays.rowvals(adj)[ptr], col)
        for col in axes(adj, 2)
        for ptr in SparseArrays.nzrange(adj, col)
        if SparseArrays.rowvals(adj)[ptr] != col
    )
end
