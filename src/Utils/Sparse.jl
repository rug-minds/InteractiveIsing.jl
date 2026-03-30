### Helper functions for sparse arrays ###

"""
Delete a value at a given index in a sparse array
"""
function deleteval!(sp_adj, i,j) 
    searchrange = sp_adj.colptr[j]:(sp_adj.colptr[j+1]-1)
    idx = findfirst(x -> x == i, sp_adj.rowval , searchrange)
    if !isnothing(idx)
        deleteat!(sp_adj.rowval, idx)
        deleteat!(sp_adj.nzval, idx)
        sp_adj.colptr[j+1:end] .-= 1
    end
    return sp_adj
end

"""
Delete all values within a range at a given column index in a sparse array
"""
function delete_colrange!(sp_adj, col, range)
    searchrange = sp_adj.colptr[col]:(sp_adj.colptr[col+1]-1)
    del_range = findorderedrange(first(range), last(range), sp_adj.rowval , searchrange)
    deleteat!(sp_adj.rowval, del_range)
    deleteat!(sp_adj.nzval, del_range)
    if !isempty(del_range)
        sp_adj.colptr[col+1:end] .-= length(del_range)
    end
    return sp_adj
end
export delete_colrange!

"""
Extend findfirst so that it can search within a range
"""
function Base.findfirst(predicate::Function, A, searchrange::UnitRange)
    idx = findfirst(predicate, (@view A[searchrange]))

    !isnothing(idx) && (idx += searchrange.start - 1)
    return idx
end

"""
Extend findlast so that it can search within a range
"""
function Base.findlast(predicate::Function, A, searchrange::UnitRange)
    idx = findlast(predicate, (@view A[searchrange]))

    !isnothing(idx) && (idx += searchrange.start - 1)
    return idx
end

"""
Find the first range of values that satisfy a predicate
"""
function findrange(predicate::Function, A)
    start = 1
    stop = 0
    for idx in eachindex(A)
        if predicate(A[idx])
            start = idx
            stop = idx-1
            break
        end
    end
    for idx in (start+1):length(A)
        if !predicate(A[idx])
            stop = idx-1
            break
        end
    end
    return start:stop
end

"""
Find the first range of values that satisfy a predicate within a given range
"""
function findrange(predicate::Function, A, searchrange)
    range = findrange(predicate, (@view A[searchrange]))
    (range = range .+ (searchrange.start - 1))
    return range
end

"""
Find the first range of values that are within a given range, in an array that is ordered w.r.p. the predicate
"""
function findorderedrange(startrange, endrange, A)
    start = 1
    stop = 0
    if A[1] > endrange
        return 1:0
    end

    for idx in eachindex(A)
        if A[idx] >= startrange
            start = idx
            stop = idx-1
            break
        end
    end
    for idx in (start+1):length(A)
        if A[idx] > endrange
            stop = idx-1
            break
        end
    end
    return start:stop
end

"""
Find the first range of values that are within a given range, in an array that is ordered w.r.p. the predicate, within a given range
"""
function findorderedrange(startrange, endrange, A, searchrange)
    range = findorderedrange(startrange, endrange, (@view A[searchrange]))
    (range = range .+ (searchrange.start - 1))
    return range
end

"""
Insert empty rows and columns into a sparse matrix at the given range.
E.g. if the range is n:m, new columns at n to m AND new rows at n:m will be added
"""
function insertrowcol(m::SparseMatrixCSC, range::UnitRange)
    newmm = m.m + length(range)
    newmn = m.n + length(range)
    colptr = copy(m.colptr)
    rowval = copy(m.rowval)
    nzval = copy(m.nzval)
    
    start_idx = first(range)
    insert!(colptr, start_idx, repeat([colptr[start_idx]], length(range)))
    for (idx, row_idx) in enumerate(rowval)
        if row_idx >= start_idx
            rowval[idx] += length(range)
        end
    end
    return SparseMatrixCSC(newmm, newmn, colptr, rowval, nzval)
end

function deleterowcol(m::SparseMatrixCSC, range::UnitRange)
    rows, cols, vals = findnz(m)

    idxs_to_remove = Int[]
    sizehint!(idxs_to_remove, length(rows))
    for idx in eachindex(rows)
        if rows[idx] in range || cols[idx] in range
            push!(idxs_to_remove, idx)
        end
        if rows[idx] > last(range)
            rows[idx] -= length(range)
        end
        if cols[idx] > last(range)
            cols[idx] -= length(range)
        end
    end
    deleteat!(rows, idxs_to_remove)
    deleteat!(cols, idxs_to_remove)
    deleteat!(vals, idxs_to_remove)

    return sparse(rows, cols, vals, m.m - length(range), m.n - length(range))
end

emptysparse(length,width, T = Float32) = sparse(Int[], Int[], T[], length, width)
export emptysparse
