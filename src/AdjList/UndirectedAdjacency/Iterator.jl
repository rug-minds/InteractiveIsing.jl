function connection_iterator(g::UndirectedAdjacency, v, include_diag = separate_diagonal(g))
    it = (
        (A.rowval[idx], j, A.nzval[idx])
        for j in 1:size(A,2)
        for idx in A.colptr[j]:(A.colptr[j+1]-1)
    )
    if include_diag
        d = diag(g)
        diag_it = ( (i,i, d[i]) for i in eachindex(d) )
        return Iterators.flatten((it, diag_it))
    else
        return it
    end
end

function index_pairs_iterator(g::UndirectedAdjacency, include_diag = separate_diagonal(g))
    it = (
        (A.rowval[idx], j)
        for j in 1:size(A,2)
        for idx in A.colptr[j]:(A.colptr[j+1]-1)
    )
    if include_diag
        d = diag(g)
        diag_it = ( (i,i) for i in eachindex(d) )
        return Iterators.flatten((it, diag_it))
    else
        return it
    end
end