@inline function Base.setindex!(A::UA, value, row::Ti, col::Ti) where {UA <: UndirectedAdjacency, Ti}
    if separate_diagonal(A) && row == col
        A.diag[row] = value
        return A
    end
    @inline _setindex!(Val(fastwrite(A)), A, value, row, col)
    return A
end

@inline function _setindex!(::Val{true}, A::UndirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    pair = row => col
    if haskey(A.indexmap, pair)
        idx = A.indexmap[pair]
        A.sp.nzval[idx] = value
        second_pair = col => row
        idx_sym = A.indexmap[second_pair]
        A.sp.nzval[idx_sym] = value
    else
        A.sp[row, col] = value
        A.sp[col, row] = value
        remap!(A)
        @warn "Adding new edge to a fastwrite UndirectedAdjacency is slow due to remapping. Consider preallocating all edges or using a non-fastwrite adjacency for dynamic topologies."
    end
end

@inline function _setindex!(::Val{false}, A::UndirectedAdjacency, value, row::Ti, col::Ti) where {Ti}
    A.sp[row, col] = value
    A.sp[col, row] = value
end

@inline function Base.getindex(A::UndirectedAdjacency, row::Ti, col::Ti) where {Ti <: Integer}
    if separate_diagonal(A) && row == col
        return A.diag[row]
    end
    if fastwrite(A)
        if haskey(A.indexmap, row => col)
            idx = A.indexmap[row => col]
            return A.sp.nzval[idx]
        else
            return zero(eltype(A.sp.nzval))
        end
    else
        return A.sp[row, col]
    end
end

@inline function _sparse_slice(A::UndirectedAdjacency{Tv, Ti}, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) where {Tv, Ti}
    sub = A.sp[rows, cols]
    if !separate_diagonal(A)
        return sub
    end

    sub_rows, sub_cols, sub_vals = findnz(sub)
    out_rows = Ti[]
    out_cols = Ti[]
    out_vals = Tv[]
    sizehint!(out_rows, length(sub_rows))
    sizehint!(out_cols, length(sub_cols))
    sizehint!(out_vals, length(sub_vals))

    for k in eachindex(sub_vals)
        r = sub_rows[k]
        c = sub_cols[k]
        if rows[r] != cols[c]
            push!(out_rows, Ti(r))
            push!(out_cols, Ti(c))
            push!(out_vals, Tv(sub_vals[k]))
        end
    end

    col_positions = Dict{Int, Vector{Int}}()
    for (c, gidx) in pairs(cols)
        positions = get!(col_positions, Int(gidx), Int[])
        push!(positions, c)
    end

    for (r, gidx) in pairs(rows)
        diag_cols = get(col_positions, Int(gidx), nothing)
        diag_cols === nothing && continue
        dval = Tv(A.diag[gidx])
        iszero(dval) && continue
        for c in diag_cols
            push!(out_rows, Ti(r))
            push!(out_cols, Ti(c))
            push!(out_vals, dval)
        end
    end

    return sparse(out_rows, out_cols, out_vals, length(rows), length(cols))
end

Base.:(==)(A1::UndirectedAdjacency, A2::UndirectedAdjacency) = A1.sp == A2.sp && A1.diag == A2.diag

@inline Base.getindex(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = _sparse_slice(A, rows, cols)
@inline Base.getindex(A::UndirectedAdjacency, ::Colon, cols::AbstractVector{<:Integer}) = _sparse_slice(A, axes(A, 1), cols)
@inline Base.getindex(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, ::Colon) = _sparse_slice(A, rows, axes(A, 2))
@inline Base.getindex(A::UndirectedAdjacency, ::Colon, ::Colon) = _sparse_slice(A, axes(A, 1), axes(A, 2))

@inline function Base.getindex(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, col::Integer)
    sub = _sparse_slice(A, rows, (col:col))
    r, _, v = findnz(sub)
    return sparsevec(r, v, length(rows))
end

@inline function Base.getindex(A::UndirectedAdjacency, row::Integer, cols::AbstractVector{<:Integer})
    sub = _sparse_slice(A, (row:row), cols)
    _, c, v = findnz(sub)
    return sparsevec(c, v, length(cols))
end

@inline Base.getindex(A::UndirectedAdjacency, ::Colon, col::Integer) = getindex(A, axes(A, 1), col)
@inline Base.getindex(A::UndirectedAdjacency, row::Integer, ::Colon) = getindex(A, row, axes(A, 2))

@inline function Base.setindex!(A::UndirectedAdjacency, val, rows::AbstractVector{<:Integer}, col::Integer)
    _set_slice!(A, val, rows, Int(col))
end
@inline function Base.setindex!(A::UndirectedAdjacency, val, ::Colon, col::Integer)
    _set_slice!(A, val, axes(A, 1), Int(col))
end
@inline function Base.setindex!(A::UndirectedAdjacency, val, row::Integer, cols::AbstractVector{<:Integer})
    _set_slice!(A, val, cols, Int(row))
end
@inline function Base.setindex!(A::UndirectedAdjacency, val, row::Integer, ::Colon)
    _set_slice!(A, val, axes(A, 2), Int(row))
end

@inline function _set_slice!(A::UndirectedAdjacency, val, inds, fixed::Int)
    sp = A.sp
    @inbounds for (i, idx) in enumerate(inds)
        v = val isa AbstractArray ? val[i] : val
        if separate_diagonal(A) && idx == fixed
            A.diag[fixed] = v
            continue
        end
        @inline _set_offdiag!(Val(fastwrite(A)), sp, A, v, idx, fixed)
    end
end

@inline function _set_offdiag!(::Val{true}, sp, A, v, row, col)
    sp.nzval[A.indexmap[row => col]] = v
    sp.nzval[A.indexmap[col => row]] = v
end

@inline function _set_offdiag!(::Val{false}, sp, A, v, row, col)
    sp[row, col] = v
    sp[col, row] = v
end

@inline function Base.setindex!(A::UndirectedAdjacency, val, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @inbounds for (j, col) in enumerate(cols)
        for (i, row) in enumerate(rows)
            v = val isa AbstractArray ? val[i, j] : val
            A[row, col] = v
        end
    end
end

struct UAView{Tv, P<:UndirectedAdjacency{Tv}, R, C} <: AbstractArray{Tv, 2}
    parent::P
    rows::R
    cols::C
end

@inline _resolve(inds::AbstractVector, i) = @inbounds inds[i]
@inline _resolve(fixed::Integer, _) = fixed
@inline _viewsize(inds::AbstractVector) = length(inds)
@inline _viewsize(::Integer) = 1

Base.size(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}) = (_viewsize(v.rows), _viewsize(v.cols))
Base.size(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}) = (_viewsize(v.cols),)
Base.size(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}) = (_viewsize(v.rows),)

Base.IndexStyle(::Type{<:UAView}) = IndexCartesian()

@inline Base.getindex(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}, i::Integer, j::Integer) =
    @inbounds v.parent[_resolve(v.rows, i), _resolve(v.cols, j)]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}, val, i::Integer, j::Integer) =
    @inbounds (v.parent[_resolve(v.rows, i), _resolve(v.cols, j)] = val)

@inline Base.getindex(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}, i::Integer) =
    @inbounds v.parent[v.rows, _resolve(v.cols, i)]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}, val, i::Integer) =
    @inbounds (v.parent[v.rows, _resolve(v.cols, i)] = val)

@inline Base.getindex(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}, i::Integer) =
    @inbounds v.parent[_resolve(v.rows, i), v.cols]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}, val, i::Integer) =
    @inbounds (v.parent[_resolve(v.rows, i), v.cols] = val)

Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = UAView(A, rows, cols)
Base.view(A::UndirectedAdjacency, ::Colon, cols::AbstractVector{<:Integer}) = UAView(A, axes(A, 1), cols)
Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, ::Colon) = UAView(A, rows, axes(A, 2))
Base.view(A::UndirectedAdjacency, ::Colon, ::Colon) = UAView(A, axes(A, 1), axes(A, 2))
Base.view(A::UndirectedAdjacency, row::Integer, cols::AbstractVector{<:Integer}) = UAView(A, Int(row), cols)
Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, col::Integer) = UAView(A, rows, Int(col))
Base.view(A::UndirectedAdjacency, row::Integer, ::Colon) = UAView(A, Int(row), axes(A, 2))
Base.view(A::UndirectedAdjacency, ::Colon, col::Integer) = UAView(A, axes(A, 1), Int(col))

Base.size(A::UndirectedAdjacency) = size(A.sp)
SparseArrays.nnz(A::UndirectedAdjacency) = nnz(A.sp)
SparseArrays.rowvals(A::UndirectedAdjacency) = rowvals(A.sp)
SparseArrays.getcolptr(A::UndirectedAdjacency) = SparseArrays.getcolptr(A.sp)
SparseArrays.getrowval(A::UndirectedAdjacency) = SparseArrays.getrowval(A.sp)
SparseArrays.getnzval(A::UndirectedAdjacency) = SparseArrays.getnzval(A.sp)
SparseArrays.sparse(A::UndirectedAdjacency) = A.sp
SparseArrays.nzrange(A::UndirectedAdjacency, col::Int) = nzrange(A.sp, col)
SparseArrays.nonzeros(A::UndirectedAdjacency) = nonzeros(A.sp)
SparseArrays.findnz(A::UndirectedAdjacency) = findnz(A.sp)
Base.transpose(A::UndirectedAdjacency) = A
Base.adjoint(A::UndirectedAdjacency) = A

@inline function _sparse_with_diag(A::UndirectedAdjacency{Tv, Ti}) where {Tv, Ti}
    if !separate_diagonal(A)
        return A.sp
    end

    m, n = size(A.sp)
    sp_rows, sp_cols, sp_vals = findnz(A.sp)

    out_rows = Ti[]
    out_cols = Ti[]
    out_vals = Tv[]
    sizehint!(out_rows, length(sp_rows))
    sizehint!(out_cols, length(sp_cols))
    sizehint!(out_vals, length(sp_vals))

    for k in eachindex(sp_vals)
        r = sp_rows[k]
        c = sp_cols[k]
        if r != c
            push!(out_rows, r)
            push!(out_cols, c)
            push!(out_vals, sp_vals[k])
        end
    end

    diag_len = min(m, length(A.diag))
    for i in 1:diag_len
        d = A.diag[i]
        if !iszero(d)
            ii = Ti(i)
            push!(out_rows, ii)
            push!(out_cols, ii)
            push!(out_vals, Tv(d))
        end
    end

    return sparse(out_rows, out_cols, out_vals, m, n)
end

function Base.summary(io::IO, A::UndirectedAdjacency)
    m, n = size(A)
    shown = _sparse_with_diag(A)
    xnnz = nnz(shown)
    print(io, m, "×", n, " ", typeof(A), " with ", xnnz, " stored ",
        xnnz == 1 ? "entry" : "entries")
    if separate_diagonal(A)
        print(io, " (diagonal stored separately)")
    end
end

function Base.show(io::IO, A::UndirectedAdjacency)
    shown = _sparse_with_diag(A)
    print(io, "UndirectedAdjacency(")
    show(io, shown)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", A::UndirectedAdjacency)
    shown = _sparse_with_diag(A)
    summary(io, A)
    print(io, ":\n")
    Base.print_array(io, shown)
end

function Base.copy(A::UndirectedAdjacency)
    indexmap = fastwrite(A) ? copy(A.indexmap) : nothing
    diag = separate_diagonal(A) ? copy(A.diag) : nothing
    UndirectedAdjacency(copy(A.sp), diag, indexmap)
end
