"""
Sparse Adjacency matrix that has fast access for fixed connectivity
    Changing connectivity is slow, but changing weights is fast
    Useful for models where the topology is fixed but weights are learned
    E.g. Ising models with fixed interactions but learnable coupling strengths
"""
struct UndirectedAdjacency{Tv, Ti, SP, D, I} <: AbstractSparseMatrix{Tv, Ti}
    sp::SP
    diag::D
    # indexmap::Dict{Pair{Ti, Ti}, Ti}
    indexmap::I
end

fastwrite(::Union{UndirectedAdjacency{Tv, Ti, SP, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, SP, D, I}}}) where {Tv, Ti, SP, D, I} = !(I <: Nothing)
separate_diagonal(::Union{UndirectedAdjacency{Tv, Ti, SP, D, I}, Type{<:UndirectedAdjacency{Tv, Ti, SP, D, I}}}) where {Tv, Ti, SP, D, I} = !(D <: Nothing)

function UndirectedAdjacency(sp::AbstractSparseMatrix{Tv, Ti}, diag::D = nothing; fastwrite = false) where {Tv, Ti, D}
    indexmap = fastwrite ? map_topology(sp) : nothing
    return UndirectedAdjacency{Tv, Ti, typeof(sp), D, typeof(indexmap)}(sp, diag, indexmap)
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

@inline function Base.getindex(A::UndirectedAdjacency, row::Ti, col::Ti) where {Ti <: Integer}
    if separate_diagonal(A) && row == col
        return A.diag[row]
    end
    if fastwrite(A)
        if haskey(A.indexmap, row => col)
            idx = A.indexmap[row => col]
            return A.sp.nzval[idx]
        else
            return zero(eltype(A.sp.nzval)) # Return zero for non-existent edges due to the sparse matrix structure
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

    # Keep only non-diagonal sparse entries from `sp`; diagonal is sourced from `diag`.
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

    # Inject separated diagonal only for selected (row, col) pairs where global idx matches.
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

# ---- Slice setindex! ----
# Handles scalar and vector values for column/row slices.
# separate_diagonal / fastwrite are compile-time constants → branches eliminated by JIT.

@inline function Base.setindex!(A::UndirectedAdjacency, val, rows::AbstractVector{<:Integer}, col::Integer)
    _set_slice!(A, val, rows, Int(col))
end
@inline function Base.setindex!(A::UndirectedAdjacency, val, ::Colon, col::Integer)
    _set_slice!(A, val, axes(A, 1), Int(col))
end
# Symmetric: row slice mirrors column slice but indexing into val is by col position
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

# Matrix slice: A[rows, cols] = val
@inline function Base.setindex!(A::UndirectedAdjacency, val, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    @inbounds for (j, col) in enumerate(cols)
        for (i, row) in enumerate(rows)
            v = val isa AbstractArray ? val[i, j] : val
            A[row, col] = v
        end
    end
end

# ---- Broadcastable views (for .= syntax) ----

struct UAView{Tv, P<:UndirectedAdjacency{Tv}, R, C} <: AbstractArray{Tv, 2}
    parent::P
    rows::R  # AbstractVector or Int
    cols::C  # AbstractVector or Int
end

@inline _resolve(inds::AbstractVector, i) = @inbounds inds[i]
@inline _resolve(fixed::Integer, _) = fixed
@inline _viewsize(inds::AbstractVector) = length(inds)
@inline _viewsize(::Integer) = 1

Base.size(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}) = (_viewsize(v.rows), _viewsize(v.cols))
Base.size(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}) = (_viewsize(v.cols),)
Base.size(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}) = (_viewsize(v.rows),)

Base.IndexStyle(::Type{<:UAView}) = IndexCartesian()

# 2D indexing (matrix view)
@inline Base.getindex(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}, i::Integer, j::Integer) =
    @inbounds v.parent[_resolve(v.rows, i), _resolve(v.cols, j)]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:AbstractVector, <:AbstractVector}, val, i::Integer, j::Integer) =
    @inbounds (v.parent[_resolve(v.rows, i), _resolve(v.cols, j)] = val)

# 1D indexing (vector views — one dim is scalar)
@inline Base.getindex(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}, i::Integer) =
    @inbounds v.parent[v.rows, _resolve(v.cols, i)]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:Integer, <:AbstractVector}, val, i::Integer) =
    @inbounds (v.parent[v.rows, _resolve(v.cols, i)] = val)

@inline Base.getindex(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}, i::Integer) =
    @inbounds v.parent[_resolve(v.rows, i), v.cols]
@inline Base.setindex!(v::UAView{<:Any, <:Any, <:AbstractVector, <:Integer}, val, i::Integer) =
    @inbounds (v.parent[_resolve(v.rows, i), v.cols] = val)

# view constructors
Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = UAView(A, rows, cols)
Base.view(A::UndirectedAdjacency, ::Colon, cols::AbstractVector{<:Integer}) = UAView(A, axes(A, 1), cols)
Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, ::Colon) = UAView(A, rows, axes(A, 2))
Base.view(A::UndirectedAdjacency, ::Colon, ::Colon) = UAView(A, axes(A, 1), axes(A, 2))
Base.view(A::UndirectedAdjacency, row::Integer, cols::AbstractVector{<:Integer}) = UAView(A, Int(row), cols)
Base.view(A::UndirectedAdjacency, rows::AbstractVector{<:Integer}, col::Integer) = UAView(A, rows, Int(col))
Base.view(A::UndirectedAdjacency, row::Integer, ::Colon) = UAView(A, Int(row), axes(A, 2))
Base.view(A::UndirectedAdjacency, ::Colon, col::Integer) = UAView(A, axes(A, 1), Int(col))

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
SparseArrays.findnz(A::UndirectedAdjacency) = findnz(A.sp)
Base.transpose(A::UndirectedAdjacency) = A # Undirected adjacency is symmetric, so transpose is the same
Base.adjoint(A::UndirectedAdjacency) = A # Undirected adjacency is symmetric, so adjoint is the same

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

    # Keep all off-diagonal sparse entries from sp.
    for k in eachindex(sp_vals)
        r = sp_rows[k]
        c = sp_cols[k]
        if r != c
            push!(out_rows, r)
            push!(out_cols, c)
            push!(out_vals, sp_vals[k])
        end
    end

    # Inject diagonal from separated storage.
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

@inline function weighted_neighbors_sum(node_idx, adj::UA, nodevals::AV; transform::F = identity, transform_weight::FW = identity) where {UA <: UndirectedAdjacency, AV<:AbstractArray, F, FW}
    total = @inline column_contraction(node_idx, nodevals, adj.sp; transform, transform_weight)
    if !separate_diagonal(adj) # Subtract self-loop contribution if diagonal is not separate
        self_weight = adj.sp[node_idx, node_idx]
        self_value = @inline getindex(nodevals, node_idx)
        total -= @inline transform_weight(self_weight) * transform(self_value)
    end
    return total
end

@inline function weighted_self(node_idx, adj::UA, nodevals::AV; transform::F = identity, transform_weight::FW = identity) where {UA <: UndirectedAdjacency, AV<:AbstractVector, F, FW}
    self_weight = @inline adj[node_idx, node_idx]
    self_value = @inline getindex(nodevals, node_idx)
    return transform_weight(self_weight) * transform(self_value)
end
