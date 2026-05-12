"""
    GraphDefectsNew{T}

Graph-independent defect store with:
    - a dense vector of alive graph indices
    - O(1) alive membership lookup via inverse positions
    - a single divider vector partitioning the alive set by layer

The alive vector is partitioned by layer, but elements inside each layer block are
unordered. This keeps updates cheap while still allowing each layer to recover its
alive block from a start:stop slice.
"""
mutable struct GraphDefectsNew{T<:Integer}
    alive::Vector{T}
    pos_in_alive::Vector{T}         # 0 means defect
    layer_partition_start::Vector{T}  # length = nlayers + 1
end

export GraphDefectsNew
export aliveindices, layerrange, isdefect, setdefect!, setalive!, setdefects!, clear_defects!

GraphDefectsNew{T}() where {T<:Integer} = GraphDefectsNew{T}(T[], T[], ones(T, 1))

function _partition_start(::Type{T}, layer_lengths) where {T<:Integer}
    starts = T[one(T)]
    running_total = zero(T)
    for len in layer_lengths
        running_total += T(len)
        push!(starts, running_total + one(T))
    end
    return starts
end

function _layer_lengths(g::AbstractIsingGraph)
    map(layer -> length(parentindices(layer)[1]), layers(g))
end

function _layer_lengths(layers_tuple::Tuple)
    map(layer -> length(parentindices(layer)[1]), layers_tuple)
end

function _set_random_defects!(gd::GraphDefectsNew, initial_defects::Integer)
    initial_defects == 0 && return gd
    0 <= initial_defects <= nstates(gd) || error("initial_defects must be between 0 and $(nstates(gd)).")
    for idx in randperm(nstates(gd))[1:initial_defects]
        gd.pos_in_alive[idx] == 0 && continue
        layeridx = searchsortedlast(gd.layer_partition_start, gd.pos_in_alive[idx])
        _setdefect!(gd, layeridx, idx)
    end
    return gd
end

function _set_random_defects!(gd::GraphDefectsNew, initial_defect_fraction::Real)
    0 <= initial_defect_fraction <= 1 || error("initial_defect_fraction must lie in [0, 1].")
    return _set_random_defects!(gd, round(Int, initial_defect_fraction * nstates(gd)))
end

function GraphDefectsNew{T}(layer_lengths::Integer...) where {T<:Integer}
    total_states = sum(layer_lengths)
    alive = T[T(i) for i in 1:total_states]
    pos_in_alive = copy(alive)
    layer_partition_start = _partition_start(T, layer_lengths)

    return GraphDefectsNew{T}(alive, pos_in_alive, layer_partition_start)
end

GraphDefectsNew(layer_lengths::Integer...) = GraphDefectsNew{Int32}(layer_lengths...)

function GraphDefectsNew{T}(layer_ranges::AbstractVector{<:AbstractUnitRange{<:Integer}}) where {T<:Integer}
    isempty(layer_ranges) && return GraphDefectsNew{T}()

    first(first(layer_ranges)) == 1 || error("GraphDefectsNew requires layer ranges to start at 1.")
    layer_lengths = Int[]
    expected_start = 1
    for r in layer_ranges
        first(r) == expected_start || error("GraphDefectsNew requires contiguous ordered layer ranges.")
        push!(layer_lengths, length(r))
        expected_start = last(r) + 1
    end

    return GraphDefectsNew{T}(layer_lengths...)
end

GraphDefectsNew(layer_ranges::AbstractVector{<:AbstractUnitRange{<:Integer}}) =
    GraphDefectsNew{Int32}(layer_ranges)

GraphDefectsNew{T}(g::AbstractIsingGraph) where {T<:Integer} = GraphDefectsNew{T}(_layer_lengths(g)...)
GraphDefectsNew(g::AbstractIsingGraph) = GraphDefectsNew{Int32}(g)
GraphDefectsNew{T}(g::AbstractIsingGraph, initial_defects::Integer) where {T<:Integer} =
    _set_random_defects!(GraphDefectsNew{T}(g), initial_defects)
GraphDefectsNew(g::AbstractIsingGraph, initial_defects::Integer) =
    GraphDefectsNew{Int32}(g, initial_defects)
GraphDefectsNew{T}(g::AbstractIsingGraph, initial_defect_fraction::Real) where {T<:Integer} =
    _set_random_defects!(GraphDefectsNew{T}(g), initial_defect_fraction)
GraphDefectsNew(g::AbstractIsingGraph, initial_defect_fraction::Real) =
    GraphDefectsNew{Int32}(g, initial_defect_fraction)
GraphDefectsNew{T}(layers_tuple::Tuple) where {T<:Integer} = GraphDefectsNew{T}(_layer_lengths(layers_tuple)...)
GraphDefectsNew(layers_tuple::Tuple) = GraphDefectsNew{Int32}(layers_tuple)

function Base.show(io::IO, gd::GraphDefectsNew{T}) where T
    print(io, "GraphDefectsNew{$T}($(nlayers(gd)) layers, $(ndefect(gd)) defects, $(nalive(gd)) alive)")
end

Base.length(gd::GraphDefectsNew) = length(gd.alive)
Base.isempty(gd::GraphDefectsNew) = isempty(gd.alive)
Base.eltype(::Type{GraphDefectsNew{T}}) where T = T
Base.eltype(::GraphDefectsNew{T}) where T = T
Base.iterate(gd::GraphDefectsNew, state...) = iterate(gd.alive, state...)
Base.getindex(gd::GraphDefectsNew, i::Integer) = gd.alive[i]

nlayers(gd::GraphDefectsNew) = length(gd.layer_partition_start) - 1
nstates(gd::GraphDefectsNew) = length(gd.pos_in_alive)
nalive(gd::GraphDefectsNew) = length(gd.alive)
ndefect(gd::GraphDefectsNew) = nstates(gd) - nalive(gd)
hasDefects(gd::GraphDefectsNew) = ndefect(gd) > 0

@inline function _aliveblock(gd::GraphDefectsNew{T}, layeridx::Integer) where T
    start = Int(gd.layer_partition_start[layeridx])
    stop = Int(gd.layer_partition_start[layeridx + 1] - one(T))
    return start:stop
end

aliveindices(gd::GraphDefectsNew) = gd.alive
aliveindices(gd::GraphDefectsNew, layeridx::Integer) = @view gd.alive[_aliveblock(gd, layeridx)]
aliveindices(gd::GraphDefectsNew, layer) = aliveindices(gd, internal_idx(layer))

function nalive(gd::GraphDefectsNew, layeridx::Integer)
    block = _aliveblock(gd, layeridx)
    return isempty(block) ? 0 : length(block)
end

nalive(gd::GraphDefectsNew, layer) = nalive(gd, internal_idx(layer))

layerrange(layer) = parentindices(layer)[1]
layerrange(::GraphDefectsNew, layer) = layerrange(layer)

function ndefect(gd::GraphDefectsNew, layer)
    return length(layerrange(layer)) - nalive(gd, layer)
end

hasDefects(gd::GraphDefectsNew, layer) = ndefect(gd, layer) > 0

function isdefect(gd::GraphDefectsNew, idx::Integer)
    @boundscheck checkbounds(gd.pos_in_alive, idx)
    return iszero(gd.pos_in_alive[idx])
end

isdefect(gd::GraphDefectsNew, layer, idx) = isdefect(gd, _graphidx(layer, idx))

@inline function _graphidx(layer, idx::Integer)
    return parentindices(layer)[1][idx]
end

@inline function _graphidx(layer, idx::CartesianIndex)
    return parentindices(layer)[1][LinearIndices(size(layer))[idx]]
end

@inline _graphidx(layer, idx::Tuple) = _graphidx(layer, CartesianIndex(idx))

@inline function _swap_positions!(alive::Vector{T}, pos_in_alive::Vector{T}, i::Int, j::Int) where T
    i == j && return nothing
    idx_i = alive[i]
    idx_j = alive[j]
    alive[i] = idx_j
    alive[j] = idx_i
    pos_in_alive[idx_i] = T(j)
    pos_in_alive[idx_j] = T(i)
    return nothing
end

function _setdefect!(gd::GraphDefectsNew{T}, layeridx::Integer, gidx::Integer) where T
    @boundscheck checkbounds(gd.pos_in_alive, gidx)
    iszero(gd.pos_in_alive[gidx]) && return false

    pos = Int(gd.pos_in_alive[gidx])
    last_in_layer = Int(gd.layer_partition_start[layeridx + 1] - one(T))

    _swap_positions!(gd.alive, gd.pos_in_alive, pos, last_in_layer)
    cursor = last_in_layer
    for j in (layeridx + 1):nlayers(gd)
        next_last = Int(gd.layer_partition_start[j + 1] - one(T))
        _swap_positions!(gd.alive, gd.pos_in_alive, cursor, next_last)
        cursor = next_last
    end

    pop!(gd.alive)
    gd.pos_in_alive[gidx] = zero(T)

    for j in (layeridx + 1):length(gd.layer_partition_start)
        gd.layer_partition_start[j] -= one(T)
    end
    return true
end

function _setalive!(gd::GraphDefectsNew{T}, layeridx::Integer, gidx::Integer) where T
    @boundscheck checkbounds(gd.pos_in_alive, gidx)
    !iszero(gd.pos_in_alive[gidx]) && return false

    push!(gd.alive, T(gidx))
    gd.pos_in_alive[gidx] = T(length(gd.alive))

    cursor = length(gd.alive)
    for j in nlayers(gd):-1:(layeridx + 1)
        _swap_positions!(gd.alive, gd.pos_in_alive, cursor, Int(gd.layer_partition_start[j]))
        cursor = Int(gd.layer_partition_start[j])
    end

    for j in (layeridx + 1):length(gd.layer_partition_start)
        gd.layer_partition_start[j] += one(T)
    end
    return true
end

setdefect!(gd::GraphDefectsNew, layer, idx) = _setdefect!(gd, internal_idx(layer), _graphidx(layer, idx))
setalive!(gd::GraphDefectsNew, layer, idx) = _setalive!(gd, internal_idx(layer), _graphidx(layer, idx))

function Base.setindex!(gd::GraphDefectsNew, val::Bool, layer, idx)
    val ? setdefect!(gd, layer, idx) : setalive!(gd, layer, idx)
    return val
end

function setdefects!(gd::GraphDefectsNew, layer, idxs)
    changed = 0
    for idx in idxs
        changed += setdefect!(gd, layer, idx)
    end
    return changed
end

function clear_defects!(gd::GraphDefectsNew, layer, idxs)
    changed = 0
    for idx in idxs
        changed += setalive!(gd, layer, idx)
    end
    return changed
end

function _reset!(gd::GraphDefectsNew{T}, layer_lengths) where T
    total = nstates(gd)
    total == sum(layer_lengths) || error("reset! layer lengths do not match GraphDefectsNew state count.")
    resize!(gd.alive, total)
    for i in 1:total
        idx = T(i)
        gd.alive[i] = idx
        gd.pos_in_alive[i] = idx
    end
    gd.layer_partition_start = _partition_start(T, layer_lengths)
    return gd
end

reset!(gd::GraphDefectsNew{T}, layer_lengths::Integer...) where {T<:Integer} = _reset!(gd, layer_lengths)
reset!(gd::GraphDefectsNew{T}, layer_ranges::AbstractVector{<:AbstractUnitRange{<:Integer}}) where {T<:Integer} =
    _reset!(gd, length.(layer_ranges))
reset!(gd::GraphDefectsNew, g::AbstractIsingGraph) = _reset!(gd, _layer_lengths(g))
reset!(gd::GraphDefectsNew, layers_tuple::Tuple) = _reset!(gd, _layer_lengths(layers_tuple))

function Random.rand(rng::AbstractRNG, gd::GraphDefectsNew)
    isempty(gd.alive) && throw(ArgumentError("Cannot sample from an empty alive set."))
    return rand(rng, gd.alive)
end

Random.rand(gd::GraphDefectsNew) = rand(Random.default_rng(), gd)
