"""
State index partitioning with static ranges encoded in the type.

This lets us dispatch from a global spin index `j` to a payload (layer, stateset,
parameters, etc.) using generated binary-search branches instead of runtime-linear
search over tuples.
"""
struct StaticStatePartition{Ranges, Payloads}
    payloads::Payloads
end

@inline ranges(::Type{<:StaticStatePartition{Ranges}}) where {Ranges} = Ranges
@inline ranges(sp::StaticStatePartition) = ranges(typeof(sp))
@inline payloads(sp::StaticStatePartition) = sp.payloads
@inline npartitions(::Type{<:StaticStatePartition{Ranges}}) where {Ranges} = length(Ranges)
@inline npartitions(sp::StaticStatePartition) = npartitions(typeof(sp))
Base.length(sp::StaticStatePartition) = npartitions(sp)

function _validate_partition_ranges(ranges::Tuple)
    isempty(ranges) && throw(ArgumentError("StaticStatePartition requires at least one range"))
    prev_last = nothing
    for (i, r) in pairs(ranges)
        r isa AbstractUnitRange || throw(ArgumentError("Partition $i must be an AbstractUnitRange, got $(typeof(r))"))
        first(r) <= last(r) || throw(ArgumentError("Partition $i has invalid range $r"))
        if !isnothing(prev_last)
            first(r) > prev_last || throw(ArgumentError("Partition ranges must be strictly increasing and non-overlapping"))
        end
        prev_last = last(r)
    end
    return ranges
end

function StaticStatePartition(ranges::Tuple, payloads::Tuple)
    _validate_partition_ranges(ranges)
    length(ranges) == length(payloads) || throw(ArgumentError("ranges and payloads must have the same length"))
    return StaticStatePartition{ranges, typeof(payloads)}(payloads)
end

function StaticStatePartition(layers::Tuple)
    rs = ntuple(i -> range(layers[i]), length(layers))
    return StaticStatePartition(rs, layers)
end

StaticStatePartition(g::AbstractIsingGraph) = StaticStatePartition(layers(g))

@generated function partition_index(sp::StaticStatePartition{Ranges}, j::Integer) where {Ranges}
    n = length(Ranges)
    n == 0 && return :(throw(ArgumentError("StaticStatePartition has no ranges")))

    lo_bound = first(Ranges[1])
    hi_bound = last(Ranges[end])

    function build(lo, hi)
        if lo == hi
            r = Ranges[lo]
            rlo = first(r)
            rhi = last(r)
            return quote
                if j < $rlo || j > $rhi
                    throw(BoundsError(sp, j))
                end
                $lo
            end
        end
        mid = (lo + hi) >>> 1
        pivot = last(Ranges[mid])
        return quote
            if j <= $pivot
                $(build(lo, mid))
            else
                $(build(mid + 1, hi))
            end
        end
    end

    return quote
        if j < $lo_bound || j > $hi_bound
            throw(BoundsError(sp, j))
        end
        $(build(1, n))
    end
end

@generated function partition_value(sp::StaticStatePartition{Ranges}, j::Integer) where {Ranges}
    n = length(Ranges)
    n == 0 && return :(throw(ArgumentError("StaticStatePartition has no ranges")))

    lo_bound = first(Ranges[1])
    hi_bound = last(Ranges[end])

    function build(lo, hi)
        if lo == hi
            r = Ranges[lo]
            rlo = first(r)
            rhi = last(r)
            return quote
                if j < $rlo || j > $rhi
                    throw(BoundsError(sp, j))
                end
                getfield(sp, :payloads)[$lo]
            end
        end
        mid = (lo + hi) >>> 1
        pivot = last(Ranges[mid])
        return quote
            if j <= $pivot
                $(build(lo, mid))
            else
                $(build(mid + 1, hi))
            end
        end
    end

    return quote
        if j < $lo_bound || j > $hi_bound
            throw(BoundsError(sp, j))
        end
        $(build(1, n))
    end
end

@generated function partition_dispatch(func_to_dispatch::F, j::Integer, sp::StaticStatePartition{Ranges}, args...) where {F, Ranges}
    n = length(Ranges)
    n == 0 && return :(throw(ArgumentError("StaticStatePartition has no ranges")))

    lo_bound = first(Ranges[1])
    hi_bound = last(Ranges[end])

    function build(lo, hi)
        if lo == hi
            r = Ranges[lo]
            rlo = first(r)
            rhi = last(r)
            return quote
                if j < $rlo || j > $rhi
                    throw(BoundsError(sp, j))
                end
                @inline func_to_dispatch(getfield(sp, :payloads)[$lo], args...)
            end
        end

        mid = (lo + hi) >>> 1
        pivot = last(Ranges[mid])
        return quote
            if j <= $pivot
                $(build(lo, mid))
            else
                $(build(mid + 1, hi))
            end
        end
    end

    return quote
        if j < $lo_bound || j > $hi_bound
            throw(BoundsError(sp, j))
        end
        $(build(1, n))
    end
end

@inline Base.getindex(sp::StaticStatePartition, j::Integer) = partition_value(sp, j)

# Compatibility with existing layer-dispatch utility names.
@inline spin_idx_to_layer_idx(j, sp::StaticStatePartition) = partition_index(sp, j)
@inline spin_idx_layer_dispatch(func_to_dispatch::F, j, sp::StaticStatePartition) where {F} = partition_dispatch(func_to_dispatch, j, sp)
@inline spin_idx_layer_dispatch(func_to_dispatch::F, j, sp::StaticStatePartition, args...) where {F} = partition_dispatch(func_to_dispatch, j, sp, args...)

function Base.show(io::IO, sp::StaticStatePartition)
    print(io, "StaticStatePartition(", npartitions(sp), " partitions; range=", first(ranges(sp)[1]), ":", last(ranges(sp)[end]), ")")
end

export StaticStatePartition, partition_index, partition_value, partition_dispatch
