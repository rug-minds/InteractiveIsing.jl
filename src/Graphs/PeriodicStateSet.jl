# AI Generated
export PeriodicStateSet, AngleStateSet, is_periodic_stateset, wrap_to_stateset

"""
    PeriodicStateSet(lower, upper)

Mark a continuous `StateSet` interval as periodic, so values outside the
interval wrap around instead of reflecting or clamping at the endpoints.
"""
struct PeriodicStateSet{S}
    states::S

    function PeriodicStateSet(states::S) where {S<:Tuple}
        length(states) == 2 ||
            throw(ArgumentError("PeriodicStateSet expects exactly two endpoints; got $(length(states))."))
        first(states) < last(states) ||
            throw(ArgumentError("PeriodicStateSet endpoints must satisfy lower < upper; got $(states)."))
        return new{S}(states)
    end
end

"""
    PeriodicStateSet(lower, upper)

Construct a periodic state interval from explicit lower and upper bounds.
"""
function PeriodicStateSet(lower::L, upper::U) where {L<:Real,U<:Real}
    return PeriodicStateSet((lower, upper))
end

"""
    AngleStateSet()

Construct the default angular state interval `[0, 2pi)`.
"""
function AngleStateSet()
    return PeriodicStateSet(0.0, 2 * pi)
end

"""
    AngleStateSet(lower, upper)

Construct a periodic angular state interval with custom bounds.
"""
function AngleStateSet(lower::L, upper::U) where {L<:Real,U<:Real}
    return PeriodicStateSet(lower, upper)
end

"""
    is_periodic_stateset(stateset)

Return whether a layer state set should wrap values periodically.
"""
@inline is_periodic_stateset(stateset::S) where {S} = false

"""
    is_periodic_stateset(stateset::PeriodicStateSet)

Return `true` for periodic state-set markers.
"""
@inline is_periodic_stateset(stateset::PeriodicStateSet{S}) where {S} = true

"""
    wrap_to_stateset(x, stateset)

Wrap `x` into the half-open periodic interval `[first(stateset), last(stateset))`.
"""
@inline function wrap_to_stateset(x::X, stateset::PeriodicStateSet{S}) where {X<:Real,S}
    lo = first(stateset)
    width = last(stateset) - lo
    return lo + mod(x - lo, width)
end

"""
    convert_stateset(T, stateset)

Convert an ordinary state-set container to graph storage precision.
"""
@inline function convert_stateset(::Type{T}, stateset::S) where {T,S}
    return convert.(T, stateset)
end

"""
    convert_stateset(T, stateset::PeriodicStateSet)

Convert a periodic state set while preserving its periodic marker.
"""
@inline function convert_stateset(::Type{T}, stateset::PeriodicStateSet{S}) where {T,S}
    return PeriodicStateSet(convert_stateset(T, stateset.states))
end

"""
    first(stateset::PeriodicStateSet)

Return the lower endpoint of a periodic state interval.
"""
@inline Base.first(stateset::PeriodicStateSet{S}) where {S} = first(stateset.states)

"""
    last(stateset::PeriodicStateSet)

Return the upper endpoint of a periodic state interval.
"""
@inline Base.last(stateset::PeriodicStateSet{S}) where {S} = last(stateset.states)

"""
    length(stateset::PeriodicStateSet)

Return the number of stored endpoints.
"""
@inline Base.length(stateset::PeriodicStateSet{S}) where {S} = length(stateset.states)

"""
    getindex(stateset::PeriodicStateSet, idx)

Return one endpoint by index.
"""
@inline Base.getindex(stateset::PeriodicStateSet{S}, idx::I) where {S,I<:Integer} = stateset.states[idx]

"""
    firstindex(stateset::PeriodicStateSet)

Return the first valid endpoint index.
"""
@inline Base.firstindex(stateset::PeriodicStateSet{S}) where {S} = firstindex(stateset.states)

"""
    lastindex(stateset::PeriodicStateSet)

Return the last valid endpoint index.
"""
@inline Base.lastindex(stateset::PeriodicStateSet{S}) where {S} = lastindex(stateset.states)

"""
    iterate(stateset::PeriodicStateSet[, state])

Iterate over the stored endpoints so existing StateSet code can collect bounds.
"""
@inline Base.iterate(stateset::PeriodicStateSet{S}) where {S} = iterate(stateset.states)
@inline Base.iterate(stateset::PeriodicStateSet{S}, state::I) where {S,I} = iterate(stateset.states, state)

"""
    eltype(stateset::PeriodicStateSet)

Return the endpoint element type.
"""
@inline Base.eltype(stateset::PeriodicStateSet{S}) where {S} = eltype(stateset.states)
@inline Base.eltype(::Type{PeriodicStateSet{S}}) where {S} = eltype(S)

"""
    show(io, stateset::PeriodicStateSet)

Display the periodic marker with its stored endpoints.
"""
function Base.show(io::IO, stateset::PeriodicStateSet{S}) where {S}
    print(io, "PeriodicStateSet", stateset.states)
end
