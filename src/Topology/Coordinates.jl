
export Coordinate, DeltaCoordinate, coord, norm2, delta

struct Coordinate{N} <: Base.AbstractCartesianIndex{N} # N dimensional coordinate
    coords::CartesianIndex{N}
end

Coordinate(t::Tuple) = Coordinate(CartesianIndex(t))

@inline function Coordinate(top::AbstractLayerTopology{U,D}, n::Vararg{Integer,D}; check = true) where {U,D}
    s = size(top)
    p = whichperiodic(top)
    new_coords = ntuple(Val(D)) do i
        ni = n[i]
        p[i] ? mod1(ni, s[i]) : ni
    end
    if check
        @inbounds for i in 1:D
            @assert 0 < new_coords[i] <= s[i] "Coordinate $n is out of bounds for topology of size $(size(top))"
        end
    end
    return Coordinate(CartesianIndex(new_coords))
end

Coordinate(top, t::Tuple; check = true) = Coordinate(top, t...; check)
Coordinate(top, ci::CartesianIndex; check = true) = Coordinate(top, ci.I...; check)

@generated function Coordinate(top::AbstractLayerTopology{U,D}, i::Int) where {U,D}
    coord_exprs = Vector{Any}(undef, D)
    for d in 1:D # Inline the coordinate calculation for each dimension
        stride_expr = d == 1 ? :(1) : :((*(s[1:$(d-1)]...)))
        coord_exprs[d] = :((((idx0 ÷ $stride_expr) % s[$d]) + 1))
    end
    return quote
        s = size(top)
        total = *(s...)
        @assert 1 <= i <= total "Index $i is out of bounds for topology of size $(size(top))"
        idx0 = i - 1
        return Coordinate(top, $(coord_exprs...); check = false)
    end
end

Base.convert(::Type{<:CartesianIndex}, c::Coordinate) = c.coords

offset(top::AbstractLayerTopology, c::Coordinate, deltas...; check = false) =
    Coordinate(top, ntuple(Val(length(c))) do i
        c.coords[i] + deltas[i]
    end...; check)

Base.:(-)(c1::Coordinate, c2::Coordinate) = delta(c1, c2)
Base.abs(c::Coordinate) = Coordinate(abs.(c.coords))

Base.getindex(c::Coordinate, i::Int) = c.coords[i]
Base.iterate(c::Coordinate, state = 1) = iterate(c.coords, state)
Base.length(c::Coordinate{N}) where N = N
Base.size(c::Coordinate{N}) where N = (length(c),)

delta(c1::Coordinate, c2::Coordinate) = DeltaCoordinate(ntuple(i -> c2.coords[i] - c1.coords[i], length(c1.coords)))
delta(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) =
    DeltaCoordinate(ntuple(i -> ci2[i] - ci1[i], length(ci1)))

coord(top, x...) = Coordinate(top, x...)

struct DeltaCoordinate{N} <: AbstractVector{Int} # N dimensional coordinate difference
    deltas::NTuple{N, Int}
end

DeltaCoordinate(ds::NTuple{N, <:Integer}) where {N} = DeltaCoordinate{N}(ntuple(i -> Int(ds[i]), Val(N)))
DeltaCoordinate(ds::Vararg{Integer, N}) where {N} = DeltaCoordinate(ntuple(i -> Int(ds[i]), Val(N)))
DeltaCoordinate(::Coordinate, ds) = DeltaCoordinate(ds)

Base.getindex(dc::DeltaCoordinate, i::Int) = dc.deltas[i]
Base.iterate(dc::DeltaCoordinate, state = 1) = iterate(dc.deltas, state)
Base.length(dc::DeltaCoordinate{N}) where N = N
Base.size(dc::DeltaCoordinate{N}) where N = (N,)
Base.size(dc::DeltaCoordinate, i::Int) = size(dc)[i]

LinearAlgebra.norm(dc::DeltaCoordinate) = sqrt(norm2(dc))
norm2(dc::DeltaCoordinate) = sum(x -> x^2, dc.deltas)
function Base.:^(dc::DeltaCoordinate, p::Integer)
    return sum(x -> x^p, dc.deltas)
end

function wrap(c::Coordinate, top::AbstractLayerTopology)
    s = size(top)
    p = whichperiodic(top)
    new_coords = ntuple(Val(length(c))) do i
        p[i] ? mod1(c.coords[i], s[i]) : c.coords[i]
    end
    return Coordinate(CartesianIndex(new_coords))
end
