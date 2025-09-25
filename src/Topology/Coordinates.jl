
export Coordinate, DeltaCoordinate, coord, norm2, delta

struct Coordinate{N,T} <: AbstractVector{Int} # N dimensional coordinate
    coords::NTuple{N, Int}
    top::T
end

Coordinate(n::Integer...; top = SquareTopology(periodic = false)) = Coordinate{length(n), typeof(top)}(n, top)
Coordinate(t::Tuple) = Coordinate(t...)
Base.:(-)(c1::Coordinate, c2::Coordinate) = Coordinate(ntuple(i->c2.coords[i]-c1.coords[i], length(c1.coords))...)
top(c::Coordinate) = c.top
Base.abs(c::Coordinate) = Coordinate(abs.(c.coords)...; top = c.top)

Base.getindex(c::Coordinate, i::Int) = c.coords[i]
Base.iterate(c::Coordinate, state = 1) = iterate(c.coords, state)
Base.length(c::Coordinate{N}) where N = N
Base.size(c::Coordinate{N}) where N = (length(c),)

function delta(c1, c2)
    @assert c1.top == c2.top "Coordinates must belong to the same topology"
    DeltaCoordinate(abs(c2-c1)..., norm2=dist2(c1,c2), norm=dist(c1,c2), top = c1.top)
end

coord(x...) = Coordinate(x...)

struct DeltaCoordinate{N,T} <: AbstractVector{Int} # N dimensional coordinate difference
    deltas::NTuple{N, Int}
    norm2::Float64
    norm::Float64
    top::T
end

Base.getindex(dc::DeltaCoordinate, i::Int) = dc.deltas[i]
Base.iterate(dc::DeltaCoordinate, state = 1) = iterate(dc.deltas, state)
Base.length(dc::DeltaCoordinate{N}) where N = N
Base.size(dc::DeltaCoordinate{N}) where N = (length(dc),)

DeltaCoordinate(n::Integer...; norm = nothing, norm2 = nothing, top = SquareTopology(periodic=false)) = DeltaCoordinate{length(n), typeof(top)}(n, isnothing(norm2) ? sum(x->x^2, n) : norm2, isnothing(norm) ? sqrt(sum(x->x^2, n)) : norm, top)
LinearAlgebra.norm(dc::DeltaCoordinate) = dc.norm
norm2(dc::DeltaCoordinate) = dc.norm2

