
struct Coordinate{N} <: AbstractVector{Int} # N dimensional coordinate
    coords::NTuple{N, Int}
end

Coordinate(n::Integer...) = Coordinate{length(n)}(n)
Coodrinate(t::Tuple) = Coordinate(t...)
Base.getindex(c::Coordinate, i::Int) = c.coords[i]
Base.iterate(c::Coordinate, state = 1) = iterate(c.coords, state)

struct DeltaCoordinate{N} <: AbstractVector{Int} # N dimensional coordinate difference
    deltas::NTuple{N, Int}
    norm2::Float64
    norm::Float64
end

Base.getindex(dc::DeltaCoordinate, i::Int) = dc.deltas[i]
Base.iterate(dc::DeltaCoordinate, state = 1) = iterate(dc.deltas, state)

Base.:(-)(c1::Coordinate, c2::Coordinate) = DeltaCoordinate(ntuple(i->c2.coords[i]-c1.coords[i], Val(length(c1.coords)))...)

DeltaCoordinate(n::Integer...; norm = nothing, norm2 = nothing) = DeltaCoordinate{length(n)}(n, isnothing(norm2) ? sum(x->x^2, n) : norm2, isnothing(norm) ? sqrt(sum(x->x^2, n)) : norm)
LinearAlgebra.norm(dc::DeltaCoordinate) = dc.norm
norm2(dc::DeltaCoordinate) = dc.norm2
Base.length(dc::DeltaCoordinate) = length(dc.deltas)

export norm, norm2
