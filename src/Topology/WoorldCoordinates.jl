export WoorldCoordinate, woorldcoordinate

struct WoorldCoordinate{N,T} <: AbstractVector{T}
    coords::SVector{N,T}
end

WoorldCoordinate(coords::NTuple{N,T}) where {N,T<:Real} = WoorldCoordinate(SVector{N,T}(coords))

Base.getindex(c::WoorldCoordinate, i::Int) = c.coords[i]
Base.iterate(c::WoorldCoordinate, state = 1) = iterate(c.coords, state)
Base.length(c::WoorldCoordinate{N}) where N = N
Base.size(c::WoorldCoordinate{N}) where N = (N,)

@inline function woorldcoordinate(top::SquareTopology{U,D,P}, coord::Coordinate{D}) where {U,D,P}
    wrapped = wrap(top, coord)
    WoorldCoordinate(ntuple(i -> origin(top)[i] + lattice_constants(top)[i] * wrapped[i], Val(D)))
end

"""
    woorldcoordinate(top, coord)

Convert an axial hexagonal coordinate into the shared world-coordinate space.
"""
@inline function woorldcoordinate(top::T, coord::Coordinate{2}) where {T<:HexagonalTopology}
    wrapped = wrap(top, coord)
    v1, v2 = primitive_vectors(top)
    return WoorldCoordinate(origin(top) + wrapped[1] * v1 + wrapped[2] * v2)
end

"""
    woorldcoordinate(top, coord)

Convert a general lattice coordinate into world space using the topology layout.
"""
@inline function woorldcoordinate(top::T, coord::Coordinate{D}) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P}}
    return WoorldCoordinate(_lattice_world_position(top, coord))
end

@inline woorldcoordinate(top::AbstractLayerTopology{U,D}, coord::CartesianIndex{D}) where {U,D} =
    woorldcoordinate(top, Coordinate(top, coord))

worldcoordinate(top::AbstractLayerTopology{U,D}, coord::Coordinate{D}) where {U,D} =
    woorldcoordinate(top, coord)
worldcoordinate(top::AbstractLayerTopology{U,D}, coord::CartesianIndex{D}) where {U,D} =
    woorldcoordinate(top, coord)
