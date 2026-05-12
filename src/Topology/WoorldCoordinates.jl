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

@generated function woorldcoordinate(top::LatticeTopology{T,U,D,P}, coord::Coordinate{D}) where {T,U,D,P}
    terms = [:(wrapped[$i] * top[$i]) for i in 1:D]
    sumexp = terms[1]
    for term in terms[2:end]
        sumexp = :($sumexp + $term)
    end
    return quote
        wrapped = wrap(top, coord)
        WoorldCoordinate(origin(top) + $sumexp)
    end
end

@inline woorldcoordinate(top::AbstractLayerTopology{U,D}, coord::CartesianIndex{D}) where {U,D} =
    woorldcoordinate(top, Coordinate(top, coord))

worldcoordinate(top::AbstractLayerTopology{U,D}, coord::Coordinate{D}) where {U,D} =
    woorldcoordinate(top, coord)
worldcoordinate(top::AbstractLayerTopology{U,D}, coord::CartesianIndex{D}) where {U,D} =
    woorldcoordinate(top, coord)
