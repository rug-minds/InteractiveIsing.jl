export CI
const CI = CartesianIndex

dist(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = sqrt(dist2(top, ci1, ci2))
dist2(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = dist2(top, Coordinate(top, ci1), Coordinate(top, ci2))    

"""
    dist(top, c1..., c2...)

Calculate metric distance from two coordinate tuples passed as positional
integers, e.g. `dist(top, i1, j1, i2, j2)`.
"""
function dist(top::T, coords::Vararg{I,N}) where {U,D,T<:AbstractLayerTopology{U,D},I<:Integer,N}
    @assert N == 2D "dist(top, coords...) needs two $D-dimensional coordinates"
    c1 = Coordinate(top, ntuple(i -> coords[i], Val(D)))
    c2 = Coordinate(top, ntuple(i -> coords[D + i], Val(D)))
    return dist(top, c1, c2)
end

"""
    dist2(top, c1..., c2...)

Calculate squared metric distance from two coordinate tuples passed as
positional integers.
"""
function dist2(top::T, coords::Vararg{I,N}) where {U,D,T<:AbstractLayerTopology{U,D},I<:Integer,N}
    @assert N == 2D "dist2(top, coords...) needs two $D-dimensional coordinates"
    c1 = Coordinate(top, ntuple(i -> coords[i], Val(D)))
    c2 = Coordinate(top, ntuple(i -> coords[D + i], Val(D)))
    return dist2(top, c1, c2)
end

dist(c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate} =
    throw(ArgumentError("dist(c1, c2) requires topology context. Use dist(topology, c1, c2)."))
dist2(c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate} =
    throw(ArgumentError("dist2(c1, c2) requires topology context. Use dist2(topology, c1, c2)."))

dist(c1::WoorldCoordinate{D}, c2::WoorldCoordinate{D}) where D = sqrt(dist2(c1, c2))
dist2(c1::WoorldCoordinate{D}, c2::WoorldCoordinate{D}) where D =
    sum((c1[i] - c2[i])^2 for i in 1:D)

dist(top::SquareTopology, c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate} = sqrt(dist2(top, c1, c2))
function dist2(top::SquareTopology, c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate}
    ps = whichperiodic(top)
    total = 0.
    for (i, isperiodic) in enumerate(ps)
        d = c2[i]-c1[i]
        if isperiodic
            halfsize = div(size(top)[i], 2)
            if abs(d) > halfsize
                d -= sign(d) * size(top)[i]
            end
        end
        total += (lattice_constants(top)[i]*d)^2
    end
    return total
end

"""
    delta_distance(top, dc)

Calculate the metric distance represented by a coordinate-space delta on `top`.
"""
@inline delta_distance(top::T, dc::DeltaCoordinate{D}) where {D,T<:AbstractLayerTopology} =
    sqrt(delta_distance2(top, dc))

"""
    delta_distance2(top, dc)

Calculate the squared metric distance represented by a coordinate-space delta on
any topology that exposes orthogonal `lattice_constants`.
"""
function delta_distance2(top::T, dc::DeltaCoordinate{D}) where {D,T<:AbstractLayerTopology}
    total = 0.0
    constants = lattice_constants(top)
    @inbounds for i in 1:D
        d = constants[i] * dc[i]
        total += d * d
    end
    return total
end

"""
    delta_distance2(top, dc)

Calculate squared metric distance from a coordinate delta on a
translation-invariant general lattice topology.
"""
function delta_distance2(top::T, dc::DeltaCoordinate{D}) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P}}
    is_translation_invariant(top) ||
        throw(ArgumentError("delta_distance2 for staggered LatticeTopology layouts requires an anchor coordinate. Use dist(top, c1, c2)."))

    offset = zero(origin(top))
    @inbounds for i in 1:D
        offset += P(dc[i]) * primitive_vectors(top)[i]
    end
    return sum(abs2, offset)
end

"""
    delta_distance2(top, dc)

Calculate the squared metric distance represented by a coordinate-space delta on
an orthogonal topology.
"""
function delta_distance2(top::T, dc::DeltaCoordinate{D}) where {D,T<:SquareTopology}
    total = 0.0
    @inbounds for i in 1:D
        d = lattice_constants(top)[i] * dc[i]
        total += d * d
    end
    return total
end

"""
    delta_distance2(top, dc)

Calculate the squared metric distance represented by an axial-coordinate delta
on a hexagonal topology.
"""
function delta_distance2(top::T, dc::DeltaCoordinate{2}) where {T<:HexagonalTopology}
    a1, a2 = lattice_constants(top)
    q = dc[1]
    r = dc[2]
    return a1 * a1 * q * q + a1 * a2 * q * r + a2 * a2 * r * r
end

"""
    dist(top, c1, c2)

Calculate the metric distance between two hexagonal-topology coordinates.
"""
@inline dist(top::T, c1::C1, c2::C2) where {T<:HexagonalTopology,C1<:Coordinate,C2<:Coordinate} =
    sqrt(dist2(top, c1, c2))

"""
    dist2(top, c1, c2)

Calculate the squared minimum-image distance between two coordinates on a
hexagonal topology.
"""
function dist2(top::T, c1::C1, c2::C2) where {T<:HexagonalTopology,C1<:Coordinate,C2<:Coordinate}
    raw_delta = DeltaCoordinate(c2[1] - c1[1], c2[2] - c1[2])
    return delta_distance2(top, top(raw_delta))
end

"""
    dist(top, c1, c2)

Calculate metric distance between two coordinates on a general lattice topology.
"""
@inline dist(top::T, c1::C1, c2::C2) where {T<:LatticeTopology,C1<:Coordinate,C2<:Coordinate} =
    sqrt(dist2(top, c1, c2))

"""
    dist2(top, c1, c2)

Calculate squared minimum-image distance between two coordinates on a general
lattice topology, using actual world positions for staggered layouts.
"""
function dist2(top::T, c1::C1, c2::C2) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P},C1<:Coordinate{D},C2<:Coordinate{D}}
    wc1 = _lattice_world_position(top, c1)
    best = Inf
    ranges = ntuple(i -> periodic(U(), Val(i)) ? (-1:1) : (0:0), Val(D))

    # Search neighboring periodic images; this also handles non-orthogonal bases.
    for shift_ci in CartesianIndices(ranges)
        shifted = Coordinate(top, ntuple(i -> c2[i] + shift_ci[i] * size(top, i), Val(D)); check = false)
        wc2 = _lattice_world_position(top, shifted; wrap_coordinate = false)
        candidate = sum(abs2, wc2 - wc1)
        candidate < best && (best = candidate)
    end
    return best
end


"""
Calculate the squared distance between two coordinates on different topoligies
    Calculate by taking origin + c1 to calculate world coordinates of c1, and origin + c2 to calculate world coordinates of c2, 
    then calculating the distance between those world coordinates. 
"""
function dist2(top1, c1, top2, c2)
    worldcoord1 = woorldcoordinate(top1, c1)
    worldcoord2 = woorldcoordinate(top2, c2)
    return sum((worldcoord1[i] - worldcoord2[i])^2 for i in 1:length(worldcoord1))
end
