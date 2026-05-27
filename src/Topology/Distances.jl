export CI
const CI = CartesianIndex

dist(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = sqrt(dist2(top, ci1, ci2))
dist2(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = dist2(top, Coordinate(top, ci1), Coordinate(top, ci2))    

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
Calculate the squared distance between two coordinates on different topoligies
    Calculate by taking origin + c1 to calculate world coordinates of c1, and origin + c2 to calculate world coordinates of c2, 
    then calculating the distance between those world coordinates. 
"""
function dist2(top1, c1, top2, c2)
    worldcoord1 = woorldcoordinate(top1, c1)
    worldcoord2 = woorldcoordinate(top2, c2)
    return sum((worldcoord1[i] - worldcoord2[i])^2 for i in 1:length(worldcoord1))
end
