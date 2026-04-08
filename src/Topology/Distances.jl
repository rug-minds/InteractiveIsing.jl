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
        total += (top.ds[i]*d)^2
    end
    return total
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
