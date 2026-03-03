export CI
const CI = CartesianIndex

dist(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = sqrt(dist2(top, ci1, ci2))
dist2(top::AbstractLayerTopology, ci1::CartesianIndex, ci2::CartesianIndex) = dist2(top, Coordinate(top, ci1), Coordinate(top, ci2))    

dist(c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate} = sqrt(dist2(c1,c2))
function dist2(c1::C1, c2::C2) where {C1<:Coordinate, C2<:Coordinate}
    @assert c1.top == c2.top "Coordinates must belong to the same topology"
    return dist2(c1.top, c1,c2)
end

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
