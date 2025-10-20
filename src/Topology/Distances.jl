dist(c1::Coordinate, c2::Coordinate) = sqrt(dist2(c1,c2))
function dist2(c1::Coordinate, c2::Coordinate) 
    @assert c1.top == c2.top "Coordinates must belong to the same topology"
    return dist2(c1.top, c1,c2)
end

function dist2(top::SquareTopology, c1, c2)
    if periodic(top) isa Periodic
        d = c2-c1
        gen = (abs(d[i]) > size(top, 1)/2 ? d[i] - sign(d[i]) * size(top, 1) : d[i] for i in 1:length(c1))
        return reduce((x,y) -> x + y^2, gen, init = 0)
    else
        return reduce((x,y) -> x + y^2, c2-c1, init = 0)
    end   
end
