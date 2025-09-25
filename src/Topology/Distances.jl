function dist2(top::SquareTopology, dc::DeltaCoordinate)
    if periodic(top) isa Periodic
        gen = (abs(dc[i]) > size(top, 1)/2 ? dc[i] - sign(dc[i]) * size(top, 1) : dc[i] for i in 1:length(dc))
        return reduce((x,y) -> x + y^2, gen, init = 0)
    else
        return reduce((x,y) -> x + y^2, dc, init = 0)
    end
   
end
dist2(top::LayerTopology, c1::Coordinate, c2::Coordinate) = dist2(top, c2 - c1)

dist(top::LayerTopology, dc::DeltaCoordinate) = sqrt(dist2(top, dc))
dist(top::LayerTopology, c1::Coordinate, c2::Coordinate) = dist(top, c2 - c1)
