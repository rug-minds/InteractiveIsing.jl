using InteractiveIsing

weight_generator = @WG "(dr) -> dr == 1 ? 1 : 0" NN = 1

layer_connections = @WG "(dr, dx, dy) -> 1" NN = 1

g = simulate(500,500, type = Continuous)
genAdj!(g[1], weight_generator)


addLayer!(g, 250, 250, type = Continuous, set = (-1f0, 1f0))
genAdj!(g[1], weight_generator)

genAdj!(g[2], weight_generator)

genAdj!(g[1], g[2], layer_connections)

g.d.bfield .= rand(Float32, length(g.d.bfield))
setSType!(g, :Magfield => true)