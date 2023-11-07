using InteractiveIsing

weight_generator = @WG "(dr) -> dr == 1 ? 1 : 0" NN = 1

layer_connections = @WG "(dr, dx, dy) -> 1" NN = 1

g = simulate(500,500, continuous = true)
genAdj!(g[1], weight_generator)


addLayer!(g, 250, 250, type = Discrete, set = (0f0, 1f0))
genAdj!(g[1], weight_generator)

genAdj!(g[2], weight_generator)

genAdj!(g[1], g[2], layer_connections)

