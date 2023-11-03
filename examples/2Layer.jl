using InteractiveIsing

weight_generator = @WG "(dr) -> 1/dr" NN = 1

layer_connections = @WG "(dr, dx, dy) -> 1" NN = 0

g = simulate(500,500, continuous = true)

addLayer!(g, 250, 250, type = Discrete)
genAdj!(g[1], weight_generator)

genAdj!(g[2], weight_generator)

genAdj!(g[1], g[2], layer_connections)

