using InteractiveIsing

weight_generator = @WG "(dr) -> 1/dr" NN = 1

layer_connections = @WG "(dr, dx, dy) -> let θ = atan(dy/dx); cos(θ) end" NN = 1

g = simulate(500,500, continuous = true)

addLayer!(g, 250, 250)
genAdj!(g[1], weight_generator)

genAdj!(g[2], weight_generator)

genAdj!(g[1], g[2], layer_connections)

