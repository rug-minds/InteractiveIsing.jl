using InteractiveIsing

g = simulate(500,500, continuous = true)

w_generator = @WG "(dr) -> 1/dr" NN = 1

genAdj!(g[1], w_generator)

addLayer!(g, 200, 100, w_generator)

layer_w_generator = @WG "(dr) -> dr" NN = 2

genAdj!(g[1], g[2], layer_w_generator)