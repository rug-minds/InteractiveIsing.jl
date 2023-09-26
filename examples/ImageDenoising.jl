using InteractiveIsing

g = simulate(500,500, continuous = true)

w_generator = @WG "(dr) -> 1/dr" NN = 1
layer_connections = @WG "() -> 1" NN = 0

addLayer!(g, 500, 500)
genAdj!(g[1],g[2], layer_connections)

clampImg!(g[1], "examples/smileys.jpg")