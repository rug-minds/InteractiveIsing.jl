using InteractiveIsing

wg = @WG "(dr) -> 1/dr" NN = 1

g = simulate(500, 500, weight_generator = wg, periodic = false)

genAdj!(g[1], wg)

