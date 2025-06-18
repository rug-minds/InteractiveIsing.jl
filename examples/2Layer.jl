using InteractiveIsing, BenchmarkTools
import InteractiveIsing as II

wg1 = @WG "(dr) -> dr == 1 ? 1 : 0" NN = 1
wg2 = @WG "(dr) -> 1/dr" NN = 1

layer_connections = @WG "(dr, dx, dy) -> 1" NN = 3

g = IsingGraph(500, 500, type = Discrete)
# interface(g)
addLayer!(g, 250, 250, type = Continuous)


genAdj!(g[1], wg1)
genAdj!(g[2], wg2)
genAdj!(g[1], g[2], layer_connections)

g.hamiltonian = Ising(g)

simulate(g)

