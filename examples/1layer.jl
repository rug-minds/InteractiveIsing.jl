using InteractiveIsing
g = IsingGraph(400,400, type = Discrete)

simulate(g)

wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(g[1], wg)
