using InteractiveIsing
g = IsingGraph(200,200, type = Discrete)

simulate(g)

wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
wg5 = @WG "dr -> 1" NN = 5
wg10 = @WG "dr -> 1" NN = 10
genAdj!(g[1], wg)





