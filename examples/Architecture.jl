using InteractiveIsing

g = IsingGraph(architecture = [(10,10), (100,30), (10,5,Discrete)], sets = [(0f0, 1f0), (0f0, 1f0), (-1f0, 1f0)])
simulate(g)