using InteractiveIsing

g = IsingGraph(architecture = [(28,28), (32,32), (2,5,Discrete)], sets = [(0f0, 1f0), (0f0, 1f0), (0, 1f0)])
simulate(g)