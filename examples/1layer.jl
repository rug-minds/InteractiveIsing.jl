using InteractiveIsing
g = IsingGraph(architecture = [(20,20, Continuous), (40,40, Discrete),(60,60)])
simulate(g)
