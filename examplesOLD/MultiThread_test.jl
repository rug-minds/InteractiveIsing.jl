using InteractiveIsing, Processes, Preferences

wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1

g = IsingGraph(500,500, type = Discrete, weights = wg)

temp(g, 20)
interface(g)

[createProcess(g) for _ in 1:8]