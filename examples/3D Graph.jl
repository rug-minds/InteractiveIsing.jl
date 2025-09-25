using InteractiveIsing
import InteractiveIsing as ii

wg = ii.@WG (dr,c1,c2) -> 1/dr NN=3

g3d = ii.IsingGraph(10,10,10, type = Discrete, periodic = true)
genAdj!(g3d[1], wg) 

# createProcess(g3d)
# interface(g3d)