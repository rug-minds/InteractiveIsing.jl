using InteractiveIsing
import InteractiveIsing as ii

wg = ii.@WG dr -> dr == 1 ? 1 : 0 NN=1

g3d = ii.IsingGraph(10,10,10, type = Discrete, periodic = true)
genAdj!(g3d[1], wg) 

createProcess(g3d)
interface(g3d)