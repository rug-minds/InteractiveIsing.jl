using InteractiveIsing, JET
import InteractiveIsing as ii

# function isingfunc(dr)
#     dr == 1 ?  1.0 : 0.
# end

isingfunc(dr) = 1/dr^2

g = IsingGraph(500,500, type = Continuous, periodic = true)
wg = @WG isingfunc NN=3
genAdj!(g[1], wg)

createProcess(g)
w = interface(g)