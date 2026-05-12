using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g = ii.IsingGraph(30,30, type = Continuous, periodic = (:x,:y))
setdist!(g, (2.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
genAdj!(g, wg)


createProcess(g)
# interface(g)