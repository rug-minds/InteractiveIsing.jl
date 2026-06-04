using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.StatefulAlgorithms
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end

wg = @WG (;dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
g = ii.IsingGraph(30,30, Continuous(), wg, periodic = (:x,:y))

createProcess(g)
# interface(g)