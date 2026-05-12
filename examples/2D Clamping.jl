using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr::R) where R
    return dr == 1 ? 1 : 0
end


g = ii.IsingGraph(30,30, 
    Ising() + Clamping(),
    Continuous(), 
    @WG((;dr) -> isingfunc(dr), NN=2),
    LatticeConstants(1.0, 1.0),
    periodic = (:x,:y))

interface(g)
createProcess(g)
# interface(g)