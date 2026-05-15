using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return dr == 1 ? 1.0f0 : 0.0f0
end

wg = @WG (; dr, c1, c2) -> isingfunc(dr, c1, c2) NN = 1

g = ii.IsingGraph(
    32,
    32,
    10,
    Discrete(),
    ii.Bilinear(),
    wg;
    precision = Float32,
)

temp!(g, 2.0f0)

interface(g)
createProcess(g, KineticMC())
