using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1 / dr
end

wg = @WG (dr, c1, c2) -> isingfunc(dr, c1, c2) NN = (3, 3, 2)

g = ii.IsingGraph(
    100,
    100,
    10,
    Continuous(),
    wg,
    LatticeConstants(1.0, 1.0, 20.0),
    StateSet(-1.5f0, 1.5f0),
    Ising(b = :homogeneous) + Clamping(1f0) + Quartic() + Sextic(),
    periodic = (:x, :y),
    self = :homogeneous,
)

interface(g)
createProcess(g, KineticMC(); dynamics = KineticMC())
