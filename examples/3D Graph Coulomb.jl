using InteractiveIsing
using InteractiveIsing.Processes

nx = 40
ny = 40
nz = 10

stepsize = 0.1f0
scaling = 1f0
screening = Inf32
recalc = 200

wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

g = IsingGraph(
    nx,
    ny,
    nz,
    Continuous(),
    wg,
    LatticeConstants(1f0, 1f0, 1f0),
    StateSet(-1.5f0, 1.5f0),
    Ising(c = ConstVal(0f0), b = 0) +
        CoulombHamiltonian(; scaling, screening, recalc),
    periodic = (:x, :y),
    precision = Float32,
)

temp!(g, 1f0)

algorithm = LocalLangevin(
    stepsize = stepsize,
    adjusted = true,
)

interface(g)
createProcess(g, algorithm)
