using InteractiveIsing
using InteractiveIsing.Processes

function isingweights(;dr::R) where R
    return dr == 1 ? 1f0 : 0f0
end


wg = @WG isingweights NN= 1

g = IsingGraph(100,100,10, 
        Continuous(), 
        wg, 
        LatticeConstants(1f0, 1f0, 1f0),
        StateSet(-1.5f0, 1.5f0),
        Ising(c = ConstVal(0f0), b = StateLike(ConstFill, 0f0)) +
        CoulombHamiltonian(recalc = 5000),
        periodic = (:x,:y))

interface(g)
createProcess(g, LangevinDynamics())
