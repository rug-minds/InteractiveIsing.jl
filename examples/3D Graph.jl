using InteractiveIsing
using InteractiveIsing.Processes

function isingweights(;dr::R) where R
    return dr == 1 ? 1f0 : 0f0
end


wg = @WG isingweights NN= 1

g = IsingGraph(100,100,10, 
        Continuous(), 
        LocalProposer(0.5),
        wg, 
        LatticeConstants(1f0, 1f0, 1f0),
        StateSet(-1f0, 1f0),
        Ising(c = ConstVal(0f0), b = 0, localpotential = 0),
        periodic = (:x,:y))

g.addons[:interactive] = true
temp!(g, 0)
interface(g)
createProcess(g)
