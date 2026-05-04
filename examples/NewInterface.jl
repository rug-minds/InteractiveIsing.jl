using InteractiveIsing
using InteractiveIsing.Windows

function isingweights(; dr)
    dr == 1 ? 1f0 : 0f0
end

wg = @WG isingweights NN = 1

g = IsingGraph( Layer(64, 64, Continuous(), wg),
    Layer(64, 64, Continuous(), wg), 
    Ising(b = [1f0 for _ in 1:64*64*2]) + Quartic();
)

createProcess(g)
host = interface(g)
sim = host[:simulation]
hp = sim.children[:hamiltonian_parameters]
hp[:entries]
