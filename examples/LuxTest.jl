using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(;dr::R = 1) where R
    return 1/dr
end

wg = @WG (;dr) -> isingfunc(;dr) NN = (3,3,2)

function ReducedBoltzmannArchitecture(layer_sizes...)
    layer_gen = (Layer(layer_sizes[i],
              Continuous(),
              Coords(0, i, 0)) for i in 1:length(layer_sizes))

    
    IsingGraph(layer_gen...,
                Ising() + Clamping())
                # default_algorithm = Langevin())
end

"""
    Create a graph copy, with separate state, but shared data
"""
function GraphFromSource(g::IsingGraph)
    IsingGraph(
        copy(state(g)),
        adj(g),
        temp(g),
        g.default_algorithm,
        g.hamiltonian,
        g.defects,
        g.addons,
        g.layers,
        wg
    )
end

ReducedBoltzmannArchitecture(100, 100, 10)

g = ii.IsingGraph(100,100,10, 
        Continuous(), 
        wg, 
        LatticeConstants(1.0, 1.0, 20.),
        StateSet(-1.5f0, 1.5f0),
        Ising(c = ConstVal(1)) + 
            Clamping(1f0)+ Quartic(c = ConstVal(1.0), ) + 
            Sextic(c = ConstVal(1.0), localpotential = StateLike(OffsetArray, 0)),
        periodic = (:x,:y))
# interface(g)
# createProcess(g, lifetime = Processes.Until(x -> x == 0, Var(g.default_algorithm, :T)))



# pause(g)
# c = context(g)
# reg = getregistry(c)
# cview = view(c, reg[1][1])
# step!(InteractiveIsing.Metropolis(), cview)
# @code_warntype step!(InteractiveIsing.Metropolis(), cview)

