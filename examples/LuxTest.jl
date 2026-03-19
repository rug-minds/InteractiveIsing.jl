using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(;dr::R = 1) where R
    return 1/dr
end

wg = @WG (;dr) -> isingfunc(;dr) NN = (3,3,2)


function ReducedBoltzmannArchitecture(layer_sizes...; precision = Float32)
    layer_gen = (Layer(
            layer_sizes[i],
            Continuous(),
            Coords(0, i, 0)) for i in 1:length(layer_sizes))

    
    IsingGraph(layer_gen...,
                Ising() + Clamping();
                iterator = g -> ii.GraphDefectsNew(g, 0))
end

"""
    Create a graph copy, with separate state, but shared data
"""
function GraphFromSource(g::IsingGraph; init! = identity)
    gnew = IsingGraph(
        copy(state(g)),
        adj(g),
        temp(g),
        g.default_algorithm,
        g.hamiltonian,
        g.index_set,
        g.addons,
        g.layers,
    )
    init!(gnew)
    return gnew
end

g = ReducedBoltzmannArchitecture(100, 100, 10)

# interface(g)
createProcess(g, lifetime = Processes.Until(x -> x == 0, Var(g.default_algorithm, :T)))



# pause(g)
# c = context(g)
# reg = getregistry(c)
# cview = view(c, reg[1][1])
# step!(InteractiveIsing.Metropolis(), cview)
# @code_warntype step!(InteractiveIsing.Metropolis(), cview)
