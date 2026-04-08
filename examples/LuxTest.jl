using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(;dr::R = 1) where R
    return 1/dr
end

wg = @WG (;dr) -> isingfunc(;dr) NN = (3,3,2)


function ReducedBoltzmannArchitecture(layer_sizes...; precision = Float32)
    layer_gen = [Layer(
            layer_sizes[i],
            Continuous(),
            Coords(0, i, 0)) for i in 1:length(layer_sizes)]
    
    weight_generators = [AllToAllWeightGenerator() for _ in 1:(length(layer_sizes)-1)]

    layers_and_wgs = Any[layer_gen[1]]
    for i in eachindex(weight_generators)
        push!(layers_and_wgs, weight_generators[i], layer_gen[i + 1])
    end

    
    IsingGraph(layers_and_wgs...,
                Ising() + Clamping();
                index_set = g -> ToggledIndexSet(g))
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

@benchmark ReducedBoltzmannArchitecture(100, 100, 10)

# interface(g)
# createProcess(g, lifetime = Processes.Until(x -> x == 0, Var(g.default_algorithm, :T)))



# pause(g)
# c = context(g)
# reg = getregistry(c)
# cview = view(c, reg[1][1])
# step!(InteractiveIsing.Metropolis(), cview)
# @code_warntype step!(InteractiveIsing.Metropolis(), cview)
