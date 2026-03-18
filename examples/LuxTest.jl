using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(;dr::R = 1) where R
    return 1/dr
end

wg = @WG (;dr) -> isingfunc(;dr) NN = (3,3,2)

# @benchmark g = ii.IsingGraph(100,100,10, 
#         Continuous(), 
#         wg, 
#         LatticeConstants(1.0, 1.0, 20.),
#         StateSet(-1.5f0, 1.5f0),
#         Ising(c = ConstVal(1)) + 
#             Clamping(1f0)+ Quartic(c = ConstVal(1.0), ) + 
#             Sextic(c = ConstVal(1.0), localpotential = StateLike(OffsetArray, 0)),
#         periodic = (:x,:y))

function ReducedBoltzmannArchitecture(layer_sizes...; precision = Float32)
    layer_gen = (Layer(
            layer_sizes[i],
            Continuous(),
            Coords(0, i, 0)) for i in 1:length(layer_sizes))

    
    IsingGraph(layer_gen...,
                Ising() + Clamping();
                iterator = g -> ii.GraphDefectsNew(g, 0), # 50% defects
                # callback! = g -> begin
                #     # set first layer defect since it's the input layer
                #     setClamp!(g[1], zero(precision))
                # end
                )
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
        g.index_set,
        g.addons,
        g.layers,
        wg
    )
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
