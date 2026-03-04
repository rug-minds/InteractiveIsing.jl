using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1/dr
end

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=(3,3,2)

g = ii.IsingGraph(100,100,10, 
        Continuous(), 
        wg, 
        LatticeConstants(1.0, 1.0, 20.), 
        periodic = (:x,:y), 
        self = :homogeneous)


# _NNt = ii.getNN(wg, 3)
# nstates = 100000
# blocksize = Int32(prod(2 .* _NNt .+ 1) - 1)
# n_conns = nstates*blocksize


# layer = g[1].data
# precision = Float32
# row_idxs = Int32[]
# col_idxs = Int32[]
# weights = Float32[]
# sizehint!(row_idxs, n_conns)
# sizehint!(col_idxs, n_conns)
# sizehint!(weights, n_conns)
# (begin empty!(row_idxs); empty!(col_idxs); empty!(weights) end)
# topo = ii.top(layer)
# # @code_warntype ii._fillSparseVecs(layer, precision, row_idxs, col_idxs, weights, topo, wg)
# # @benchmark ii._fillSparseVecs(layer, precision, row_idxs, col_idxs, weights, topo, wg) setup = (begin empty!(row_idxs); empty!(col_idxs); empty!(weights) end)
# @benchmark ii._fillSparseVecsNew(layer, precision, row_idxs, col_idxs, weights, topo, wg) setup = (begin empty!(row_idxs); empty!(col_idxs); empty!(weights) end)
# ii._fillSparseVecs(layer, precision, row_idxs, col_idxs, weights, topo, wg)
# ii._fillSparseVecsNew(layer, precision, row_idxs, col_idxs, weights, topo, wg)
# copy_row_idxs = copy(row_idxs)
# copy_col_idxs = copy(col_idxs)
# copy_weights = copy(weights)

interface(g)

struct PolTracker{T} <: ProcessAlgorithm end
PolTracker() = PolTracker{Float32}()
function Processes.init(::PolTracker, context)
    (;isinggraph) = context
    initial = sum(state(isinggraph))

    (;pols = Float32[initial])
end

function Processes.step!(::PolTracker, context)
    (;proposal, pols) = context
    push!(pols, delta(proposal))
    return (;)
end

# g.hamiltonian = h = Ising(g) 
isingmetro = InteractiveIsing.IsingMetropolis()
isingmetro = g.default_algorithm
algo = ii.IsingLangevin()
# g.hamiltonian = h = Ising(g) + CoulombHamiltonian(g, 2f0, screening = 20f0, recalc = 200);
g.hamiltonian = h = Ising(g, b = :homogeneous)
# reinit(g)
# CompositeAlgorithm(isingmetro, PolTracker(), (1, 1),  
#     Share(isingmetro, PolTracker())
#     )

# createProcess(g, algo, dynamics = algo)
createProcess(g)





# createProcess(g, algo, Input(PolTracker(), isinggraph = g))

# using Random
# proposer = ii.get_proposer(g)
# proposal = rand(Random.MersenneTwister(123), proposer)
# hargs = (;s = InteractiveIsing.state(g), w = InteractiveIsing.adj(g), self = InteractiveIsing.self(g), g.hamiltonian...)
# ii.ΔH(g.hamiltonian[3], hargs, proposal)