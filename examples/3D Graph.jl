using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2*exp(-dr)
end


g = ii.IsingGraph(100,100,10, type = Continuous, periodic = (:x,:y))
# g.default_algorithm = Metropolis()
setdist!(g, (2.0, 2.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=3
# println(@report_opt genAdj!(g, wg) )
genAdj!(g, wg)

interface(g)
# createProcess(g)
# homogeneousself!(g, 0.5f0)

struct Recalc <: Processes.ProcessAlgorithm 
    i::Int
end
function Processes.step!(r::Recalc, context)
    (;hamiltonian) = context
    ii.recalc!(hamiltonian[r.i])
    return (;)
end

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
g.hamiltonian = h = Ising(g) + CoulombHamiltonian(g, 1f0, screening = 4f0);
reprepare(g)
# g.hamiltonian = h = Ising(g) + Quartic(g)
# algo = Processes.CompositeAlgorithm((isingmetro, Recalc(), PolTracker()), (1, 200, 10000),  
#     Share(isingmetro, Recalc()), 
#     Share(isingmetro, PolTracker())
#     )
algo = Processes.CompositeAlgorithm(isingmetro, Recalc(3), PolTracker(), (1, 1000, 1),  
    Share(isingmetro, Recalc(3)),
    Share(isingmetro, PolTracker())
    )

createProcess(g, algo, Input(PolTracker(), isinggraph = g))

using Random
proposer = ii.get_proposer(g)
proposal = rand(Random.MersenneTwister(123), proposer)
hargs = (;s = InteractiveIsing.state(g), w = InteractiveIsing.adj(g), self = InteractiveIsing.self(g), g.hamiltonian...)
ii.ΔH(g.hamiltonian[3], hargs, proposal)