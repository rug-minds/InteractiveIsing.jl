using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g = ii.IsingGraph(30,30,10, type = Continuous, periodic = (:x,:y))
setdist!(g, (2.0, 1.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
genAdj!(g, wg)


struct Recalc <: Processes.ProcessAlgorithm 
    i::Int
end
function Processes.step!(r::Recalc, context)
    (;hamiltonian) = context
    recalc!(hamiltonian[r.i])
    return (;)
end

struct PolTracker{T} <: ProcessAlgorithm end
PolTracker() = PolTracker{Float32}()
function Processes.prepare(::PolTracker, context)
    (;isinggraph) = context
    initial = sum(state(isinggraph))

    (;pols = Float32[initial])
end

function Processes.step!(::PolTracker, context)
    (;proposal, pols) = context
    push!(pols, delta(proposal))
    return (;)
end

isingmetro = g.default_algorithm
g.hamiltonian = h = Ising(g) + CoulombHamiltonian2(g, 1f0)

algo = Processes.CompositeAlgorithm((isingmetro, Recalc(3), PolTracker()), (1, 200, 1),  
    Share(isingmetro, Recalc(3)),
    Share(isingmetro, PolTracker())
    )


axis_window()
# createProcess(g, algo, Input(PolTracker, isinggraph = g))
# interface(g)