using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g = ii.IsingGraph(100,100,10, type = Continuous, periodic = (:x,:y))
# g.default_algorithm = Metropolis()
setdist!(g, (2.0, 2.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
# println(@report_opt genAdj!(g, wg) )
genAdj!(g, wg)

interface(g)
# createProcess(g)
# homogeneousself!(g, 0.5f0)

struct Recalc <: Processes.ProcessAlgorithm 
    i::Int
end
# function Processes.step!(r::Recalc, context)
#     (;hamiltonian) = context
#     ii.recalc_tridiag!(hamiltonian[r.i])
#     return (;)
# end

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
# g.hamiltonian = h = Ising(g) + CoulombHamiltonian(g, 1f0, screening = 0.5f0);
g.hamiltonian = h = Ising(g) + Quartic(g)
# algo = Processes.CompositeAlgorithm((isingmetro, Recalc(), PolTracker()), (1, 200, 10000),  
#     Share(isingmetro, Recalc()), 
#     Share(isingmetro, PolTracker())
#     )
algo = Processes.CompositeAlgorithm(isingmetro, Recalc(3), PolTracker(), (1, 1000, 1),  
    Share(isingmetro, Recalc(3)),
    Share(isingmetro, PolTracker())
    )

# createProcess(g, algo, Input(PolTracker(), isinggraph = g))
# # interface(g)


# h = g.hamiltonian = Ising(g) + Quartic(g) + DepolField(g, top_layers = 2, zfunc = z -> 3/z, NN = 2) + Sextic(g)
# # reprepare(g)
# # h[4].c[] = 1/(2*2500)

# w = interface(g)
# createProcess(g)
# interface(g)


# # pause(g)
# # args = fetch(g)
# args = ii.init(Metropolis(), (;g = g))
# dr = FlipProposal(:s, j = 1 => -1f0)


# pref = ii.@ParameterRefs (s[j]-delta_1[j])*b[j]
# dref = pref[1][2]

# ii.get_assignments(pref, hargs, (;j = 1))

# dr = FlipProposal(:s, j = 1 => -1f0)

# h = g.hamiltonian

# hargs = (;args..., s = args.gstate, w = args.gadj, self = args.self, delta_1 = dr, h...)

# ii.ΔH(ising, hargs, dr)
# ii.ΔH(g.hamiltonian, hargs, dr)

# pr = Base.eval(ii.generated_func_calls.args[1])
# ii.generate_block(pr, hargs, (;j = 1))

# pr(hargs, j = 1)

# pr_e = ii.get_ΔH_expr.(ii.H_types(ising))
# prefs = ii.to_ParameterRefs.(pr_e)
# pref = reduce(+, Base.eval.(prefs))

# @code_warntype pref(hargs, (;j = 1))

