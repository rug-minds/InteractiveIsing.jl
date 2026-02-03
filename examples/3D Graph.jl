using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g = ii.IsingGraph(80,80,10, type = Continuous, periodic = (:x,:y))
# g.default_algorithm = Metropolis()
setdist!(g, (2.0, 1.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
# println(@report_opt genAdj!(g, wg) )
genAdj!(g, wg)

# homogeneousself!(g, 0.5f0)

struct Recalc <: Processes.ProcessAlgorithm end
function Processes.step!(::Recalc, context)
    (;hamiltonian) = context
    @show typeof(hamiltonian)
    recalc!(hamiltonian[3])
end

g.hamiltonian = h = Ising(g) + CoulombHamiltonian2(g, 1f0)
interface(g)
algo = Processes.CompositeAlgorithm((Metropolis(), Recalc()), (1,200),  Processes.DestructureInput(), Share(DestructureInput(), Metropolis()), Share(Metropolis(), Recalc()))

createProcess(g, algo)


# h = g.hamiltonian = Ising(g) + Quartic(g) + DepolField(g, top_layers = 2, zfunc = z -> 3/z, NN = 2) + Sextic(g)
# # refresh(g)
# # h[4].c[] = 1/(2*2500)

# w = interface(g)
# createProcess(g)
# interface(g)


# # pause(g)
# # args = fetch(g)
# args = ii.prepare(Metropolis(), (;g = g))
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

