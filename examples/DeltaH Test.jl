using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes, Random
import InteractiveIsing as ii

rng = Random.MersenneTwister()

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g = ii.IsingGraph(30,30,10, type = Continuous, periodic = (:x,:y))
# g.default_algorithm = Metropolis()
setdist!(g, (2.0, 1.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=2
# println(@report_opt genAdj!(g, wg) )
genAdj!(g, wg)


proposer = ii.get_proposer(g)
proposal = ii.rand(rng, proposer)
hargs = (;s = InteractiveIsing.state(g), w = InteractiveIsing.adj(g), self = InteractiveIsing.self(g), g.hamiltonian...)
ii.ΔH_exp(g.hamiltonian, (;self = InteractiveIsing.self(g), s = InteractiveIsing.state(g), w = InteractiveIsing.adj(g), g.hamiltonian...), proposal)
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

