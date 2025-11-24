using InteractiveIsing, JET
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 2/dr^2
end


g3d = ii.IsingGraph(80,80,10, type = Continuous, periodic = (:x,:y))
setdist!(g3d, (2.0, 1.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=3
genAdj!(g3d, wg) 
println(@report_opt genAdj!(g3d[1], wg) )

# homogeneousself!(g3d, 0.5f0)



h = g3d.hamiltonian = Ising(g3d) + Quartic(g3d) + DepolField(g3d, top_layers = 2, zfunc = z -> 3/z, NN = 5) + Sextic(g3d)
# refresh(g3d)
# h[4].c[] = 1/(2*2500)
createProcess(g3d)

w = interface(g3d)


# pause(g3d)
# # args = fetch(g3d)
# args = ii.prepare(Metropolis(), (;g = g3d))
# dr = DeltaRule(:s, j = 1 => -1f0)


# pref = ii.@ParameterRefs (s[j]-delta_1[j])*b[j]
# dref = pref[1][2]

# ii.get_assignments(pref, hargs, (;j = 1))

# dr = DeltaRule(:s, j = 1 => -1f0)

# h = g3d.hamiltonian

# hargs = (;args..., s = args.gstate, w = args.gadj, self = args.self, delta_1 = dr, h...)

# ii.ΔH(ising, hargs, dr)
# ii.ΔH(g3d.hamiltonian, hargs, dr)

# pr = Base.eval(ii.generated_func_calls.args[1])
# ii.generate_block(pr, hargs, (;j = 1))

# pr(hargs, j = 1)

# pr_e = ii.get_ΔH_expr.(ii.H_types(ising))
# prefs = ii.to_ParameterRefs.(pr_e)
# pref = reduce(+, Base.eval.(prefs))

# @code_warntype pref(hargs, (;j = 1))

