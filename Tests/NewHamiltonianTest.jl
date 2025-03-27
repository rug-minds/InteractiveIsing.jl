using InteractiveIsing, Processes, BenchmarkTools, JET
import InteractiveIsing as II

g = IsingGraph(30,30,30, type = Discrete, periodic = (:z, :y))
# g.hamiltonian = NIsing
wg = @WG "(dr) -> dr == 1 ? 1 : 0" NN = (1,1,1)
genAdj!(g[1], wg)

createProcess(g, MetropolisNew, overrides = (;hamiltonian = II.deltaH(II.NIsing(g))))

close(process(g))
as = (;fetch(g)..., newstate =II.NewState(0f0))
quit(g)

paramH = II.deltaH(NIsing)

interface(g)

# createProcess(g, II.MetropolisNew())

createProcess(g, MetropolisNew, overrides = (;hamiltonian = II.deltaH(NIsing())))

# contract = @ParameterRefs (s_i*w_ij)
# reduce = @ParameterRefs (sn_j-s_j)
# together = @ParameterRefs (s_i*w_ij)*(sn_j-s_j)

# as = (;prepare(MetropolisNew(), (;g))..., newstate = II.NewState(-state(g)[2]))
# contract(as; j = Int32(2))
# reduce(as; j = Int32(2))

# together(as; j = Int32(2))