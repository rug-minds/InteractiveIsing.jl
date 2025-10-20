using InteractiveIsing, InteractiveIsing.Processes, BenchmarkTools, JET, Tullio
import InteractiveIsing as II

g = IsingGraph(30,30,30, type = Discrete, periodic = (:z, :y))
# g.hamiltonian = NIsing
wg = @WG (dr) -> dr == 1 ? 1 : 0 NN = (1,1,1)
genAdj!(g[1], wg)

# g.hamiltonian = NIsing(g) + DepolField(g)
g.hamiltonian = Ising(g)
# g.hamiltonian = II.deactivateparam(g.hamiltonian, :b)

getparam(g.hamiltonian, :b)

function tulliotest(state, adj, hamiltonian, idx)
    (;b) = hamiltonian
    @tullio threads = false avx = true sum := b[j]*state[j]  + adj[idx, j]*state[j]
    return sum
end

createProcess(g, Metropolis)
args = getargs(process(g))
pause(g)

_state = II.state(g)
_adj = II.adj(g)
gh = g.hamiltonian


@benchmark tulliotest($_state, $_adj, $gh, 1)
h = II.deltaH(g.hamiltonian)
@benchmark h(args, j = 1)


interface(g)
createProcess(g, Metropolis)


close(process(g))
as = (;fetch(g)..., newstate =II.NewState(0f0))
quit(g)

# paramH = II.deltaH(NIsing)


# createProcess(g, II.MetropolisNew())

# createProcess(g, MetropolisNew, overrides = (;hamiltonian = II.deltaH(NIsing())))

# contract = @ParameterRefs (s_i*w_ij)
# reduce = @ParameterRefs (sn_j-s_j)'
# together = @ParameterRefs (s_i*w_ij)*(sn_j-s_j)

# as = (;prepare(MetropolisNew(), (;g))..., newstate = II.NewState(-state(g)[2]))
# contract(as; j = Int32(2))
# reduce(as; j = Int32(2))

# together(as; j = Int32(2))

