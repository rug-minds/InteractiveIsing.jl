using InteractiveIsing, InteractiveIsing.Processes, BenchmarkTools, JET, Tullio, AcceleratedKernels, SparseArrays
import AcceleratedKernels as AK
import InteractiveIsing as II

function ising(dr)
    return dr == 1 ? 1 : 0
end

g = IsingGraph(30,30,30, type = Discrete, periodic = (:z, :y))
# g.hamiltonian = NIsing
wg = @WG ising NN = (1,1,1)

genAdj!(g[1], wg)

# g.hamiltonian = NIsing(g) + DepolField(g)
g.hamiltonian = Ising(g)
# g.hamiltonian = II.deactivateparam(g.hamiltonian, :b)
g.hamiltonian = activateparam(g.hamiltonian, :b)
getparam(g, :b)

function tulliotest(state, adj, hamiltonian, idx)
    (;b) = hamiltonian
    @tullio threads = false sum := b[idx]*state[idx]  + adj[j, idx]*state[j]
    return sum
end

function ak_test(state, adj, hamiltonian, idx)
    (;b) = hamiltonian
    sum = Float32(0)
    AK.foraxes(adj, 2) do j
        sum += b[idx]*state[idx]  + adj[idx, j]*state[j]
    end
    return sum
end

createProcess(g, Metropolis)
context = getcontext(process(g))
pause(g)

_state = II.state(g)
_adj = II.adj(g)
gh = g.hamiltonian
tadj = Tensor(_adj)

tulliotest(_state, _adj, gh, 1, proposal_state = 1)
@benchmark tulliotest($_state, $_adj, $gh, 1)
@benchmark ak_test($_state, $_adj, $gh, 1)
h = II.deltaH(g.hamiltonian)
@benchmark h(context, j = 1)
context.newstate

function finchtest(state, adj, hamiltonian, idx, proposal_state = 1)
    (;b) = hamiltonian
    # sum = 0f0
    @einsum sum[] += b[idx]*(state[idx]-proposal_state)  + adj[idx, j]*(state[j]-proposal_state)
    return sum
end

finchtest(_state, _adj, gh, 1)[]
h(context, j = 1)
@benchmark finchtest($_state, $_adj, $gh, 1)

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
