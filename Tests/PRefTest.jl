using InteractiveIsing, Processes, LoopVectorization, BenchmarkTools, SparseArrays
import InteractiveIsing as II
g = IsingGraph(30,30,30, type = Discrete, periodic = (:z, :y))
createProcess(g)
w = @WG "dr -> 1/dr" NN=4
genAdj!(g[1], w)

resize!(getparams(g, :self).val, 900)
getparams(g,:self).val .= rand(Float32, 900)
getparams(g, :b ).val .= rand(Float32, 900)
setparam!(g, :self, rand(Float32, 900), true)
setparam!(g, :b, rand(Float32, 900), true)

function testrc(as; j = 1)
    adj = as.gadj
    cumsum = zero(eltype(as.gstate))
    @turbo for ptr in nzrange(adj, j)
        i = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        cumsum += wij * as.gstate[i] 
    end
    return cumsum
end

struct NewState
    Val
end
Base.getindex(n::NewState, i) = n.Val
as = (getargs(process(g))..., j = 1, params = getparams(g), newstate = NewState(1))



rc = @ParameterRefs (s_i)*w_ij 
si = @ParameterRefs s_i
wij = @ParameterRefs w_ij

red = @ParameterRefs s_i+b_i-self_i
red(as; i = 1) 

@benchmark $rc($as; j = 1)



rc(as; j = 1)


ms = @ParameterRefs (s_j^2-sn_j^2)*self_j+(s_j-sn_j)*(b_j)
ms1 = @ParameterRefs (s_j-sn_j)*(b_j)
