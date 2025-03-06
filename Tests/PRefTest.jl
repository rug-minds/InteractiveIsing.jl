using InteractiveIsing, Processes, LoopVectorization, BenchmarkTools, SparseArrays
import InteractiveIsing as II
g = IsingGraph(30,30)
createProcess(g)
w = @WG "dr -> 1/dr" NN=4
genAdj!(g[1], w)

resize!(getparams(g, :self).val, 900)
getparams(g,:self).val .= rand(Float32, 900)
getparams(g, :b ).val .= rand(Float32, 900)

as = (getargs(process(g))..., j = 1, params = getparams(g))

rc = @ParameterRefs (s_i+b_i)*w_ij 
si = @ParameterRefs s_i
wij = @ParameterRefs w_ij

red = @ParameterRefs s_i+b_i-self_i
red(as; i = 1) 

@benchmark rc(as; i = 1)
rc(as; i = 1)


# II.reduce_contraction(rc, 1, as, II.VecRef(), II.SparseMatrixRef())


# @benchmark II.reduce_contraction(rc, 1, as, II.VecRef(), II.SparseMatrixRef())

# sum_p_coll(pc)