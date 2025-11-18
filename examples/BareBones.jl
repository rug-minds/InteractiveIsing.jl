using InteractiveIsing, SparseArrays

# isingfunc(dr) = dr == 1 ?  1.0 : 0.
isingfunc(dr) = 1/dr^2

g = IsingGraph(500,500, type = Continuous, periodic = true)

wg = @WG isingfunc NN=3
adj = sparse(genAdj(g[1], wg)...)

bg = BareGraph(Float32, [(500,500)] , adj, zeros(Float32, 500*500))
simulateBare(bg, 1)
