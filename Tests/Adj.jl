using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    # @show c1
    # @show c2
    # @show dr
    return 2/dr^2
end


g = ii.IsingGraph(2,2,2, type = Continuous, periodic = (:x,:y))
setdist!(g, (2.0, 1.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=1
genAdj!(g, wg)

adj1 = deepcopy(g.adj)

setdist!(g, (1.0, 10., 20.))
genAdj!(g, wg)

adj2 = deepcopy(g.adj)

@show adj1 != adj2