using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1
end

function normalize_adj_by_average_col(adj::A, scaling = one(eltype(adj))) where A
    cols = eltype(adj)[]
    for j in axes(adj, 2)
        s = sum(abs, @view adj[:, j])
        push!(cols, s)
    end
    avg_col_sum = mean(cols)
    return (scaling/avg_col_sum) .* adj
end


g = ii.IsingGraph(100,100,10, type = Continuous, periodic = (:x,:y))
setdist!(g, (2.0, 2.0, 1.0))

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=3
genAdj!(g, wg)

adj = g.adj
newadj = normalize_adj_by_average_col(adj, 1f0)

@show adj[:,1]
@show newadj[:,1]