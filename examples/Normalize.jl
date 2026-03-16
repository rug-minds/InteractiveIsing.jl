using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1
end

function normalize_adj_by_average_col!(adj::A, scaling = one(eltype(adj))) where A
    adj = adj.sp
    cols = eltype(adj)[]
    for j in axes(adj, 2)
        s = sum(abs, @view adj[:, j])
        push!(cols, s)
    end
    avg_col_sum = mean(cols)
    return adj .*= (scaling/avg_col_sum) 
end

function isingfunc(dr, c1, c2)
    return 1/dr
end

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=(3,3,2)

g = ii.IsingGraph(100,100,10, 
        Continuous(), 
        wg, 
        LatticeConstants(1.0, 1.0, 4.),
        StateSet(-1.5f0, 1.5f0),
        Ising(c = ConstVal(1)) + 
            Clamping(1f0)+ Quartic(c = ConstVal(1.0), ) + 
            Sextic(c = ConstVal(1.0), localpotential = StateLike(OffsetArray, 0)),
        periodic = (:x,:y))

normalize_adj_by_average_col!(g.adj, 10f0)

@show adj[:,1]
@show newadj[:,1]