using LoopVectorization, BenchmarkTools
function ttest_separate(g, adj, idx)
    sum = zero(eltype(g))
    sum2 = zero(eltype(g))
    gstate = state(g)
    @turbo for ptr in nzrange(adj, idx)
        j = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        sum += wij*gstate[j]
    end
    @turbo for ptr in nzrange(adj, idx)
        j = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        sum2 += wij*(gstate[j])^2
    end
    return sum + sum2
end

function ttest_together(g, adj, idx)
    sum = zero(eltype(g))
    sum2 = zero(eltype(g))
    gstate = state(g)
    @turbo for ptr in nzrange(adj, idx)
        j = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        sum += wij*gstate[j]
        sum2 += wij*(gstate[j])^2
    end
    return sum + sum2
end

