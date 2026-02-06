using BenchmarkTools

sizestate = 1000

const adj1 = ([1:sizestate;],[rand(sizestate);])
const adj2 = [(i,adj1[2][i]) for i in 1:sizestate]
const state = rand(sizestate)

# main_algo_sum_tup(state, adj) = 
#   sum(i -> @inbounds( state[adj[1][i]]*adj[2][i] ), 1:length(@inbounds adj[1]))

main_algo_sum_tup(state, adj) = 
  sum(i -> ( state[adj[1][i]]*adj[2][i] ), 1:length(adj[1]))

# main_algo_sum_vec(state, adj) =
#     sum(i -> @inbounds( state[adj[i][1]]*adj[i][2] ), 1:length(adj))

main_algo_sum_vec(state, adj) =
    sum(i -> ( state[adj[i][1]]*adj[i][2] ), 1:length(adj))

function main_algo_loop(state, adj)
    e = 0
    for (idx, conn) in enumerate(adj[1])
        e += adj[2][idx]*state[conn]
    end
    return e
end

sizeadj = 100

const adj3 = (sort!(rand(1:sizestate, sizeadj)), [rand(sizeadj);])
const adj4 = [(adj3[1][i], adj3[2][i]) for i in 1:sizeadj]

function loopalgo_sum_tup(state, adj)
    e = zero(eltype(state))
    for _ in 1:100
        e += main_algo_sum_tup(state,adj)
    end
    return e
end

function loopalgo_sum_vec(state, adj)
    e = zero(eltype(state))
    for _ in 1:100
        e += main_algo_sum_vec(state,adj)
    end
    return e
end

function testalgo_loop(state, adj)
    e = zero(eltype(state))
    for _ in 1:100
        e += main_algo_loop(state,adj)
    end
    return e
end

@btime main_algo_sum_tup($state, $adj3)
@btime main_algo_sum_vec($state, $adj4)

@btime main_algo_loop(state,adj3)

@btime loopalgo_sum_tup(state, adj3)
@btime loopalgo_sum_vec(state, adj4)
@btime testalgo_loop(state, adj3)


function main_algo_inbounds_simd(state, adj)
    e = zero(eltype(state))
    @inbounds @simd for loop_idx in eachindex(adj[1])
        connection_idx = adj[1][loop_idx]
        e += state[connection_idx]*adj[2][loop_idx]
    end
    return e
end

function main_algo_inbounds_simd_vec(state, adj_vec)
    e = zero(eltype(state))
    @inbounds @simd for loop_idx in eachindex(adj_vec)
        connection_idx = adj_vec[loop_idx][1]
        e += state[connection_idx]*adj_vec[loop_idx][2]
    end
    return e
end

@btime main_algo_inbounds_simd($state, adj3)

function testalgo_loop_simd(state, adj)
    e = zero(eltype(state))
    for _ in 1:100
        e += main_algo_inbounds_simd(state, adj)
    end
    return e
end

function testalgo_loop_simd_vec(state, adj)
    e = zero(eltype(state))
    for _ in 1:100
        e += main_algo_inbounds_simd_vec(state, adj)
    end
    return e
end

@btime testalgo_loop_simd($state,$adj3)
@btime testalgo_loop_simd_vec($state,$adj4)