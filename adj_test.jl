using BenchmarkTools

const state = rand(1000)
const n_connections = 4

# (connections, weights)
const adj = (sort!(rand(1:1000, n_connections)), rand(n_connections))

function main_algo_loop(state, adj)
    e = 0
    for (loop_idx, connection_idx) in enumerate(adj[1])
        e += state[connection_idx]*adj[2][loop_idx]
    end
    return e
end

function main_algo_broadcast(state, adj)
    return sum((@view state[adj[1]]) .* adj[2])
end

function main_algo_broadcast_inbounds(state, adj)
    return @inbounds sum((@view state[adj[1]]) .* adj[2])
end

@btime main_algo_loop($state, $adj)
@btime main_algo_broadcast($state, $adj)
@btime main_algo_broadcast_inbounds($state, $adj)

main_algo_sum(state, adj) = 
  sum(i -> @inbounds( state[adj[1][i]]*adj[2][i] ), 1:length(adj[1]))
  
main_algo_sum3(stateview, adj) = 
    sum(i -> @inbounds( stateview[i]*adj[2][i] ), 1:length(adj[1]))

@btime main_algo_sum(state, adj)


const stateview = @view state[adj[1]]
@btime main_algo_sum3(stateview, adj)
@btime main_algo_sum3((@view state[adj[1]]), adj)