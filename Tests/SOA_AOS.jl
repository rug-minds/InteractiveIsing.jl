using LoopVectorization
using BenchmarkTools
using SIMD

struct Connections
    idxs::Vector{Int32}
    weights::Vector{Float32}
end

struct NConnections{N}
    idxs::NTuple{N, Int32}
    weights::NTuple{N, Float32}
end


struct Graph
    state::Vector{Float32}
    adj_aos::Vector{Vector{Tuple{Int32, Float32}}}
    adj_soa::Vector{Connections}
    adj_n::Vector{NConnections}
end

# Initialize graph with given size and number of random connections per node
function Graph(size, n_connections)
    # Initialize state
    state = rand(Float32, size)

    # Initialize the array of structures
    adj_aos = Vector{Vector{Tuple{Int32, Float32}}}(undef, size)
    for i in 1:size
        adj_aos[i] = Vector{Tuple{Int32, Float32}}(undef, n_connections)
        for j in 1:n_connections
            adj_aos[i][j] = (rand(UnitRange{Int32}(1:size)), rand(Float32))
        end
        
        sort!(adj_aos[i], by = x -> x[1])
    end

    # Copy the data into a structure of arrays
    adj_soa = Vector{Connections}(undef, size)
    for i in 1:size
        conns = Connections([x[1] for x in adj_aos[i]], [x[2] for x in adj_aos[i]])
        adj_soa[i] = conns
    end

    adj_n = Vector{NConnections}(undef, size)
    for i in 1:size
        conns = NConnections{n_connections}(NTuple{n_connections, Int32}( [x[1] for x in adj_aos[i]] ), NTuple{n_connections, Float32}( [x[2] for x in adj_aos[i]] ))
        adj_n[i] = conns
    end

    return Graph(state, adj_aos, adj_soa, adj_n)
end

# Get the energy for a given state and connections
function getE_aos(state, connections)
    e = 0f0
    @inbounds @simd for conns_idx in eachindex(connections)
        c_idx = connections[conns_idx][1]
        weight = connections[conns_idx][2]
        e += -state[c_idx]*weight
    end
    return e
end

function getE_aos_simd(state, connections, ::Type{Vec{N, Float32}})
    e = 0f0
    @inbounds @simd for conns_idx in eachindex(connections)
        c_idx = connections[conns_idx][1]
        weight = connections[conns_idx][2]
        e += -state[c_idx]*weight
    end
    return e
end

# Get the energy for a given state and connections
function getE_soa(state, connections)
    e = 0f0
    idxs = connections.idxs
    weights = connections.weights
    @turbo for conns_idx in eachindex(idxs)
        idx = idxs[conns_idx]
        weight = weights[conns_idx]
        e += -state[idx]*weight
    end
    return e
end

# Get the energy for a given state and connections
function getE_n(state, connections)
    e = 0f0
    idxs = connections.idxs
    weights = connections.weights
    @turbo for conns_idx in eachindex(idxs)
        idx = idxs[conns_idx]
        weight = weights[conns_idx]
        e += -state[idx]*weight
    end
    return e
end

# Main loop AOS
function mainloop_aos(graph, iterations)
    etot = 0.
    @inbounds for _ in 1:iterations
        idx = rand(UnitRange{Int32}(1:length(graph.state)))
        etot += graph.state[idx]*getE_aos(graph.state, graph.adj_aos[idx])
    end

    return etot
end


function mainloop_soa(graph, iterations)
    etot = 0.
    @inbounds for _ in 1:iterations
        idx = rand(UnitRange{Int32}(1:length(graph.state)))
        etot += graph.state[idx]*getE_soa(graph.state, graph.adj_soa[idx])
    end

    return etot
end

const g8 = Graph(500^2, 8)
const g10 = Graph(500^2, 10)
const g100 = Graph(500^2, 100)
const g1000 = Graph(500^2, 1000)


@benchmark getE_aos($g1000.state, $g1000.adj_aos[1])
@benchmark getE_soa($g1000.state, $g1000.adj_soa[1])
@benchmark getE_soa($g1000.state, $g1000.adj_n[1])

# @benchmark getE_aos($g8.state, $g8.adj_aos[1])
# @benchmark getE_soa($g8.state, $g8.adj_soa[1])

# @benchmark getE_aos($g10.state, $g10.adj_aos[1])
# @benchmark getE_soa($g10.state, $g10.adj_soa[1])

# @benchmark getE_aos($g100.state, $g100.adj_aos[1])
# @benchmark getE_soa($g100.state, $g100.adj_soa[1])

# @benchmark getE_aos($g1000.state, $g1000.adj_aos[1])
# @benchmark getE_soa($g1000.state, $g1000.adj_soa[1])

# @benchmark mainloop_aos($g10, 1e5)
# @benchmark mainloop_soa($g10, 1e5)

# @benchmark mainloop_aos($g100, 1e5)
# @benchmark mainloop_soa($g100, 1e5)

# @benchmark mainloop_aos($g1000, 1e5)
# @benchmark mainloop_soa($g1000, 1e5)

# g1000.adj_aos .= deepcopy(g1000.adj_aos)
# g1000.adj_soa .= deepcopy(g1000.adj_soa)