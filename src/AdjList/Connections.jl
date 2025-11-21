"""
Get a list of the coordinates of the outgoing connections of a node in a graph.
"""
function conn_coords(g,i)
    idxToCoord.(graph(g).adj.rowval[nzrange(graph(g).adj,i)],Ref(size(g)))
end

function conn_idxs(g::AbstractIsingGraph,i)
    conn_idxs(graph(g).adj,i)
end

function conn_weights(g::AbstractIsingGraph,i)
    conn_weights(graph(g).adj,i)
end

function conn_weights(adj::SparseMatrixCSC, i)
    adj.nzval[nzrange(adj,i)]
end

function conn_idxs(adj::SparseMatrixCSC, i)
    adj.rowval[nzrange(adj,i)]
end

function show_connections(g,i)
    coords = conn_coords(g,i)
    weights = conn_weights(g,i)
    for (coord, weight) in zip(coords, weights)
        println("$coord => $weight")
    end
end
show_connections(g, i, j) = show_connections(g, coordToIdx(Int32.((i, j)), size(g)))
show_connections(g, i, j, k) = show_connections(g, coordToIdx(Int32.((i, j, k)), size(g)))


function show_relative_connections(g, i)
    coords = conn_coords(g,i)
    weights = conn_weights(g,i)
    i_coords = idxToCoord(i, size(g))
    for (coord, weight) in zip(coords, weights)
        println("$(coord .- i_coords) => $weight")
    end
end

show_relative_connections(g, i, j) = show_relative_connections(g, coordToIdx(Int32.((i, j)), size(g)))
show_relative_connections(g, i, j, k) = show_relative_connections(g, coordToIdx(Int32.((i, j, k)), size(g)))


export show_connections, show_relative_connections