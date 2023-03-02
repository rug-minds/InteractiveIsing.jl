# Setting elements

"""
Set spins either to a value or clamp them
"""

setSpins!(sim, g, idxs, brush, clamp = false) = setGraphSpins!(sim, g, idxs, brush, clamp)
setSpins!(sim, g, tupls, brush, clamp = false) = setGraphSpins!(sim, g, coordToIdx.(tupls, glength(g)), brush, clamp)

"""
Adds weight to adjacency matrix
"""
function addWeight!(adj::Vector, idx, conn_idx, weight)
    insert_and_dedup!(adj[idx], (conn_idx, weight))
    insert_and_dedup!(adj[conn_idx], (idx, weight))
end

addWeight!(g::IsingGraph, idx, conn_idx, weight) = addWeight!(adj(g), idx, conn_idx, weight)

addWeight!(g::IsingLayer, idx, conn_idx, weight) = addWeight!(graph(g), idxLToG(g, idx), idxLToG(g, conn_idx), weight)

export addWeight!

