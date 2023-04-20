# Setting elements

"""
Set spins either to a value or clamp them
"""
#Clean this up
setSpins!(sim, g, idxs::Vector{Int32}, brush, clamp = false) = setOrClamp!(sim, g, idxs, brush, clamp)
setSpins!(sim, g, coords::Vector{Tuple{Int16,Int16}}, brush, clamp = false) = setSpins!(sim, g, coordToIdx.(coords, glength(g)), brush, clamp)

function setSpins!(sim, g::AbstractIsingGraph{T}, idx::Int32, brush, clamp = false) where T
    if T == Int8
        clamp = brush == 0 ? true : clamp
    end

    setdefect(g, clamp, idx)

    setSimHType!(sim, "Defects" =>  hasDefects(defects(graph(g))))
    
    @inbounds state(g)[idx] = brush
end

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

# Assuming the two layes are part of the same underlying graph, which is always for now
addWeight!(layer1::IsingLayer, layer2::IsingLayer, idx1::Integer, idx2::Integer, weight) = addWeight!(graph(layer1), idxLToG(layer1, idx1), idxLToG(layer2, idx2), weight)
# Using coordinate indexing
addWeight!(layer1::IsingLayer, layer2::IsingLayer, coords1::Tuple, coords2::Tuple, weight) = addWeight!(layer1, layer2, coordToIdx(coords1, glength(layer1)), coordToIdx(coords2, glength(layer2)), weight)

