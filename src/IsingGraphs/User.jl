# Setting elements

"""
Set spins either to a value or clamp them
"""
#Clean this up
function setSpins!(g::AbstractIsingGraph{T}, idxs::AbstractArray, brush, clamp = false, refresh = true) where T
    if T == Int8
        clamp = brush == 0 ? true : clamp
    end

    # Set the defects
    setrange!(defects(g), clamp, idxs)
    
    # Set the stype to wether it has defects or not
    setSType!(graph(g), :Defects => hasDefects(graph(g)), force_refresh = refresh)

    # Set the spins
    @inbounds state(g)[idxs] .= brush
end

setSpins!(g::AbstractIsingGraph, coords::Vector{Tuple{Int16,Int16}}, brush, clamp = false) = setSpins!(g, coordToIdx.(coords, glength(g)), brush, clamp)

function setSpins!(g::AbstractIsingGraph{T}, idx::Int32, brush, clamp = false, refresh = false) where T
    if T == Int8
        clamp = brush == 0 ? true : clamp
    end

    setdefect(g, clamp, idx)

    setSType!(graph(g), :Defects => hasDefects(graph(g)), force_refresh = refresh)

    @inbounds state(g)[idx] = brush
end


export addWeight!

"""
Adding weights
"""

addWeight!(g::IsingGraph, idx, conn_idx, weight; sidx1 = 1, sidx2 = 1, eidx1 = nothing, eidx2 = nothing) = addWeight!(adj(g), idx, conn_idx, weight; sidx1, sidx2, eidx1, eidx2)

addWeight!(g::IsingLayer, idx, conn_idx, weight; sidx1 = 1, sidx2 = 1, eidx1 = nothing, eidx2 = nothing) = addWeight!(adj(graph(g)), idxLToG(g, idx), idxLToG(g, conn_idx), weight; sidx1, sidx2, eidx1, eidx2)


addWeight!(layer1::IsingLayer, layer2::IsingLayer, idx1::Integer, idx2::Integer, weight; sidx1 = 1) = 
    addWeight!(adj(graph(layer1)), idxLToG(layer1, idx1), idxLToG(layer2, idx2), weight; sidx1)

# Using coordinate indexing
addWeight!(layer1::IsingLayer, layer2::IsingLayer, coords1::Tuple, coords2::Tuple, weight) = addWeight!(layer1, layer2, coordToIdx(coords1, glength(layer1)), coordToIdx(coords2, glength(layer2)), weight)


# For performance, but should generally not be used
# Sidx is the first index of the adjacency matrix to start looking for connections
# This is useful if you know that the connections are in a certain range for performance
addWeightDirected!(layer1::IsingLayer, layer2::IsingLayer, idx1::Integer, idx2::Integer, weight; sidx = 1) = 
    addWeightDirected!(adj(graph(layer1)), idxLToG(layer1, idx1), idxLToG(layer2, idx2), weight; sidx)

addWeightDirected!(layer1::IsingLayer, layer2::IsingLayer, coords1::Tuple, coords2::Tuple, weight; sidx = 1) = 
    addWeightDirected!(layer1, layer2, coordToIdx(coords1, glength(layer1)), coordToIdx(coords2, glength(layer2)), weight; sidx)

addWeightDirected!(layer::IsingLayer, idx, conn_idx, weight; sidx = 1) = addWeightDirected!(adj(graph(layer)), idxLToG(layer, idx), idxLToG(layer, conn_idx), weight; sidx)
export addWeightDirected!
# Clamp an image to a layer

function clampImg!(layer::IsingLayer, imgfile)
    # Load the image
    img = load(imgfile)

    # Resize the image
    img = imresize(img, (Int64(glength(layer)), Int64(gwidth(layer))))

    # # Convert to black and white image
    img = Gray.(img)
    img = img .> 0.5
    img = img .*2 .- 1

    setSpins!(layer, [1:length(img);], img[:] , true)

end

clampImg!(g, layeridx::Integer, imgfile) = clampImg!(layers(g)[layeridx], imgfile)
export clampImg!

function copyState!(layer1, layer2, clamp = false)
    imresize(state(layer1), (Int64(glength(layer2)), Int64(gwidth(layer2))))
    state(layer2) .= state(layer1)
end
copyState!(g, layeridx1::Integer, layeridx2::Integer, clamp = false) = copyState!(layer(g, layeridx1), layer(g, layeridx2), clamp)
export copyState!

function overlayNoise!(layer::IsingLayer, p; noise_values = [-1, 1])
    maskVec = rand(length(state(layer))) .< (p/100)
    idxs = [i for (i, x) in enumerate(maskVec) if x]
    states = rand(noise_values, length(idxs))
    state(layer)[idxs] .= states
end

overlayNoise!(g, layeridx::Integer, p; noise_values = [-1, 1]) = overlayNoise!(layers(g)[layeridx], p; noise_values)
export overlayNoise!