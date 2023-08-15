# struct LayerConnFunc{T}
#     func::Function{T}
#     alignment::Symbol
#     NN::Integer
# end

# Alignment modes
# 1: :none - no alignment (default)
# 2: :center - center layer 2 in layer 1

# Gives a vector of the relative lattice positions
function dlayer(layer1, layer2)
    if coords(layer1) != nothing && coords(layer2) != nothing
        rel_coords = coords(layer2) .- coords(layer1)
    else
        error("Can only connect adjacent layers")
    end
end

# Gives the relative position of layer 2 in layer 1's coordinates
function dcoords(layer1, layer2, alignment = :center)
    dly, dlx, dlz = dlayer(layer1, layer2)

    xsize1 = gwidth(layer1)
    ysize1 = glength(layer1)
    xsize2 = gwidth(layer2)
    ysize2 = glength(layer2)

    center_mask = alignment == :center ? 1 : 0
    # First factor is for offset based on relative positions, second is for centering
    return (dly*ysize1+ (abs(dlx) + abs(dlz))*center_mask*(ysize1-ysize2)/2, dlx*xsize1 + (abs(dly)+ abs(dlz))*center_mask*(xsize1-xsize2)/2, dlz)
    
end

# Interprets a coordinate of layer 1 as on of layer 2 given the relative position
coordsl1tol2(i1, j1, layer1, layer2, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); (i1, j1) .- dyxz[1:2] end
coordsl1tol2((i1, j1)::Tuple, layer1, layer2, alignment = :center) = coordsl1tol2(i1, j1, layer1, layer2, alignment)
# Interprets a coordinate of layer 2 as on of layer 1 given the relative position
coordsl2tol1(i2, j2, layer1, layer2, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); (i2, j2) .+ dyxz[1:2] end
coordsl2tol1((i2, j2)::Tuple, layer1, layer2, alignment = :center) = coordsl2tol1(i2, j2, layer1, layer2, alignment)
export coordsl1tol2, coordsl2tol1

# put x in between min and max
minmax(lower, x, upper) = max(min(x, upper), lower)

# Snaps a coordinate to the nearest edge of a layer
snaptolayer(x, layer) = minmax(1, x, glength(layer))

# Returns the nearest neighbor of (i1, j1) in layer2's coordinates
# Works by translating the coordinate of layer one to one of layer2, and then snapping
# it to the nearest edge of the layer
function nearest(i1, j1, layer1, layer2, alignment = :center)
    i1to2, j1to2 = coordsl1tol2(i1, j1, layer1, layer2, alignment)

    nny = minmax(1, i1to2, glength(layer2))
    nnx = minmax(1, j1to2, gwidth(layer2))

    return nny, nnx
end
export nearest

function isin(i, j, layer)
    if (1 <= i <= glength(layer)) && (1 <= j <= gwidth(layer))
        return true
    end

    return false
end

isin((i, j), layer) = isin(i, j, layer)

function dist2(i1, j1, i2, j2, layer1, layer2, alignment::Symbol = :center)
    _, _, dlz = dlayer(layer1, layer2)
    i21, j21 = coordsl2tol1(i2, j2, layer1, layer2, alignment)
    return (i1 - i21)^2 + (j1 - j21)^2 + dlz^2
end

function dist2(idx1, idx2, layer1::IsingLayer, layer2::IsingLayer, alignment::Symbol = :center) 
    i1, j1 = idxToCoord(idx1, glength(layer1))
    i2 ,j2 = idxToCoord(idx2, glength(layer2))

    return dist2(i1, j1, i2, j2, layer1, layer2, alignment)
end

function lattice2_iterator(i1, j1, layer1, layer2, NN, alignment = :center)
    _, _, dlz = abs.(dlayer(layer1, layer2))
    coordinates2 = coordsl1tol2.([(i1+i,j1+j) for i in (-(NN-dlz)):(NN-dlz), j in (-(NN-dlz)):(NN-dlz)], Ref(layer1), Ref(layer2), Ref(alignment))
    filter = isin.(coordinates2, Ref(layer2))
    leftovers = coordinates2[filter]
    return coordToIdx.(leftovers, glength(layer2))
end

export lattice2_iterator
# Func in the form of (;dr, i1, j1, i2, j2) -> body
function connectLayers!(g, layeridx1, layeridx2, func, NN = 1, alignment = :center)
    layer1 = layers(g)[layeridx1]
    layer2 = layers(g)[layeridx2]

    connectLayersLoop!(layer1, layer2, func, NN, alignment)

end

function connectLayersLoop!(layer1, layer2, func, NN = 1, alignment = :center)
    for idx1 in 1:nStates(layer1)
        i1, j1 = idxToCoord(idx1, glength(layer1))
        for idx2 in lattice2_iterator(i1, j1 , layer1, layer2, NN, alignment)
            i2 ,j2 = idxToCoord(idx2, glength(layer2))
            dist = sqrt(dist2(i1, j1, i2, j2, layer1, layer2, alignment))
            weight = func(dr = dist; i1, i2, j1, j2)
            if weight != 0
                addWeight!(layer1, layer2, idx1, idx2, weight)
            end
        end
    end
end
export connectLayers!

function disconnectLayers!(g, layeridx1, layeridx2)
    
    layer1 = layers(g)[layeridx1]
    layer2 = layers(g)[layeridx2]
    idxrange1 = start(layer1):endidx(layer1)
    idxrange2 = start(layer2):endidx(layer2)

    idxs = []
    for idx1 in idxrange1
        for (adj_idx,conn) in enumerate(adj(g)[idx1])
            if connIdx(conn) ∈ idxrange2
                push!(idxs, adj_idx)
            end
        end
        # println(idxs)
        deleteat!(adj(g)[idx1], idxs)
        idxs = []
    end
    for idx2 in idxrange2
        for (adj_idx,conn) in enumerate(adj(g)[idx2])
            if connIdx(conn) ∈ idxrange1
                push!(idxs, adj_idx)
            end
        end
        deleteat!(adj(g)[idx2], idxs)
        idxs = []
    end
end
export disconnectLayers!