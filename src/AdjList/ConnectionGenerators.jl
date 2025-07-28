
# Symbol mappings for the coordinates
const coord_symbs = (:i, :j, :k, :l, :m, :n)

# Gives a vector of the relative lattice positions
function dlayer(layer1::IsingLayer, layer2)::NTuple{3,Int32}
    if coords(layer1) != nothing && coords(layer2) != nothing
        return rel_coords = coords(layer2) .- coords(layer1)
    else
        error("Can only connect adjacent layers")
        return Int32.((0,0,0))
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
export dcoords

# Interprets a coordinate of layer 1 as on of layer 2 given the relative position
coordsl1tol2(layer1, layer2,  i1, j1, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); floor.(Int32,(i1, j1) .- dyxz[1:2]) end
coordsl1tol2(layer1, layer2, (i1, j1)::Tuple, alignment = :center) = coordsl1tol2(layer1, layer2, i1, j1, alignment)
# Interprets a coordinate of layer 2 as on of layer 1 given the relative position
coordsl2tol1(layer1, layer2, i2, j2, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); floor.(Int32, (i2, j2) .+ dyxz[1:2]) end
coordsl2tol1(layer1, layer2, (i2, j2)::Tuple, alignment = :center) = coordsl2tol1(layer1, layer2, i2, j2, alignment)
export coordsl1tol2, coordsl2tol1

# put x in between min and max
minmax(lower, x, upper) = max(min(x, upper), lower)

"""
Snaps a coordinate to the nearest edge/corner of a layer
"""
snaptolayer(x, layer) = minmax(1, x, glength(layer))


# TODO: This implementation should only work for 2D layers
# Returns the nearest neighbor of (i1, j1) in layer2's coordinates
# Works by translating the coordinate of layer one to one of layer2, and then snapping
# it to the nearest edge of the layer
function nearest(i1, j1, layer1, layer2, alignment = :center)
    i1to2, j1to2 = coordsl1tol2(layer1, layer2, i1, j1, alignment)

    nny = minmax(1, i1to2, glength(layer2))
    nnx = minmax(1, j1to2, gwidth(layer2))

    return nny, nnx
end
export nearest

"""
Check wether a coordinate is in the layer
"""
function isin(i, j, layer::IsingLayer)
    if (1 <= i <= glength(layer)) && (1 <= j <= gwidth(layer))
        return true
    end

    return false
end

isin((i, j)::NTuple{2,T}, layer) where T = isin(i, j, layer)

function isin(i,j,k,layer::IsingLayer)
    _size = size(layer)
    if (1 <= i <= _size[1]) && (1 <= j <= _size[2]) && (1 <= k <= _size[3])
        return true
    end

    return false
end

isin((i, j, k)::NTuple{3,T}, layer) where T = isin(i, j, k, layer)

function dist2(layer1::IsingLayer, layer2::IsingLayer, i1, j1, i2, j2, alignment::Symbol = :center)
    _, _, dlz = dlayer(layer1, layer2)
    i21, j21 = coordsl2tol1(layer1, layer2, i2, j2, alignment)
    return (i1 - i21)^2 + (j1 - j21)^2 + dlz^2
end

function dist2(layer1::IsingLayer, layer2::IsingLayer, idx1, idx2, alignment::Symbol = :center) 
    i1, j1 = idxToCoord(idx1, glength(layer1))
    i2 ,j2 = idxToCoord(idx2, glength(layer2))

    return @inline dist2(layer1, layer2, i1, j1, i2, j2, alignment)
end

dist(layer1::IsingLayer, layer2::IsingLayer, i1, j1, i2, j2, alignment::Symbol = :center) = sqrt(dist2(layer1, layer2, i1, j1, i2, j2, alignment))

function lattice2_iterator(layer1::IsingLayer, layer2::IsingLayer, i1, j1, NN, alignment = :center)
    # Get a square around the coordinate
    coordsl1 = [(i1+i,j1+j) for i in (-(NN)):(NN), j in (-(NN)):(NN)]
    # Given the relative positions of the layers
    # Give to what coordinates in layer 2 the coordinates in layer 1 correspond
    map!(x -> coordsl1tol2(x, layer1, layer2, alignment), coordsl1, coordsl1)
    # Reshape to a vector
    coordinates2 = reshape(coordsl1, (2*NN+1)^2)
    # Filter out coordinates that are not in layer 2
    filter!(x -> isin(x, layer2), coordinates2)
    # Return the graph indices of the coordinates
    return map(x -> coordToIdx(x, glength(layer2)), coordinates2)
end
"""
Give two layers and a coordinate in the first one
Return the connections of the coordinate in the first layer to the second layer
    by returning the coordinates of the connections and the values of the connections
"""
function lattice2_iterator(layer1::IsingLayer, layer2::IsingLayer, i1, j1, NNi, NNj, prealloc::AbstractPreAlloc, alignment = :center)
    # it = (-(NN)):(NN)
    for j in -NNj:NNj
        for i in -NNi:NNi
            i2, j2 = coordsl1tol2(layer1, layer2, i1+i, j1+j, alignment)
            if isin(i2, j2, layer2)
                push!(prealloc, (i2, j2, coordToIdx(i2, j2, glength(layer2))))
            end
        end
    end
    return
end
# """
# Returns an iterator that creates indexes and coordinates 
#     of the connections of a vertex in layer1 to vertices in layer2
# """
# function lattice2_iterator(layer1, layer2, i1, j1, NN, alignment = :center)
#     return ((i2, j2, coordToIdx(i2, j2, glength(layer2))) )
# end

"""
Get all indices of a vertex with idx vert_idx and coordinates vert_i, vert_j
that are larger than vert_idx
Works in layer indices
"""
function getConnIdxs!(top, vert_idx, vert_i, vert_j, (len, wid)::NTuple{2,Int32}, NNi, NNj, conn_idxs, conn_is, conn_js)
    for j in -NNj:NNj
        for i in -NNi:NNi
            (i == 0 && j == 0) && continue

            if (!periodic(top, :x) && (vert_i + i < 1 || vert_i + i > len)) || 
                (!periodic(top, :y) && (vert_j + j < 1 || vert_j + j > wid))
                continue
            end
            
            conn_i, conn_j = latmod(vert_i + i, vert_j + j, len, wid)

           

            if conn_i == 0 || conn_j == 0
                continue
            end

            conn_idx = coordToIdx(conn_i, conn_j, len)

            conn_idx < vert_idx && continue

            push!(conn_is, conn_i)
            push!(conn_js, conn_j)
            push!(conn_idxs, conn_idx)
        end
    end
end

function getConnIdxs!(topology, vert_idx, coord_vert::NTuple{3,Int32}, size::NTuple{3,Int32}, NNi, NNj, NNk, conn_idxs, conn_is, conn_js, conn_ks)
    for k in -NNk:NNk
        for j in -NNj:NNj
            for i in -NNi:NNi
                
                (i == 0 && j == 0 && k == 0) && continue
                vert_i, vert_j, vert_k = coord_vert
                
                if (!periodic(topology, :x) && (vert_i + i < 1 || vert_i + i > size[1])) || 
                    (!periodic(topology, :y) && (vert_j + j < 1 || vert_j + j > size[2])) ||
                    (!periodic(topology, :z) && (vert_k + k < 1 || vert_k + k > size[3]))
                    continue
                end

                #Apply the periodicity
                conn_i, conn_j, conn_k = latmod((vert_i + i, vert_j + j, vert_k + k), size)
                
                if conn_i == 0 || conn_j == 0 || conn_k == 0
                    continue
                end

                
                #Turn the coordinates into an index
                conn_idx = coordToIdx((conn_i, conn_j, conn_k), size)

                conn_idx < vert_idx && continue

                push!(conn_is, conn_i)
                push!(conn_js, conn_j)
                push!(conn_ks, conn_k)
                push!(conn_idxs, conn_idx)
            end
        end
    end
end


