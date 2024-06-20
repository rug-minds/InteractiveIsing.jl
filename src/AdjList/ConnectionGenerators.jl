# abstract type SelfType end
# struct Self <: SelfType end
# struct NoSelf <: SelfType end

# function SelfType(wg::WeightGenerator{A,SelfFunc,B,C}) where {A,SelfFunc,B,C}
#     if isa(SelfFunc, Type{Nothing})
#         return NoSelf()
#     else
#         return Self()
#     end
# end
# export SelfType

# abstract type Alignment end
# struct None <: Alignment end
# struct Center <: Alignment end

# Alignment modes
# 1: :none - no alignment (default)
# 2: :center - center layer 2 in layer 1

# Gives a vector of the relative lattice positions
function dlayer(layer1, layer2)::NTuple{3,Int32}
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
coordsl1tol2(i1, j1, layer1, layer2, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); floor.(Int32,(i1, j1) .- dyxz[1:2]) end
coordsl1tol2((i1, j1)::Tuple, layer1, layer2, alignment = :center) = coordsl1tol2(i1, j1, layer1, layer2, alignment)
# Interprets a coordinate of layer 2 as on of layer 1 given the relative position
coordsl2tol1(i2, j2, layer1, layer2, alignment = :center)::Tuple{Int32,Int32} = let dyxz = dcoords(layer1, layer2, alignment); floor.(Int32, (i2, j2) .+ dyxz[1:2]) end
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

dist(i1, j1, i2, j2, layer1, layer2, alignment::Symbol = :center) = sqrt(dist2(i1, j1, i2, j2, layer1, layer2, alignment))

function lattice2_iterator(i1, j1, layer1, layer2, NN, alignment = :center)
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
function lattice2_iterator(layer1, layer2, i1, j1, NN, prealloc::AbstractPreAlloc, alignment = :center)
    it = (-(NN)):(NN)
    for j in it
        for i in it
            i2, j2 = coordsl1tol2(i1+i, j1+j, layer1, layer2, alignment)
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
function getConnIdxs!(::NoSelf, vert_idx, vert_i, vert_j, len, wid, NN, pre_3tuple)
    for j in -NN:NN
        for i in -NN:NN
            (i == 0 && j == 0) && continue
            conn_i, conn_j = latmod(vert_i + i, vert_j + j, len, wid)
            conn_idx = coordToIdx(conn_i, conn_j, len)

            conn_idx < vert_idx && continue

            push!(pre_3tuple, (conn_i, conn_j, conn_idx))
        end
    end
end

"""
Get all indices of a vertex with idx vert_idx and coordinates vert_i, vert_j
that are larger than vert_idx and include self connection
"""
function getConnIdxs!(::Self, vert_idx, vert_i, vert_j, len, wid, NN, pre_3tuple)
    for j in -NN:NN
        for i in -NN:NN
            conn_i, conn_j = latmod(vert_i + i, vert_j + j, len, wid)
            conn_idx = coordToIdx(conn_i, conn_j, len)

            conn_idx < vert_idx && continue

            push!(pre_3tuple, (conn_i, conn_j, conn_idx))
        end
    end
end
