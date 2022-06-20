"""
Stuff for initialization of adjacency matrix
"""
__precompile__()

module SquareAdj

export fillAdjList!, numEdges, latmod


# Matrix Coordinates to vector Coordinates
@inline  function coordToIdx(i,j,N)
    return (i-1)*N+j
end

# Insert coordinates as tuple
coordToIdx((i,j),N) = coordToIdx(i,j,N)

# Go from idx to lattice coordinates
@inline function idxToCoord(idx::Int,N)
    return ((idx-1)÷N+1,(idx-1)%N+1)
end

# Put a lattice index (i or j) back onto lattice by looping it around
function latmod(i,N)
    if i < 1 || i > N
        return i = mod((i-1),N) +1
    end
    return i
end


# Input idx, gives all other indexes which it is coupled to. NN is how many nearest neighbors, 
# can set periodic or not
# rfunc is distance function
function adjEntry(idx, N, NN = 1, periodic = true, rfunc = r->1/r^2)::Vector{Tuple{Int32,Float32}}
    (i,j) = idxToCoord(idx, N)
    if periodic
        couple = [(coordToIdx(latmod(i2,N),latmod(j2,N),N), rfunc( sqrt((i-i2)^2 + (j-j2)^2) )) for i2 in (i-NN):(i+NN), j2 in (j-NN):(j+NN) if (!(i2 == i && j2 == j) && ( rfunc( sqrt((i-i2)^2 + (j-j2)^2)) != 0.)  )]
    else
        couple =[ (coordToIdx(i2, j2,N), rfunc( sqrt((i-i2)^2 + (j-j2)^2) )) for i2 in (i-NN):(1+NN), j2 in (j-NN):(j+NN) if ( !(i2 == i && j2 == j) && (i2 > 0 && j2 > 0 && i2 <= N && j2 <= N) && (rfunc( sqrt((i-i2)^2 + (j-j2)^2) ) != 0.) )]
    end

    return couple
end

# Should also include function!!!!
# Init the adj list of g
function fillAdjList!(adj, N , NN = 1; periodic = true, rfunc = r-> r == 1 ? 1. : 0.)

    for idx in 1:N*N
        adj[idx] = adjEntry(idx, N, NN, periodic, rfunc)
    end
    
end


""" New Implementation """

"""
Stuff for initialization of adjacency matrix
"""



""" Old functions """

# Count number of edges in adjacency matrix
function numEdges(adj::Vector)
    num = 0
    for (in,outs) in enumerate(adj)
        for out in outs
            if in <= out
                num += 1
            end
        end
    end
    return num 
end

# Returns the indices of all the spins a particular spin is connected with
function coupleIndices(i,j,N)
    
    # Left right up down
    # Sets any of the above to false if spins are at the edge of lattice
    # Representing that there are no spins to that side anymore
    l,r,u,d = (true,true,true,true)
    if j == 1 
        l = false 
    end
    if j == N
        r = false 
    end
    if i == 1 
        u = false 
    end
    if i == N 
        d = false 
    end
    
    # Filters nearest neighbors that are outside of grid
    """lu u ru r rd d dl l"""
    idxs = [l && u, u , r && u , r , r && d, d, d && l , l]
    cn = (y, x) -> coordToIdx(y,x,N)
    nn = [cn(i-1,j-1),cn(i-1,j),cn(i-1,j+1),cn(i,j+1),cn(i+1,j+1),cn(i+1,j),cn(i+1,j-1),cn(i,j-1)]
    return nn[idxs]                    
end
       
# Fills adjacency list for a square grid
function fillAdjListOld!(adj,N)
    size = N*N
    
    @inbounds for idx in 1:size
        adj[idx] = []
    end

    @inbounds for idx in 1:size
        i,j = idxToCoord(idx,N)
        adj[idx] = coupleIndices(i,j,N)
    end
end

# Connection rule for lattice, not used right now
function conn_rule(i,j, N)
    i1,j1 = idxToCoord(i,N)
    i2,j2 = idxToCoord(j,N)
    
    # If any of the distances is greater than 1, not nearest neighbor
    if max(abs(i1-i2),abs(j1-j2)) > 1 || i==j
        return False
    end
    
    return True

end

end