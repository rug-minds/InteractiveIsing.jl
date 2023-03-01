using BenchmarkTools
using OrderedCollections
# Go from idx to lattice coordinates, for rectangular grids
@inline function idxToCoord(idx::Integer, length::Integer, width::Integer)
    return ((idx - 1) % length + 1, (idx - 1) ÷ width + 1)
end

# Matrix Coordinates to vector Coordinates
@inline function coordToIdx(i, j, length)::Int32
    return Int32(i + (j - 1) * length)
end

# Insert coordinates as tuple
coordToIdx((i, j), length) = coordToIdx(i, j, length)

# Go from idx to lattice coordinates, for square grids
@inline idxToCoord(idx::Integer, N::Integer) = idxToCoord(idx, N, N)

@inline function latmod(coord, N)
    return mod((coord - 1), N) + 1
end

function createAdj(NN, len, wid, invoke = (;dr, i ,j) -> dr^2, self = false, selfweight = Float32(1))
    adj = [Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]

    fillEmptyAdj(adj, NN, len, wid, invoke)

    if self
        Threads.@threads for (idx,entry) in collect(enumerate(adj))
            push!(entry, (idx, selfweight) )
            sort!(entry)
        end
    else
        Threads.@threads for entry in adj
            sort!(entry)
        end
    end

    return adj
end

function fillEmptyAdj(adj, NN, len, wid, invoke)
    Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
        for dj in 0:(NN)
            for di in (-NN):(NN)
                if di < 1 && dj == 0
                    continue
                end
                vert_i, vert_j = idxToCoord(idx, len, wid)

                conn_i = latmod(vert_i + di, len)
                conn_j = latmod(vert_j + dj, wid)

                weight = invoke(dr = sqrt(di^2+dj^2), i = (vert_i+conn_i)/2, j = (vert_j+conn_j)/2)
                if weight != 0
                    conn_idx = coordToIdx(conn_i,conn_j,len)
                    addWeight!(adj, idx, conn_idx, weight)
                end
            end
        end
    end
end

function addWeight!(adj::Vector{Vector{Tuple{Int32,Float32}}}, idx, conn_idx, weight)
    push!(adj[idx], (conn_idx, weight))
    push!(adj[conn_idx], (idx, weight))
end

insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), [x]); v)::Vector{Tuple{Int32,Float32}}

function getUniqueConnIdxs(idx, NN, len, wid, invoke)
    vec::Vector{Tuple{Int32,Float32}} = []

    for dj in (-NN):(NN)
        for di in 0:(NN)
            if dj < 1 && di == 0
                continue
            end

            vert_i, vert_j = idxToCoord(idx, len, wid)

            conn_i = latmod(vert_i + di, len)
            conn_j = latmod(vert_j + dj, wid)

            weight = Float32(invoke(dr = sqrt(di^2+dj^2), i = (vert_i+conn_i)/2, j = (vert_j+conn_j)/2))
            if weight != 0
                conn_idx = coordToIdx(conn_i,conn_j,len)
                insert_and_dedup!(vec, (conn_idx, weight))
            end
        end
    end

    return vec
end

function getUniqueConnIdxsSelf(idx, NN, len, wid, invoke, selfweight)::Vector{Tuple{Int32,Float32}}
    vec::Vector{Tuple{Int32,Float32}} = []

    for dj in (-NN):(NN)
        for di in 0:(NN)
            if dj < 1 && di == 0
                if dj == 0
                    vert_i, vert_j = idxToCoord(idx, len, wid)
                    sw = selfweight(vert_i, vert_j)
                    insert_and_dedup!(vec, (idx, sw ))
                end
                continue
            end

            vert_i, vert_j = idxToCoord(idx, len, wid)

            conn_i = latmod(vert_i + di, len)
            conn_j = latmod(vert_j + dj, wid)

            weight = Float32(invoke(dr = sqrt(di^2+dj^2), i = (vert_i+conn_i)/2, j = (vert_j+conn_j)/2))
            if weight != 0
                conn_idx = coordToIdx(conn_i,conn_j,len)
                insert_and_dedup!(vec, (conn_idx, weight))
            end
        end
    end

    return vec
end

function createSqAdj(NN, len, wid, invoke = (;dr, i ,j) -> dr^2; self = false, selfweight = (i,j) -> Float32(1))
    adj = [Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]
    
    uniqueIdxFunc(idx, NN, len, wid, invoke) = !self ? getUniqueConnIdxs(idx, NN, len, wid, invoke) : getUniqueConnIdxsSelf(idx, NN, len, wid, invoke, selfweight)

    Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
        idxs_weights = uniqueIdxFunc(idx, NN, len, wid, invoke)
        
        for idx_weight in idxs_weights
            addWeight!(adj, idx, idx_weight[1], idx_weight[2])
        end
    end
    
    return adj
end