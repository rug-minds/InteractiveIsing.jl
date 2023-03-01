using BenchmarkTools
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

function createAdj(NN, len, wid, invoke, self = false, selfweight = Float32(1))
    adj = [Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]

    Threads.@threads for idx in eachindex(adj)
        for dj in 0:(NN)
            for di in (-NN):(NN)
                if di < 1 && dj == 0
                    continue
                end
                i, j = idxToCoord(idx, len, wid)

                conn_i = latmod(i + di, len)
                conn_j = latmod(j + dj, wid)

                weight = invoke(sqrt(di^2+dj^2), (i+conn_i)/2, (j+conn_j)/2)
                conn_idx = coordToIdx(conn_i,conn_j,len)

                push!(adj[idx], (conn_idx, weight))
                push!(adj[conn_idx], (idx, weight))
            end
        end
    end

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