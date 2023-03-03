function connectMatrix(NN)::Matrix{Tuple{Tuple{Int16,Int16},Float32}}
    m = Matrix{Tuple{Tuple{Int16,Int16},Float32}}(undef, 2*NN + 1, 2*NN+1)
    mid = NN+1
    idx = 1
    for j in 1:(2NN+1)
        for i in 1:(2*NN+1)
            m[i,j] = ((i,j),sqrt((i-mid)^2+(j-mid)^2))
            idx +=1
        end
    end
    return m
end

shiftCoord(tup, di, dj) = ((Int16(tup[1][1]+di), Int16(tup[1][2]+dj)), Float32(tup[2]))

@inline function latmod(idx, N)
    return mod((idx - 1), N) + 1
end

# Go from idx to lattice coordinates, for rectangular grids
@inline function idxToCoord(idx::Integer, length::Integer, width::Integer)
    return ((idx - 1) % length + 1, (idx - 1) ÷ length + 1)
end

# Go from idx to lattice coordinates, for square grids
@inline idxToCoord(idx::Integer, N::Integer) = idxToCoord(idx, N, N)

# Matrix Coordinates to vector Coordinates
@inline function coordToIdx(i, j, length)::Int32
    return Int32(i + (j - 1) * length)
end

# Insert coordinates as tuple
coordToIdx((i, j), length) = coordToIdx(i, j, length)

mapOutside(tup::Tuple, N) = tup[1] < N ? tup : (tup[1] , 0)

function invTuple(tup, inv, length, periodic = true)::Tuple{Int32,Float32}
    if periodic 
        return (coordToIdx(latmod.(tup[1],length), length), inv(tup[2], tup[1]...))
    else
        return mapOutside.((coordToIdx(latmod.(tup[1],length), length), inv(tup[2], tup[1]...)), length)
    end
end

function fastPrune(mat)::Vector

    function pruneLoop!(matidxs, matidx, weight)
        if weight == 0
            push!(matidxs, matidx)
        end
    end

    matidxs = []
    for (matidx,conn) in enumerate(mat)
        pruneLoop!(matidxs, matidx, conn[2])
    end

    newvec = Vector(undef, length(mat) - length(matidxs))

    newvecidx = 1
    for el in mat
        if el[2] == 0
            continue
        else
            newvec[newvecidx] = el
            newvecidx += 1
        end
    end
    return newvec
end

function fastPrune!(mat)
    idx = 1
    while idx <= length(mat)
        if mat[idx][2] == 0
            deleteat!(mat,idx)
        end
        idx += 1
    end
end

function invMatrix(mat, inv, length, periodic = true)
    reintmat = reinterpret(Tuple{Int32,Float32}, mat)
    for idx in eachindex(mat)
        reintmat[idx] =  (coordToIdx(latmod.(mat[idx][1], length), length), inv(mat[idx][2],mat[idx][1]...))
    end
    return reintmat
end

function makeAdj(NN, len, wid, invoke = (dr, i, j) -> isodd(i) ? 0 : dr^2)
    connMat::Matrix{Tuple{Tuple{Int16,Int16},Float32}} = connectMatrix(NN)
    
    adj = Vector(undef, len*wid)

    # Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
    #     coords = idxToCoord(idx, len, wid) .- (Int16(NN+1), Int16(NN+1))
    #     newmat = invTuple.(shiftCoord.(connMat, coords...), invoke , len)

    #     fastPrune!(reshape(newmat, length(newmat)))

    #     adj[idx] = newmat
    # end

    Threads.@threads for idx in eachindex(adj)
        mat = copy(connMat)
        coords = idxToCoord(idx, len, wid) .- (Int16(NN+1), Int16(NN+1))

        newmat = invMatrix(shiftCoord.(mat, coords...), invoke, len)

        fastPrune!(reshape(mat, length(mat)))

        adj[idx] = newmat
    end

    return adj
end



