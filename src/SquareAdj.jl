#=
Stuff for initialization of adjacency matrix
=#
 
export fillAdjList!, numEdges, latmod, adjToMatrix

"""
Creates a square adjacency matrix (NN must be smaller than width/2 & length/2)
"""
function createSqAdj(len, wid, weightFunc = defaultIsingWF)
    adj = [Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]

    Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
        idxs_weights = getUniqueConnIdxs(weightFunc, idx, len, wid)

        for idx_weight in idxs_weights
            addWeight!(adj, idx, idx_weight[1], idx_weight[2])
        end
    end
    
    return adj
end

export createSqAdj

getUniqueConnIdxs(
    wf, 
    idx, 
    len, 
    wid, 
    wt::WeightType = wt(wf)) =
    uniqueIdxsInner(idx, len, wid, NN(wf), 
        (dr,i,j) -> getWeight(wf, dr, i ,j), 
        shapefunc = (dj,di, idx, vec) -> 
            (
                if di < 1 && dj == 0
                    return true
                end;
                return false
            )
    )


getUniqueConnIdxs(
    wf, 
    idx,
    len, 
    wid, 
    wt::WeightType{A,B,Periodic,true} = wt(wf)) where {A,B,Periodic} = 
    
    uniqueIdxsInner(idx, len, wid, NN(wf),
        (dr,i,j) -> getWeight(wf, dr, i ,j),
        selfweight = selfweight(wf),
        shapefunc =
        (dj, di, idx, vec) -> 
            (
                if di < 1 && dj == 0
                    if dj == 0
                        vert_i, vert_j = idxToCoord(idx, len, wid)
                        sw = selfweight(vert_i, vert_j)
                        push!(vec, (idx, sw ))
                    end
                    return true
                end;
                return false
            )
    )

function uniqueIdxsInner(idx, len, wid, NN, getWeight; selfweight = (i,j) -> 1, shapefunc)
    vec::Vector{Tuple{Int32,Float32}} = []

    for dj in 0:(NN)
        for di in (-NN):(NN)
            if shapefunc(dj, di, idx, vec)
                continue
            end
            
            vert_i, vert_j = idxToCoord(idx, len, wid)

            conn_i = latmod(vert_i + di, len)
            conn_j = latmod(vert_j + dj, wid)

            weight = Float32(getWeight(sqrt(di^2+dj^2), (vert_i+conn_i)/2, (vert_j+conn_j)/2))
            if weight != 0
                conn_idx = coordToIdx(conn_i,conn_j,len)
                push!(vec, (conn_idx, weight))
            end
        end
    end

    return vec
end
export getUniqueConnIdxs


"""
Reads an adjacency list as a matrix
"""
function adjToMatrix(adj, length, width)
    matr = Matrix{Float32}(undef, length, width)
    for (idx, tupls) in enumerate(adj)
        for tupl in tupls
            matr[idx, tupl[1]] = tupl[2]
        end
    end
    return matr
end

adjToMatrix(g) = adjToMatrix(adj(g), glength(g), gwidth(g))


function setAdj!(sim, layeridx, wf)
    g = sim.layers[layeridx]
    adj(g) = initSqAdj(glength(g), gwidth(g), weightFunc = wf)
    refreshSim(sim)
end
export setAdj!

# Doesn't work with layers
function setGAdj!(sim, idx, weightFunc)
    g = gs(sim)[idx]
    adj(graph(g))[iterator(g)] = initSqAdj(glength(g), gwidth(g); weightFunc)
    refreshSim(sim)
end
export setGAdj!

# Find connection in adjacency list (directed)
function findconn(adj, idx, connidx; startidx = 1, endidx = length(adj[idx]))::Tuple{Symbol, Int32}
    for i in startidx:endidx
        conn = adj[idx][i]
        if connIdx(conn) < connidx
            continue
        elseif connIdx(conn) > connidx
            return :Insert , i
        elseif connIdx(conn) == connidx
            return :Found , i
        end
    end

    return :Append, length(adj[idx])
end
export findconn

"""
Adds weight to adjacency matrix
"""
function addWeight!(adj::Vector, idx, conn_idx, weight; sidx1 = 1, sidx2 = 1, eidx1 = length(adj[idx]), eidx2 = length(adj[conn_idx]))
    action, insertidx1 = findconn(adj, idx, conn_idx, startidx = sidx1, endidx = eidx1)

    if action == :Append
        push!(adj[idx], (conn_idx, weight))
    elseif action == :Insert
        insert!(adj[idx], insertidx1, (conn_idx, weight))
    else #if found
        adj[idx][insertidx1] = (conn_idx, weight)
    end

    action, insertidx2 = findconn(adj, conn_idx, idx, startidx = sidx2, endidx = eidx2)

    if action == :Append
        push!(adj[conn_idx], (idx, weight))
    elseif action == :Insert
        insert!(adj[conn_idx], insertidx2, (idx, weight))
    else
        adj[conn_idx][insertidx2] = (idx, weight)
    end

    return insertidx1, insertidx2
end

function addWeightDirected!(adj::Vector, idx, conn_idx, weight; sidx = 1, eidx = length(adj[idx]))
    action, insertidx = findconn(adj, idx, conn_idx, startidx = sidx, endidx = eidx)

    if action == :Append
        push!(adj[idx], (conn_idx, weight))
    elseif action == :Insert
        insert!(adj[idx], insertidx, (conn_idx, weight))
    else #if found
        adj[idx][insertidx] = (conn_idx, weight)
    end

    return insertidx
end

function addWeightOld!(adj::Vector, idx, conn_idx, weight)
    insert_and_dedup!(adj[idx], (conn_idx, weight))
    insert_and_dedup!(adj[conn_idx], (idx, weight))
end
export addWeightOld!   

function removeWeight!(adj::Vector, idx, conn_idx)
    action, insertidx = findconn(adj, idx, conn_idx)

    if (action == :Append || action == :Insert)
        return
    else # action == :Found
        deleteat!(adj[idx], insertidx)
    end

    _, insertidx = findconn(adj, conn_idx, idx)

    deleteat!(adj[conn_idx], insertidx)

end

# Only removes the weight to conn_idx from adj[idx]
# Only to be used if adj[idx] is later removed
function removeWeightDirected!(adj::Vector, idx, conn_idx; sidx = 1, eidx = length(adj[idx]))
    action, insertidx = findconn(adj, idx, conn_idx; startidx = sidx, endidx = eidx)

    if (action == :Append || action == :Insert)
        return false
    else # action == :Found
        deleteat!(adj[idx], insertidx)
    end

    return insertidx

end

shiftWeight(tupl, idx_offset) = (tupl[1] + idx_offset, tupl[2])

export addWeight!
export removeWeight!
