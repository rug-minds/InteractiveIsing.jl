#=
Stuff for initialization of adjacency matrix
=#
 
export fillAdjList!, numEdges, adjToMatrix

"""
Creates a square adjacency matrix (NN must be smaller than width/2 & length/2)
"""
function createSqAdj(len, wid, weightfunc = defaultIsingWF)
    adj = Vector{Tuple{Int32,Float32}}[Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]

    
    Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
        idxs_weights = getUniqueConnIdxs(weightfunc, idx, len, wid, wt(weightfunc))

        for idx_weight in idxs_weights
            addWeight!(adj, idx, idx_weight[1], idx_weight[2])
        end
    end
    
    return adj
end

export createSqAdj

# TODO: COMPLETELY CHANGE THIS!
getUniqueConnIdxs(
    wf, 
    idx, 
    len, 
    wid, 
    wt::WeightType{A,B,Periodic,false}) where {A,B,Periodic} =
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
    wt::WeightType{A,B,Periodic,true}) where {A,B,Periodic} = 
    
    uniqueIdxsInner(idx, len, wid, NN(wf),
        (dr,i,j) -> getWeight(wf, dr, i ,j),
        selfwfunc = (i,j) -> 1,
        # selfwfunc = (selfweight(wf)),
        shapefunc =
        (dj, di, idx, vec) -> 
            (
                if di < 1 && dj == 0
                    if dj == 0
                        vert_i, vert_j = idxToCoord(idx, len)
                        # sw = selfwfunc(vert_i, vert_j)
                        sw = (i,j) -> 1
                        push!(vec, (idx, sw ))
                    end
                    return true
                end;
                return false
            )
    )

function uniqueIdxsInner(idx, len, wid, NN, getWeight; selfwfunc = (i,j) -> 1, shapefunc)
    vec::Vector{Tuple{Int32,Float32}} = []

    for dj in 0:(NN)
        for di in (-NN):(NN)
            if shapefunc(dj, di, idx, vec)
                continue
            end
            
            vert_i, vert_j = idxToCoord(idx, len)

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
    adj(g) = initSqAdj(glength(g), gwidth(g), weightfunc = wf)
    restart(g)
end
export setAdj!

# Doesn't work with layers
function setGAdj!(sim, idx, weightfunc)
    g = gs(sim)[idx]
    adj(graph(g))[iterator(g)] = initSqAdj(glength(g), gwidth(g); weightfunc)
    restart(g)
end
export setGAdj!



#TODO:: MOVE THIS TO A DIFFERENT FILE
# MAKE ADJ it's own type

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

    return :Push, length(adj[idx])+1
end
export findconn

"""
Adds weight to adjacency matrix
"""
function addWeight!(adj::Vector, idx, conn_idx, weight; sidx1 = 1, sidx2 = 1, eidx1 = nothing, eidx2 = nothing)
    
    isnothing(eidx1) && (eidx1 = length(adj[idx]))
    action, insertidx1 = findconn(adj, idx, conn_idx, startidx = sidx1, endidx = eidx1)

    if action == :Push
        push!(adj[idx], (conn_idx, weight))
    elseif action == :Insert
        insert!(adj[idx], insertidx1, (conn_idx, weight))
    else #if found
        adj[idx][insertidx1] = (conn_idx, weight)
    end

    isnothing(eidx2) && (eidx2 = length(adj[conn_idx]))
    action, insertidx2 = findconn(adj, conn_idx, idx, startidx = sidx2, endidx = eidx2)

    if action == :Push
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

    if action == :Push
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

    if (action == :Push || action == :Insert)
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

    if (action == :Push || action == :Insert)
        return false
    else # action == :Found
        deleteat!(adj[idx], insertidx)
    end

    return insertidx

end

shiftWeight(tupl, idx_offset) = (tupl[1] + idx_offset, tupl[2])

function clearAdj!(adj::Vector{Vector{Tuple{Int32,Float32}}})
    for vert_idx in eachindex(adj)
        adj[vert_idx] = Vector{Tuple{Int32,Float32}}()
    end
end
export clearAdj!

export addWeight!
export removeWeight!

function sparse2tuples(sp_adj)
    vec = Vector{Vector{Tuple{Int32,Float32}}}(undef, size(sp_adj,2))

    for vert_idx in eachindex(vec)
        vec[vert_idx] = Vector{Tuple{Int32,Float32}}(undef, length(nzrange(sp_adj, vert_idx)))
        for conn_idx in 1:length(nzrange(sp_adj, vert_idx))
            vec[vert_idx][conn_idx] = (sp_adj.rowval[nzrange(sp_adj, vert_idx)[conn_idx]], sp_adj.nzval[nzrange(sp_adj, vert_idx)[conn_idx]])
        end
    end
    return deepcopy(vec)
end
export sparse2tuples