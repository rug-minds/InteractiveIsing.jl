#=
Stuff for initialization of adjacency matrix
=#
 
export fillAdjList!, numEdges, latmod, adjToMatrix

"""
Creates a square adjacency matrix (NN must be smaller than width/2 & length/2)
"""
function createSqAdj(len, wid, weightfunc = defaultIsingWF)
    adj = [Vector{Tuple{Int32,Float32}}() for _ in 1:(len*wid)]

    Threads.@threads for idx in eachindex(adj)
    # for idx in eachindex(adj)
        idxs_weights = getUniqueConnIdxs(weightfunc, idx, len, wid)

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


function setAdj!(sim, layer, wf)
    g = sim.layers[layer]
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