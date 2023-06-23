struct AdjList{T} <: AbstractVector{Vector{T}}
    data::Vector{Vector{T}}
    lastaccess::Ref{Union{Nothing,Vector{Int32}}}

    function AdjList(n::Integer)
        data = Vector{Vector{Tuple{Int32,Float32}}}(undef, n)
        for i in 1:n
            data[i] = Vector{Tuple{Int32,Float32}}()
        end
        lastaccess = Ref{Union{Nothing,Vector{Int32}}}(nothing)
        return new{Tuple{Int32,Float32}}(data, lastaccess)
    end
end

lastaccess(adj::AdjList) = adj.lastaccess[]
lastaccess(adj::AdjList, vec) = adj.lastaccess[] = vec

@inline Base.length(adj::AdjList) = length(adj.data)
@inline Base.isempty(adj::AdjList) = isempty(adj.data)
@inline Base.getindex(adj::AdjList, idx) = adj.data[idx]
@inline Base.setindex!(adj::AdjList, val, idx) = adj.data[idx] = val
@inline Base.push!(adj::AdjList, val) = push!(adj.data, val)
@inline Base.pop!(adj::AdjList) = pop!(adj.data)
@inline Base.deleteat!(adj::AdjList,i) = deleteat!(adj.data,i)
@inline Base.first(adj::AdjList) = first(adj.data)
@inline Base.resize!(adj::AdjList, n) = resize!(adj.data, n)
@inline Base.eltype(adj::AdjList) = Vector{Tuple{Int32,Float32}}
@inline Base.size(adj::AdjList) = size(adj.data)

# function fill!(adj::AdjList, NN, weighttransform, distfunc) 


#     increasing_coups = 
# end
function getIncreasingIdxs(layer, idx, NN, ::Type{Periodic})::Vector{Int32}
    l = glength(layer)
    w = gwidth(layer)
    # Maximum neighbors
    vec = Vector{Int32}(undef, (2*NN+1)^2)
    added = 0
    vert_i, vert_j = idxToCoord(idx, l)
    for j in (-NN):NN
        for i in (-NN):(NN)
            if i == 0 && j == 0
                continue
            end

            conn_idx = coordToIdx(latmod(i+vert_i, l), latmod(j+vert_j, w), l)

            if conn_idx > idx
                added += 1
                vec[added] = conn_idx
            end
        end
    end
    resize!(vec, added)
    return sort!(vec)
end
export getIncreasingIdxs

function fillAdj!(layer::AbstractIsingLayer, weightfunc)
    startidx = start(layer)
    eidx = endidx(layer)

    # Get the adjlist
    gadj = adj(graph(layer))
    fNN = NN(weightfunc)

    lastaccess = zeros(Int32, eidx-startidx+1)

    # lastaccess(adj, zeros(Int32, eidx-startidx+1))

    for vert_idx in startidx:eidx

        vert_i, vert_j = idxToCoord(vert_idx, glength(layer))
        for conn_idx in getIncreasingIdxs(layer, vert_idx, fNN, Periodic)
            conn_i, conn_j = idxToCoord(conn_idx, glength(layer))
            dr2 = (vert_i-conn_i)^2+(vert_j-conn_j)^2
            weight = getWeight(weightfunc, sqrt(dr2), (vert_i+conn_i)/2, (vert_j+conn_j)/2)
            
            if weight != 0
                foundidx1, foundidx2 = addWeight!(gadj, vert_idx, conn_idx, weight, sidx1 = lastaccess[vert_idx]+1, sidx2 = lastaccess[conn_idx]+1)
                lastaccess[vert_idx] = foundidx1
                lastaccess[conn_idx] = foundidx2
            end
        end 
    end
end
export fillAdj!

function resetAdjLayer!(layer::AbstractIsingLayer) 
    startidx = start(layer)
    eidx = endidx(layer)

    # Get the adjlist
    gadj = adj(graph(layer))
    for vert_idx in startidx:eidx
        gadj[vert_idx] = Vector{Tuple{Int32,Float32}}()
    end
end
export resetAdjLayer!

@inline function fillEntry!(adj::AdjList, idx, conn_idx, weight)
    vert_i, vert_j = idxToCoord(vert_idx, glength(layer))
        for conn_idx in getIncreasingIdxs(layer, vert_idx, NN, Periodic)
            conn_i, conn_j = idxToCoord(conn_idx, glength(layer))
            weight = weightfunc(sqrt((vert_i-conn_i)^2+(vert_j-conn_j)^2), (vert_i+conn_i)/2, (vert_j+conn_j)/2)
            if weight != 0
                push!(adj[vert_idx], (conn_idx, weight))
            end
    end
end

# Find connection in adjacency list (directed)
function findconn(adj::AdjList, idx, connidx; startidx = 1, endidx = length(adj[idx]))::Tuple{Symbol, Int32}
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


"""
Adds weight to adjacency matrix
"""
function addWeight!(adj::AdjList, idx, conn_idx, weight; sidx1 = 1, sidx2 = 1, eidx1 = length(adj[idx]), eidx2 = length(adj[conn_idx]))
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

function addWeightDirected!(adj::AdjList, idx, conn_idx, weight; sidx = 1, eidx = length(adj[idx]))
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

function removeWeight!(adj::AdjList, idx, conn_idx)
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
function removeWeightDirected!(adj::AdjList, idx, conn_idx; sidx = 1, eidx = length(adj[idx]))
    action, insertidx = findconn(adj, idx, conn_idx; startidx = sidx, endidx = eidx)

    if (action == :Append || action == :Insert)
        return false
    else # action == :Found
        deleteat!(adj[idx], insertidx)
    end

    return insertidx

end

# shiftWeight(tupl, idx_offset) = (tupl[1] + idx_offset, tupl[2])

export addWeight!
export removeWeight!
