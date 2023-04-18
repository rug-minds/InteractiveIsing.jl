#temp for testing
Vert = Int32

mutable struct GraphDefects
    g::IsingGraph
    hasDefects::Bool
    isDefect::Vector{Bool}
    aliveList::Vector{Vert}
    defectList::Vector{Vert}
end

#Extend bas show for graph defects, showing wether there are defects, and if so the sum of isDefect vector
function Base.show(io::IO, defects::GraphDefects)
    if hasDefects(defects)
        print(io, "GraphDefects with $(sum(defects.isDefect)) defects")
    else
        print(io, "GraphDefects with no defects")
    end
end

@setterGetter GraphDefects

#Initialize GraphDefects
GraphDefects(g) = GraphDefects(g, false, Bool[], Int32[], Int32[])

function layerdefects(gd::GraphDefects)
    # Init empty layer vector
    vec = Vector{LayerDefects}(undef, length(layers(g(gd))) )
    for (idx, layer) in enumerate(layers( g(gd)) )
        vec[idx] = defects(layer)
    end

    return vec
end

# Zip elements from d_idxs into zipList and remove them from removeList using the set and get functions
# Should be a fast way to add elements to an ordered list and remove them from another
function zipAndRemove!(gd, zipListGetSet, removeListGetSet, d_idxs)
    newAddList = zipOrderedLists(zipListGetSet(gd), d_idxs)  # Add d_idxs to the list that needs to be zipped
    newRemoveList = remOrdEls(removeListGetSet(gd), d_idxs)  # Removes them from the other list
    zipListGetSet(gd, newAddList)                            # Set the corresponding lists of the graph
    removeListGetSet(gd, newRemoveList)
end

import Base: setindex!, getindex
getindex(gd::GraphDefects, idx) = gd.isDefect[idx]


#Set a spin as defect or not
function setindex!(gd::GraphDefects, val, idx::Int32)
    # setting to alive
    if !val
        # If already alive, do nothing
        isDefect(gd)[idx] == val && return val

        # Lattice has defects
        hasDefects(gd, true)

        # Remove first item that matches the index and add it to the alive list
        rem_idx = removeFirst!(defectList(gd),  idx) 
        insert!(aliveList(gd), idx-(rem_idx-1), idx)
    else 
        # If already defect, do nothing
        isDefect(gd)[idx] == val && return val

        # Remove item from alive list and start searching backwards from idx
        rem_idx = revRemoveSpin!(aliveList(gd), idx)

        insert!(defectList(gd), idx-rem_idx+1, idx)

        # If no defects left, mark lattice as not containing defects
        if length(defectList(gd)) == 0
            hasDefects(gd, false)
        end
    end

    return isDefect(gd)[idx] = val

end
setindex!(gd::GraphDefects, val, idx::Int64) = setindex!(gd, val, Int32(idx))


setrange!(gd::GraphDefects, val, idxs::Vector{Int64}) = setrange!(gd, val, Int32.(idxs))

function setrange!(gd::GraphDefects, val, idxs::Vector{Int32})::Int32
    if !val
        return setaliverange!(gd, idxs)
    else
        return setdefectrange!(gd, idxs)
    end
end

function setaliverange!(gd::GraphDefects, idxs)::Int32
    # If there were no defects, do nothing
    if !hasDefects(gd)
        return 0
    end
    
    # Only keep ones that are actually defect
    d_idxs = @inbounds idxs[isDefect(gd)[idxs]]


    # If all are already alive, do nothing
    if isempty(d_idxs)
        return 0
    end
    
    # Remove from defect list
    zipAndRemove!(gd, aliveList, defectList, d_idxs)

    # Spins not defect anymore
    @inbounds isDefect(gd)[d_idxs] .= false

    if length(defectList(gd)) == 0
        hasDefects(gd, false)
    end

    #return amount set
    return -length(d_idxs)
end

function setdefectrange!(gd::GraphDefects, idxs)::Int32
    # Only keep elements that are not defect already
    d_idxs = @inbounds idxs[map(!,isDefect(gd)[idxs])]

    # If all are already defect, do nothing
    if isempty(d_idxs)
        return 0
    end
            
    # Add to defect list
    zipAndRemove!(gd, defectList, aliveList, d_idxs)

    # Mark corresponding spins to defect
    @inbounds isDefect(gd)[d_idxs] .= true

    hasDefects(gd, true)

    #return amount set
    return length(d_idxs)
end

function reset!(gd::GraphDefects)
    hasDefects(gd, false)
    isDefect(gd) .= false
    aliveList(gd, Int32[1:length(isDefect(gd));])
    defectList(gd, Int32[] )
    for layer in layerdefects(gd)
        reset!(layer)
    end
end

function addLayer!(gd::GraphDefects, layer)
    #set new isdefect vector 
    gd.isDefect = vcat(gd.isDefect, [false for x in 1:nStates(layer)])

    #set new alive list
    gd.aliveList = vcat(gd.aliveList, Int32[start(layer):(start(layer)+nStates(layer)-1);])

end

function removeLayer!(gd::GraphDefects, lidx)
    layer = layers(g(gd))[lidx]

    l_ndefects = ndefects(layer)

    preceding_defects = precedingDefects(defects(layer))
    preceding_alives = precedingAlives(defects(layer))

    # Remove defects from defect list
    newdefectlist = Vector{Int32}(undef, length(defectList(gd))-l_ndefects)
    newdefectlist[1:preceding_defects] = defectList(gd)[1:preceding_defects]
    newdefectlist[preceding_defects+1:end] = (defectList(gd)[(preceding_defects+l_ndefects+1):end] .- nStates(layer))

    # Fix alive list
    l_alivelist_length = nStates(layer) - l_ndefects
    newalivelist = Vector{Int32}(undef, length(aliveList(gd)) - l_alivelist_length)
    newalivelist[1:preceding_alives] = aliveList(gd)[1:preceding_alives]
    newalivelist[(preceding_alives+1):end] = (aliveList(gd)[(preceding_alives+1 + l_alivelist_length):end] .- nStates(layer))


    gd.defectList = newdefectlist
    gd.aliveList = newalivelist

    return
    # # Removing defects
    # ndefects = collectNumDefects(layervec)
    # newdefects = remPartitionAscendingList(defectList(g), ndefects, layeridx, nStates(layer))
    # defectList(g, newdefects)

    # # Fixing aliveList
    # nalives = nStates.(layervec) .- ndefects
    # newalives = remPartitionAscendingList(aliveList(g), nalives, layeridx, nStates(layer))
    # aliveList(g, newalives)
end

"""
Takes an ascending list that is partitioned
Removes one of the partitions, and shifts al elements after the partition
down by an amount of the maxsize of the partition
|part1|part2|part3| -> |part1|part3 .- maxsize(part2)|
"""
function remPartitionAscendingList(list, npartitions, partitionidx, maxSize)
    nElementsToRemove = npartitions[partitionidx]
    startidx = 1 + sum(npartitions[1:(partitionidx-1)])
    endidx = startdx + nElementsToRemove
    removeEntries(list, startidx, endidx, x -> x - maxSize)
end

"""
From vector of layers for one underlying graph
collect the number of defects
"""
function collectNumDefects(layers)
    defectvec = Vector{Int32}(undef, length(layers))
    for (idx,layer) in enumerate(layers)
        defectvec[idx] = ndefects(layer)
    end
    return defectvec
end