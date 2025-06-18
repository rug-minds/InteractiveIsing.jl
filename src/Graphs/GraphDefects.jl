mutable struct GraphDefects
    graph::Union{Nothing, IsingGraph}
    hasDefects::Bool
    isDefect::Vector{Bool}
    aliveList::Vector{Int32}
    defectList::Vector{Int32}
    layerdefects::Vector{Int32}
end

function getIterator(gd::GraphDefects)
    it = Int32[]
    for idx in eachindex(gd.isDefect)
        if !gd.isDefect[idx]
            push!(it, idx)
        end
    end
    
    # If everything is alive return whole range
    length(it) == length(gd.isDefect) && return UnitRange{Int32}(1:length(gd.isDefect))

    return it
end
export getIterator

#Extend bas show for graph defects, showing wether there are defects, and if so the sum of isDefect vector
function Base.show(io::IO, defects::GraphDefects)
    if hasDefects(defects)
        print(io, "GraphDefects with $(sum(defects.isDefect)) defects")
    else
        print(io, "GraphDefects with no defects")
    end
end

@setterGetter GraphDefects
# @inline graph(gd::GraphDefects)::IsingGraph = gd.graph
# @inline graph(gd::GraphDefects, g::IsingGraph) = gd.graph = g

#Initialize GraphDefects
GraphDefects(g) = GraphDefects(g, false, Bool[], Int32[], Int32[], Int32[])


# Zip elements from d_idxs into zipList and remove them from removeList using the set and get functions
# Should be a fast way to add elements to an ordered list and remove them from another
function zipAndRemove!(gd, zipListGetSet, removeListGetSet, d_idxs)
    newAddList = zipOrderedLists(zipListGetSet(gd), d_idxs)  # Add d_idxs to the list that needs to be zipped
    newRemoveList = remOrdEls(removeListGetSet(gd), d_idxs)  # Removes them from the other list
    zipListGetSet(gd, newAddList)                            # Set the corresponding lists of the graph
    removeListGetSet(gd, newRemoveList)
end

# import Base: setindex!, getindex
Base.getindex(gd::GraphDefects, idx) = gd.isDefect[idx]

# TODO: Alivelist only matters when starting a loop (?)
# Maybe it's easier to only have the vector with bools
# And then just have a function generate the iterator alivelist on the spot
# This way we don't have to keep track of the alivelist and defectlist


#Set a spin as defect or not
function Base.setindex!(gd::GraphDefects, val, idx::Int32)
    # setting to alive
    if !val
        # If already alive, do nothing
        isDefect(gd)[idx] == val && return val

        # Mark Lattice as having defects
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

    layeridx = spinidx2layer_i_index(graph(gd), idx)
    layerdefects(gd)[layeridx] += val ? 1 : -1

    return isDefect(gd)[idx] = val

end
Base.setindex!(gd::GraphDefects, val, idx::Int64) = setindex!(gd, val, Int32(idx))
Base.setindex!(gd::GraphDefects, val, idxs::AbstractRange) = setrange!(gd, val, collect(idxs))
Base.setindex!(gd::GraphDefects, val, idxs::AbstractVector) = setrange!(gd, val, idxs)
"""
Set a range of spins as defect or not
    val = true -> set as defect
    val = false -> set as alive
This function is faster than setting each spin individually
"""
function setrange!(gd::GraphDefects, val, idxs::AbstractVector)
    if !val
        return setaliverange!(gd, idxs)
    else
        return setdefectrange!(gd, idxs)
    end
end

function setlayerdefects(gd, graph, idxs, defect)
    idxs_startidx = 1
    for (layeridx, graphidxs) in enumerate(layeridxs(graph))
        idxs_startidx = _setlayerdefectsloop(gd, layeridx, graphidxs, idxs_startidx, idxs, defect)
        (idxs_startidx <= (nStates(graph))) && break
    end
    return nothing
end

function _setlayerdefectsloop(gd, layeridx, graphidxs, spin_startidx, idxs, defect)
    defects = 0
    lastidxs_idx = 0
    for idxs_idx in spin_startidx:length(idxs)
            if idxs[idxs_idx] ∈ graphidxs
                defects += 1
            else
                lastidxs_idx = idxs_idx
                break
            end
    end
    layerdefects(gd)[layeridx] += (defect ? 1 : -1)*defects
    return lastidxs_idx
end

# TODO: Does it assume idxs to be ordered? I think so
"""
Set a range of spins as alive
This function is faster than setting each spin individually in a loop
"""
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

    # Set all the defects in the layers
    setlayerdefects(gd, graph(gd), d_idxs, false)

    if length(defectList(gd)) == 0
        hasDefects(gd, false)
    end

    #return amount set
    return -length(d_idxs)
end

"""
Set a range of spins as defect
This function is faster than setting each spin individually
"""
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

    # Mark lattice as having defects
    hasDefects(gd, true)

    # Set all the defects in the layers
    # TODO: NOT SETTING CORRECT
    # setlayerdefects(gd, graph(gd), d_idxs, true)

    #return amount set
    return length(d_idxs)
end

function setdefectrange!(g::IsingGraph, idxs)
    setdefectrange!(defects(g), idxs)
end
function setdefectrange!(l::IsingLayer, idxs)
    d_idxs = setdefectrange!(defects(graph(l)), idxLToG.(idxs, Ref(l)))
    defects(graph(l)).layerdefects[internal_idx(l)] += d_idxs
end
function setaliverange!(l::IsingLayer, idxs)
    d_idxs = setaliverange!(defects(graph(l)), idxLToG.(idxs, Ref(l)))
    defects(graph(l)).layerdefects[internal_idx(l)] -= d_idxs
end
export setdefectrange!, setaliverange!

"""
Reset all the defects
"""
function reset!(gd::GraphDefects)
    hasDefects(gd, false)
    isDefect(gd) .= false
    aliveList(gd, Int32[1:length(isDefect(gd));])
    defectList(gd, Int32[])
    layerdefects(gd) .= 0
    gd
end
export reset!

"""
Add a layer to the graph defects
"""
function addLayer!(gd::GraphDefects, layer)
    layer_idx =  internal_idx(layer)
    _startidx = startidx(layer)
    #set new isdefect vector 
    splice!(gd.isDefect, _startidx:_startidx-1, [false for x in 1:nStates(layer)])
    # gd.isDefect = vcat(gd.isDefect, [false for x in 1:nStates(layer)])

    #set new alive list
    splice!(gd.aliveList, _startidx:_startidx-1, Int32[startidx(layer):(startidx(layer)+nStates(layer)-1);])
    # gd.aliveList = vcat(gd.aliveList, Int32[start(layer):(start(layer)+nStates(layer)-1);])

    insert!(layerdefects(gd), layer_idx, 0)
end

"""
Remove a layer from the graph defects
"""
# NEEDS TO BE CALLED AFTER REMOVING THE LAYER FROM THE GRAPH
function removeLayer!(gd::GraphDefects, lidx)
   
    _graph = graph(gd)
    _layers = unshuffled(layers(_graph))
    preceding_layers = _layers[1:(lidx-1)]
    # Get the number of defects in the layer
    l_ndefect = layerdefects(gd)[lidx]
    nstates_layer = length(gd.isDefect) - sum(nStates.(_layers)) 
    lidx = min(lidx, length(_layers))
    start_idx = startidx(_layers[lidx])

    # TODO: Make lazy
    # Get the number of defects and alives in the preceding layers
    preceding_defects = 0
    preceding_alives = 0
    if !isempty(preceding_layers)
        preceding_defects = sum(ndefect.(preceding_layers))
        preceding_alives = sum(nalive.(preceding_layers))
    end
    

    # Remove defects from defect list
    newdefectlist = Vector{Int32}(undef, length(defectList(gd))-l_ndefect)
    newdefectlist[1:preceding_defects] = defectList(gd)[1:preceding_defects]
    newdefectlist[preceding_defects+1:end] = (defectList(gd)[(preceding_defects+l_ndefect+1):end] .- nstates_layer)

    # Fix alive list
    l_alivelist_length = nstates_layer - l_ndefect
    newalivelist = Vector{Int32}(undef, length(aliveList(gd)) - l_alivelist_length)
    newalivelist[1:preceding_alives] = aliveList(gd)[1:preceding_alives]
    newalivelist[(preceding_alives+1):end] = (aliveList(gd)[(preceding_alives+1 + l_alivelist_length):end] .- nstates_layer)

    gd.defectList = newdefectlist
    gd.aliveList = newalivelist
    deleteat!(gd.isDefect, start_idx:(start_idx+nstates_layer-1))

    # Fix layerdefects
    deleteat!(layerdefects(gd), lidx)

    return
end