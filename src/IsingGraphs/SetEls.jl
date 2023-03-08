"""
Adding a defect to lattice
"""
@inline function upNDefects(g::IsingGraph, n)
    return
end


# FIX?: Maybe do a try block around first two expressions
function zipAndRemove(g, zipListGetSet, removeListGetSet, d_idxs)
    newAddList = zipOrderedLists(zipListGetSet(g), d_idxs)  # Add d_idxs to the list that needs to be zipped
    newRemoveList = remOrdEls(removeListGetSet(g), d_idxs)  # Removes them from the other list
    zipListGetSet(g, newAddList)                            # Set the corresponding lists of the graph
    removeListGetSet(g, newRemoveList)
end

function addDefects!(sim, g::AbstractIsingGraph, spin_idxs::Vector{T}) where T <: Integer
    # Only keep elements that are not defect already
    d_idxs = @inbounds spin_idxs[map(!,defectBools(g)[spin_idxs])]

    if isempty(d_idxs)
        return
    end
    
    if length(spin_idxs) > 1
        zipAndRemove(g, defectList, aliveList, d_idxs)
    else    #Faster for singular elements
            #Remove item from alive list and start searching backwards from spin_idx
            # Since aliveList is sorted, spin_idx - idx_found gives number of smaller elements in list
        rem_idx = revRemoveSpin!(aliveList(g),spin_idxs[1])
        insert!(defectList(g),spin_idxs[1]-rem_idx+1,spin_idxs[1])
    end

    # Mark corresponding spins to defect
    @inbounds defectBools(g)[d_idxs] .= true

    upNDefects(g)

    # If first defect, mark lattice as containing defects
    if getHParam(htype(g), :Defects) == false
        editHType!(g, :Defects => true)
    end

    # Branchsim since ising_it needs to be reset
    branchSim(sim)
end

"""
Removing defects, insert ordered list!
"""
function remDefects!(sim, g::AbstractIsingGraph, spin_idxs::Vector{T}) where T <: Integer

    # If there were no defects, do nothing
    if getHParam(g, :Defects) == false
        return
    end
    
    # Only keep ones that are actually defect
    d_idxs = @inbounds spin_idxs[defectBools(g)[spin_idxs]]

    if isempty(d_idxs)
        return
    end

    if length(spin_idxs) > 1
        zipAndRemove(g, aliveList, defectList, d_idxs)
    else    #Is faster for singular elements
            # Add to alive list
            # Adds it to original index offset by how many smaller numbers are also removed
        rem_idx = removeFirst!(defectList(g), spin_idxs[1]) 
        insert!(aliveList(g),spin_idxs[1]-(rem_idx-1),spin_idxs[1])
    end

    # Spins not defect anymore
    @inbounds defectBools(g)[d_idxs] .= false

    upNDefects(g)

    # If defectlist is now empty, no defects anymore
    if isempty(defectList(g))
        editHType!(g, :Defects => false)
    end

    # Branchsim since ising_it needs to be reset
    branchSim(sim)
end

# Not used I think?
"""
Removes multiple defects
"""
function remDefects!(sim, g, idxs::Vector{Any})
    for idx in idxs
        remDefect!(sim, g, idx)
    end
end

"""
Setting an alive element
"""
function setNormal!(sim, g::AbstractIsingGraph, spin_idxs::Vector{Int32}, brush)
    remDefects!(sim, g, spin_idxs)
    @inbounds state(graph(g))[spin_idxs] .= brush
end

"""
Set clamped spins, change htype, and branch sim
"""
function setClamp!(sim, g::AbstractIsingGraph, spin_idxs::Vector{Int32}, brush)
    # If no defects before, now has defects
    if getHParam(htype(g), :Defects) == false
        editHType!(g, :Defects => true)
    end

    addDefects!(sim, g, spin_idxs)
    @inbounds state(graph(g))[spin_idxs] .= brush
end

"""
Chooses normal set or clamp
"""
function setOrClamp!(sim, g, idxs , brush, clamp = false)
    if !clamp
        setNormal!(sim, g, idxs, brush)
    else
        setClamp!(sim, g, idxs, brush)
    end
end

# Setting functions for different types of graphs
setGraphSpins!(sim, g::IsingGraph, idxs, brush, clamp) = setOrClamp!(sim, g, idxs, brush, clamp)
setGraphSpins!(sim, g::IsingGraph{Int32}, idxs, brush, clamp) = let (clamp = brush == 0 ? true : clamp) ; setOrClamp!(sim, g, idxs, brush, clamp) end
setGraphSpins!(sim, layer::IsingLayer, idxs, brush, clamp) = setOrClamp!(sim, layer, idxLToG.(Ref(layer), idxs), brush, clamp)

"""
Removes all defects from graph and branch sim
"""
function restoreState!(sim, g)
    (@view state(g)[defectList(g)]) .= rand(length(defectlist(g)))
    remDefects!(sim, g, defectList(g))
    editHType!(g, :Defects => false)
    branchSim(sim)
end
    