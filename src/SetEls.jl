# module SetEls

# Setting Elements and defects 

# Removing and adding defects, and clamping

# Adding a defect to lattice
function addDefects!(sim, g::IsingGraph, spin_idxs::Vector{T}) where T <: Integer
    # Only keep elements that are not defect already
    # @inbounds d_idxs::Vector{Int32} = spin_idxs[map(!,g.d.defectBools[spin_idxs])]
    d_idxs = spin_idxs[map(!,g.d.defectBools[spin_idxs])]

    if isempty(d_idxs)
        return
    end
    
    if length(spin_idxs) > 1
        try
            newdefectList = zipOrderedLists(g.d.defectList,d_idxs)  #Add els to defectlist
            newaliveList = remOrdEls(g.d.aliveList,d_idxs) #Remove them from alivelist
            g.d.defectList = newdefectList
            g.d.aliveList = newaliveList
        catch
            println("Aborting adding defects")
            return
        end
    else #Faster for singular elements

        #Remove item from alive list and start searching backwards from spin_idx
        # Since aliveList is sorted, spin_idx - idx_found gives number of smaller elements in list
        rem_idx = revRemoveSpin!(g.d.aliveList,spin_idxs[1])
        insert!(g.d.defectList,spin_idxs[1]-rem_idx+1,spin_idxs[1])
    end

    # Mark corresponding spins to defect
    @inbounds g.d.defectBools[d_idxs] .= true

    # If first defect, mark lattice as containing defects
    if getHParam(g, :Defects) == false
        editHType!(g, :Defects => true)
        branchSim(sim)
    end

end

# Removing defects, insert ordered list!
function remDefects!(sim, g::IsingGraph, spin_idxs::Vector{T}) where T <: Integer
    
    # Only keep ones that are actually defect
    @inbounds d_idxs = spin_idxs[g.d.defectBools[spin_idxs]]
    if isempty(d_idxs)
        return
    end

    # Remove defects from defect list and add to aliveList
    # Assumes that els are in list!
    if length(spin_idxs) > 1
        try
            newaliveList = zipOrderedLists(g.d.aliveList, d_idxs)
            newdefectList = remOrdEls(g.d.defectList,d_idxs)
            g.d.aliveList = newaliveList
            g.d.defectList = newdefectList
        catch
            println("Aborting removing defects")
            return
        end
            

        
    else    #Is faster for singular elements
        # Add to alive list
        # Adds it to original index offset by how many smaller numbers are also removed
        rem_idx = removeFirst!(g.d.defectList,spin_idxs[1]) 
        insert!(g.d.aliveList,spin_idxs[1]-(rem_idx-1),spin_idxs[1])
    end

    # Spins not defect anymore
    @inbounds g.d.defectBools[d_idxs] .= false

    if isempty(g.d.defectList) && getHParam(g, :Defects) == true
        editHType!(g, :Defects => false)
        branchSim(sim)
    end

end

remDefect!(sim, g, spin_idx::T) where T <: Integer = remDefects!(sim, g,[spin_idx]) 
addDefect!(sim, g, spin_idx::T) where T <: Integer = addDefects!(sim, g,[spin_idx]) 

# Lattice indexing
addDefect!(sim, g,i,j) = addDefect!(sim, g,coordToIdx(i,j,g.N))
remDefect!(sim, g,i,j) = remDefect!(sim, g,coordToIdx(i,j,g.N))

# Removes Multiple Defects
# Not used
function remDefects!(sim, g,idxs::Vector{Any})
    for idx in idxs
        remDefect!(sim, g,idx)
    end
end

# Removes All defects
function restoreState!(sim, g)
    g.state[g.d.defectList] = rand(length(defectlist))
    remDefects!(g,g.d.defectList)
    
    editHType!(g, :Defects => false)
    branchSim(sim)
end
    
# Setting Elements

# Backend 

# Setting an alive element
function setNormal!(sim, g::AbstractIsingGraph, spin_idxs::Vector{Int32} , brush)
    # First remove defect if it was defect
    remDefects!(sim, g,spin_idxs)
    # Then set element
    @inbounds g.state[spin_idxs] .= brush
end

setNormal!(sim, g,spin_idx::Integer,brush) = setNormal!(sim, g, [spin_idx], brush)
setNormal!(sim, g,i::Integer,j::Integer,brush) =  setNormal!(sim, g,Int32.(coordToIdx(i,j,g.N)),brush)
setNormal!(sim, g,tupls::Vector{Tuple{Int16,Int16}},brush) = setNormal!(sim, g,Int32.(coordToIdx.(tupls,g.N)),brush)

function setClamp!(sim, g::IsingGraph, spin_idxs::Vector{Int32} , brush)
    addDefects!(sim, g,spin_idxs)
    @inbounds g.state[spin_idxs] .= brush

    editHType!(g, :Defects => true)
    branchSim(sim)
end

setClamp!(sim, g,spin_idx::Integer,brush) = setClamp!(sim, g, [spin_idx], brush)
setClamp!(sim, g,i::Integer,j::Integer,brush) =  setClamp!(sim, g,Int32.(coordToIdx(i,j,g.N)),brush)
setClamp!(sim, g,tupls::Vector{Tuple{Int16,Int16}},brush) = setClamp!(sim, g, Int32.(coordToIdx.(tupls,g.N)),brush)

# User Functions 

# Set spins either to a value or clamp them
function setSpins!(sim, g::IsingGraph{Int8}, idxs , brush, clamp = false)
    # Always clamp if brush is zero, otherwise only if clamping
    if brush != 0 && !clamp
        setNormal!(sim, g, idxs, brush)
    else
        setClamp!(sim, g, idxs, brush)
    end
end

function setSpins!(sim, g::IsingGraph{Float32}, idxs , brush, clamp = false)
    if !clamp
        setNormal!(sim, g, idxs,brush)
    else
        setClamp!(sim, g, idxs,brush)
    end
end

setSpin!(sim, g::IsingGraph, i::Integer, j::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(sim, g, [coordToIdx(i,j,g.N)], brush, clamp)

setSpin!(sim, g::IsingGraph, idx::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(sim, g,[idx],brush,clamp)

setSpins!(sim, g::IsingGraph, tupls::Vector{Tuple{Int32,Int32}}, brush, clamp) = setSpins!(sim, g, coordToIdx.(tupls,g.N), brush, clamp)


# end
