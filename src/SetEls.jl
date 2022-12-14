# module SetEls

# Setting Elements and defects 

# Removing and adding defects, and clamping

# Adding a defect to lattice
function addDefects!(sim, g::AbstractIsingGraph, spin_idxs::Vector{T}) where T <: Integer
    # Only keep elements that are not defect already
    # @inbounds d_idxs::Vector{Int32} = spin_idxs[map(!,defectBools(g)[spin_idxs])]
    d_idxs = spin_idxs[map(!,defectBools(g)[spin_idxs])]

    if isempty(d_idxs)
        return
    end
    
    if length(spin_idxs) > 1
        try
            newdefectList = zipOrderedLists(defectList(g), d_idxs)  #Add els to defectlist
            newaliveList = remOrdEls(aliveList(g), d_idxs) #Remove them from alivelist
            defectList(g, newdefectList)
            aliveList(g, newaliveList)
        catch(err)
            println("Aborting adding defects")
            throw(err)
            # error(err)
            return
        end
    else #Faster for singular elements

        #Remove item from alive list and start searching backwards from spin_idx
        # Since aliveList is sorted, spin_idx - idx_found gives number of smaller elements in list
        rem_idx = revRemoveSpin!(aliveList(g),spin_idxs[1])
        insert!(defectList(g),spin_idxs[1]-rem_idx+1,spin_idxs[1])
    end

    # Mark corresponding spins to defect
    @inbounds defectBools(g)[d_idxs] .= true

    # If first defect, mark lattice as containing defects
    if getHParam(htype(g), :Defects) == false
        editHType!(g, :Defects => true)
    end
    branchSim(sim)
end

# Removing defects, insert ordered list!
function remDefects!(sim, g::AbstractIsingGraph, spin_idxs::Vector{T}) where T <: Integer
    
    # Only keep ones that are actually defect
    # @inbounds d_idxs = spin_idxs[defectBools(g)[spin_idxs]]
    d_idxs = spin_idxs[defectBools(g)[spin_idxs]]
    if isempty(d_idxs)
        return
    end

    # Remove defects from defect list and add to aliveList
    # Assumes that els are in list!
    if length(spin_idxs) > 1
        try
            newaliveList = zipOrderedLists(aliveList(g), d_idxs)
            newdefectList = remOrdEls(defectList(g), d_idxs)
            aliveList(g, newaliveList)
            defectList(g, newdefectList)
        catch(err)
            println("Aborting removing defects")
            # error(err)
            return
        end
            

        
    else    #Is faster for singular elements
        # Add to alive list
        # Adds it to original index offset by how many smaller numbers are also removed
        rem_idx = removeFirst!(defectList(g), spin_idxs[1]) 
        insert!(aliveList(g),spin_idxs[1]-(rem_idx-1),spin_idxs[1])
    end

    # Spins not defect anymore
    # @inbounds defectBools(g)[d_idxs] .= false
    defectBools(g)[d_idxs] .= false

    if isempty(defectList(g))  && getHParam(g, :Defects) == true
        editHType!(g, :Defects => false)
    end

    branchSim(sim)
end

remDefect!(sim, g, spin_idx::T) where T <: Integer = remDefects!(sim, g,[spin_idx]) 
addDefect!(sim, g, spin_idx::T) where T <: Integer = addDefects!(sim, g,[spin_idx]) 

# Lattice indexing
addDefect!(sim, g,i,j) = addDefect!(sim, g, coordToIdx(i,j,glength(g)))
remDefect!(sim, g,i,j) = remDefect!(sim, g, coordToIdx(i,j,glength(g)))

# Removes Multiple Defects
# Not used
function remDefects!(sim, g, idxs::Vector{Any})
    for idx in idxs
        remDefect!(sim, g, idx)
    end
end

# Removes All defects
function restoreState!(sim, g)
    (@view state(g)[defectList(g)]) .= rand(length(defectlist(g)))
    remDefects!(sim, g, defectList(g))
    editHType!(g, :Defects => false)
    branchSim(sim)
end
    
# Setting Elements

# Backend 

# Setting an alive element
function setNormal!(sim, g::AbstractIsingGraph, spin_idxs::Vector{Int32} , brush)
    # First remove defect if it was defect
    remDefects!(sim, g, spin_idxs)
    # Then set element
    # @inbounds state(g)[spin_idxs] .= brush
    state(g)[spin_idxs] .= brush
end

setNormal!(sim, g,spin_idx::Integer,brush) = setNormal!(sim, g, [spin_idx], brush)
setNormal!(sim, g,i::Integer,j::Integer,brush) =  setNormal!(sim, g,Int32.(coordToIdx(i,j,glength(g))),brush)
setNormal!(sim, g,tupls::Vector{Tuple{Int16,Int16}},brush) = setNormal!(sim, g,Int32.(coordToIdx.(tupls,glength(g))),brush)

function setClamp!(sim, g::AbstractIsingGraph, spin_idxs::Vector{Int32} , brush)
    addDefects!(sim, g, spin_idxs)
    @inbounds state(g)[spin_idxs] .= brush

    editHType!(g, :Defects => true)
    branchSim(sim)
end

setClamp!(sim, g,spin_idx::Integer,brush) = setClamp!(sim, g, [spin_idx], brush)
setClamp!(sim, g,i::Integer,j::Integer,brush) =  setClamp!(sim, g,Int32.(coordToIdx(i,j,glength(g))),brush)
setClamp!(sim, g,tupls::Vector{Tuple{Int16,Int16}},brush) = setClamp!(sim, g, Int32.(coordToIdx.(tupls,glength(g))),brush)

# User Functions 

# Set spins either to a value or clamp them
function setSpins!(sim, g::AbstractIsingGraph{Int8}, idxs , brush, clamp = false)
    if brush != 0 && !clamp
        setNormal!(sim, g,idxs,brush)
    else
        setClamp!(sim, g,idxs,brush)
    end
end

function setSpins!(sim, g::AbstractIsingGraph{Float32}, idxs , brush, clamp = false)
    if !clamp
        setNormal!(sim, g, idxs,brush)
    else
        setClamp!(sim, g, idxs,brush)
    end
end

setSpin!(sim, g::IsingGraph, i::Integer, j::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(sim, g, [coordToIdx(i,j,glength(g))], brush, clamp)

setSpin!(sim, g::IsingGraph, idx::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(sim, g,[idx],brush,clamp)

setSpins!(sim, g::IsingGraph, tupls::Vector{Tuple{Int32,Int32}}, brush, clamp) = setSpins!(sim, g, coordToIdx.(tupls,glength(g)), brush, clamp)


# end
