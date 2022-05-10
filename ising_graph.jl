# Ising Graph Representation and functions

mutable struct IsingGraph
    N::Int32
    size::Int32
    state::Vector{Int8}
    adj::Dict{Int32,Vector{Int32}}
    # For tracking defects
    aliveList::Vector{Int32}
    defects::Bool
    defectBools::Vector{Bool}
    defectList::Vector{Int32}
end

"""
INITIALIZERS
"""
# Initialize without defects
IsingGraph(a,b,c,d) = IsingGraph(a,b,c,d,[1:b;], false ,[false for x in 1:b],[])

#Initialization using only N
IsingGraph(N::Int) = IsingGraph(N,N*N,initRandomState(N),initAdj(N))

#Initialization of graph using a state and adjacency matrix
IsingGraph(state::Vector{Int8},adj::Dict{Int32,Vector{Int32}}) = let size = length(state)
    IsingGraph(sqrt(size), size, copy(state), adj)
end

function reInitGraph!(g::IsingGraph)
    println("Reinitializing graph")
    g.state = initRandomState(g.N)
    g.defects = false
    g.aliveList = [1:g.size;]
    g.defectBools = [false for x in 1:g.size]
    g.defectList = []
end

"""
Methods
"""
# Matrix Coordinates to vector Coordinates
function coordToIdx(i,j,N)
    return (i-1)*N+j
end
                
function graphLookup(idx,N)
    return ((idx-1)÷N+1,(idx-1)%N+1)
end

# Initialization of state
function initRandomState(N)::Vector{Int8}
    return [sample([-1,1]) for x in 1:(N*N) ]
end

# Initialization of adjacency matrix for a given ND
function initAdj(N)
    adj = Dict{Int32,Vector{Int32}}()
    fillAdjList!(adj,N)
    return adj
end


# Returns an iterator over the ising lattice
# If there are no defects, returns whole range
# Otherwise it returns all alive spins
function ising_it(g::IsingGraph)
    if !g.defects
        it::UnitRange{Int32} = 1:g.size
        return it
    else
        return g.aliveList
    end

end

""" Setting Elements and defects """

""" Adding Defects """
function addDefect!(g,spin_idx)
    # If already defect do nothing
    if g.defectBools[spin_idx] == true
        return
    end

    #Remove item from alive list and start searching backwards from spin_idx
    # Since aliveList is sorted, spin_idx - idx_found gives number of smaller elements in list
    rem_idx = revRemoveSpin!(g.aliveList,spin_idx)

    insert!(g.defectList,spin_idx-rem_idx+1,spin_idx)

    # If first defect, mark lattice as containing defects
    if g.defects == false
        g.defects = true
    end

    # Mark corresponding spin to defect
    g.defectBools[spin_idx] = true


    # Set state to zero
    g.state[spin_idx] = 0

end

# Removing defects
function remDefect!(g, spin_idx)
    # If it was not defect, skip
    if g.defectBools[spin_idx] == false
        # println("No defect to remove")
        return
    end

    # Spin not defect anymore
    g.defectBools[spin_idx] = false

    # Remove from defect list
    rem_idx = removeFirst!(g.defectList,spin_idx)
    
    # Add to alive list
    # Adds it to original index offset by how many smaller numbers are also removed
    insert!(g.aliveList,spin_idx-(rem_idx-1),spin_idx)

    if isempty(g.defectList) && g.defects == true
        g.defects = false
    end
end

# Lattice indexing
addDefect!(g,i,j) = addDefect!(g,coordToIdx(i,j,g.N))
remDefect!(g,i,j) = remDefect!(g,coordToIdx(i,j,g.N))

# Vector Version
# Not used
function addDefects!(g,idxs::Vector{Any})
    for idx in idxs
        addDefect!(g,idx)
    end
end

# Add percantage of defects randomly to lattice
function addRandomDefects!(g,p)
    if isempty(g.aliveList) || p[] == 0
        return nothing
    end

    for def in 1:round(length(g.aliveList)*p[]/100)
        idx = rand(g.aliveList)
        addDefect!(g,idx)
    end
    p[] = 0         # Reset observable of percantage of elements to be poked
end

"""Setting Elements"""
# Set element to -1 or +1, shouldn't be 0
function setEl!(g,spin_idx, brush)
    # First remove defect if it was defect
    remDefect!(g,spin_idx)
    # Then set element
    g.state[spin_idx] = brush

end

setEl!(g,i,j,brush) =  setEl!(g,coordToIdx(i,j,g.N),brush)

function remDefects!(g,idxs::Vector{Any})
    for idx in idxs
        remDefect!(g,idx)
    end
end

# Removes All defects
function restoreState!(g)
    remDefects!(g,g.defectList)
end

# Remove first element equal to el and returns correpsonding index
function removeFirst!(list,el)
    for (idx,item) in enumerate(list)
        if item == el
            deleteat!(list,idx)
            return idx
        end
    end
end

# Searches backwards from idx in list and removes item
# This is because spin idx can only be before it's own index in aliveList
function revRemoveSpin!(list,spin_idx)
    init = min(spin_idx, length(list)) #Initial search index
    for offset in 0:(init-1)
        if list[init-offset] == spin_idx
            deleteat!(list,init-offset)
            return init-offset # Returns index where element was found
        end
    end
end

