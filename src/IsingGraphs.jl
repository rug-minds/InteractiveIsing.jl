# Ising Graph Representation and functions

export AbstractIsingGraph, IsingGraph, CIsingGraph, reInitGraph!, coordToIdx, idxToCoord, ising_it, setSpins!, setSpin!, addDefects!, remDefects!, addDefect!, remDefect!, 
    connIdx, connW, initSqAdj, HFunc, HWeightedFunc, HMagFunc, HWMagFunc, setGHFunc!

# Aliases
const Edge = Pair{Int32,Int32}
const Vert = Int32
const Weight = Float32
const Conn = Tuple{Vert, Weight}

export AbstractIsingGraph
abstract type AbstractIsingGraph end

mutable struct IsingData
    # For tracking defects
    aliveList::Vector{Vert}
    defects::Bool
    defectBools::Vector{Bool}
    defectList::Vector{Vert}
    
    # Magnetic field
    mlist::Vector{Float32}
end

# IsingData(g, weighted = false) = IsingData( weighted, Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}() , false, zeros(Float32, g.size), weighted ? HFunc : HWeightedFunc)
IsingData(g) = IsingData(Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}(), zeros(Float32, g.size))

mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph
    # Global graph props to be tracked for performance
    N::Int32
    size::Int32
    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    htype::HType
    d::IsingData


    IsingGraph(type::DataType, N::Integer; state, adj, weighted = false) = 
        (   h = new{type}(
                N,
                N^2,
                state,
                adj,
                generateHType(weighted,false)
                );
            h.d = IsingData(h);
            return h
        )
end


# Minimal Initialization using N and optional args
IsingGraph(N::Integer; weighted = false, weightFunc = defaultIsingWF) = IsingGraph(Int8, N, state = initRandomState(N^2), adj = initSqAdj(N, weightFunc = weightFunc); weighted)
# Initialize continuous ising graph
CIsingGraph(N::Integer; weighted = false, weightFunc = defaultIsingWF, selfE = true) =  
    IsingGraph(
        Float32,
        N, 
        state = initRandomCState(N^2), 
        adj = selfE ? initSqAdjSelf(N, weightFunc = weightFunc) : initSqAdj(N, weightFunc = weightFunc);
        weighted
    )

# Copy graph data to new one
IsingGraph(g::IsingGraph) = deepcopy(g)

# Initialization of state
export initRandomIsing!
function initRandomIsing!(g)
    if typeof(g) == IsingGraph{Int8}
        g.state = initRandomState(g.size)
    else
        g.state = initRandomCState(g.size)
    end
end

export initRandomState
""" 
Initialize a random discrete state 
"""
function initRandomState(size)::Vector{Int8}
    return rand([-1,1],size)
end

export initRandomCState
""" 
Initialize a random continuous state 
"""
function initRandomCState(size)::Vector{Float32}
    return 2 .* (rand(Float32,size) .- .5)
end

#=
Methods
=#

""" 
Returns an iterator over the ising lattice
If there are no defects, returns whole range
Otherwise it returns all alive spins
"""
# function ising_it(g::AbstractIsingGraph)
#     # if !g.d.defects
#         it::UnitRange{Int32} = 1:g.size
#         return it
#     # else
#     #     return g.d.aliveList
#     # end

# end

@generated function ising_it(g::IsingGraph, htype::HType{Symbs,Params}) where {Symbs,Params}
    # Assumes :Defects will be found
    defectIdx = 1
    for symb in Symbs
        if symb == :Defects
            defectIdx += 1
            break
        end
    end

    defects = Params[defectIdx]

    if !defects
        return Expr(:block, :(return it::UnitRange{Int32} = 1:g.size) )
    else
        return Expr(:block, :(return g.d.aliveList))
    end

end

"""
Get index of connection
"""
@inline function connIdx(conn::Conn)
    conn[1]
end

"""
Get weight of connection
"""
@inline function connW(conn::Conn)
    conn[2]
end

"""
Initialization of adjacency matrix for a given N
and using a weightfunc
"""
function initSqAdj(N; weightFunc = defaultIsingWF)
    adj = Vector{Vector{Conn}}(undef,N^2)
    fillAdjList!(adj,N, weightFunc)
    return adj
end

"""
Initialization of adjacency matrix for a given N
and using a weightfunc with a self energy
"""
function initSqAdjSelf(N; selfWeights = -1 .* ones(N^2), weightFunc = defaultIsingWF)
    adj = Vector{Vector{Conn}}(undef,N^2)
    fillAdjList!(adj,N, weightFunc)
    for (idx,el) in enumerate(adj)
        append!(el,[(idx,selfWeights[idx])])
    end
    return adj
end


# Setting Elements and defects 

# Removing and adding defects, and clamping
    
    # Adding a defect to lattice
    function addDefects!(g::AbstractIsingGraph,spin_idxs::Vector{T}) where T <: Integer
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
        if g.d.defects == false
            g.d.defects = true
        end

        # # Set states to zero
        # @inbounds g.state[d_idxs] .= 0
    end

    # Removing defects, insert ordered list!
    function remDefects!(g::AbstractIsingGraph, spin_idxs::Vector{T}) where T <: Integer
        
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
    
        if isempty(g.d.defectList) && g.d.defects == true
            g.d.defects = false
        end

    end

    remDefect!(g, spin_idx::T) where T <: Integer = remDefects!(g,[spin_idx]) 
    addDefect!(g, spin_idx::T) where T <: Integer = addDefects!(g,[spin_idx]) 
    
    # Lattice indexing
    addDefect!(g,i,j) = addDefect!(g,coordToIdx(i,j,g.N))
    remDefect!(g,i,j) = remDefect!(g,coordToIdx(i,j,g.N))

    # Removes Multiple Defects
    # Not used
    function remDefects!(g,idxs::Vector{Any})
        for idx in idxs
            remDefect!(g,idx)
        end
    end

    # Removes All defects
    function restoreState!(g)
        g.state[g.d.defectList] = rand(length(defectlist))
        remDefects!(g,g.d.defectList)
    end
     
# Setting Elements

    # Backend 
        # Setting an alive element
        function setNormal!(g::AbstractIsingGraph, spin_idxs::Vector{Int32} , brush)
            # First remove defect if it was defect
            remDefects!(g,spin_idxs)
            # Then set element
            @inbounds g.state[spin_idxs] .= brush
        end

        setNormal!(g,spin_idx::Integer,brush) = setNormal!(g, [spin_idx], brush)
        setNormal!(g,i::Integer,j::Integer,brush) =  setNormal!(g,Int32.(coordToIdx(i,j,g.N)),brush)
        setNormal!(g,tupls::Vector{Tuple{Int16,Int16}},brush) = setNormal!(g,Int32.(coordToIdx.(tupls,g.N)),brush)

        function setClamp!(g::AbstractIsingGraph, spin_idxs::Vector{Int32} , brush)
            addDefects!(g,spin_idxs)
            @inbounds g.state[spin_idxs] .= brush
        end

        setClamp!(g,spin_idx::Integer,brush) = setClamp!(g, [spin_idx], brush)
        setClamp!(g,i::Integer,j::Integer,brush) =  setClamp!(g,Int32.(coordToIdx(i,j,g.N)),brush)
        setClamp!(g,tupls::Vector{Tuple{Int16,Int16}},brush) = setClamp!(g,Int32.(coordToIdx.(tupls,g.N)),brush)

    # User Functions 
        # Set spins either to a value or clamp them
        function setSpins!(g::IsingGraph{Int8}, idxs , brush, clamp = false)
            if brush != 0 && !clamp
                setNormal!(g,idxs,brush)
            else
                setClamp!(g,idxs,brush)
            end
        end

        function setSpins!(g::IsingGraph{Float32}, idxs , brush, clamp = false)
            if !clamp
                setNormal!(g,idxs,brush)
            else
                setClamp!(g,idxs,brush)
            end
        end

        setSpin!(g::AbstractIsingGraph, i::Integer, j::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(g, [coordToIdx(i,j,g.N)], brush, clamp)
        
        setSpin!(g::AbstractIsingGraph, idx::Integer, brush::Union{Int8,Float32}, clamp::Bool = false) = setSpins!(g,[idx],brush,clamp)

        setSpins!(g::AbstractIsingGraph, tupls::Vector{Tuple{Int32,Int32}}, brush, clamp) = setSpins!(g, coordToIdx.(tupls,g.N), brush, clamp)
        


