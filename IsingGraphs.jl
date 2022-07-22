# Ising Graph Representation and functions
__precompile__()

module IsingGraphs
push!(LOAD_PATH, pwd())
   
# include("SquareAdj.jl")
using Random, Distributions, Observables, SquareAdj

# include("WeightFuncs.jl")
using WeightFuncs

export AbstractIsingGraph, IsingGraph, CIsingGraph, reInitGraph!, coordToIdx, idxToCoord, ising_it, setSpins!, setSpin!, addDefects!, remDefects!, addDefect!, remDefect!, 
    connIdx, connW, initSqAdj, HFunc, HWeightedFunc, HMagFunc, HWMagFunc, setMIdxs!

# Aliases
const Edge = Pair{Int32,Int32}
const Vert = Int32
const Weight = Float32
const Conn = Tuple{Vert, Weight}

abstract type AbstractIsingGraph end

mutable struct IsingData
    # is weighted
    weighted::Bool

    # For tracking defects
    aliveList::Vector{Vert}
    defects::Bool
    defectBools::Vector{Bool}
    defectList::Vector{Vert}
    
    # Magnetic field
    mactive::Bool
    mlist::Vector{Float32}

    hFuncRef::Ref

end

# IsingData(g, weighted = false) = IsingData( weighted, Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}() , false, zeros(Float32, g.size), weighted ? HFunc : HWeightedFunc)
IsingData(g, weighted = false; hFuncRef::Ref = Ref(HFunc)) = IsingData( weighted, Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}() , false, zeros(Float32, g.size), hFuncRef)

""" Discrete """

mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph
    # Global graph props to be tracked for performance
    N::Int32
    size::Int32
    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    d::IsingData

    IsingGraph(type::DataType, N::Integer; state, adj, weighted = false, weightFunc = defaultIsingWF) = 
        (   h = new{type}(
                N,
                N^2,
                state,
                adj
                );
            h.d = IsingData(h,weighted);
            return h
        )
end

IsingGraph(N::Integer; weighted = false, weightFunc = defaultIsingWF) = IsingGraph(Int8, N, state = initRandomState(N^2), adj = initSqAdj(N, weightFunc = weightFunc); weighted)
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

function reInitGraph!(g::IsingGraph, state = typeof(g) == IsingGraph{Int8} ? initRandomState(g.size) : initRandomCState(g.size))
    println("Reinitializing graph")
    g.state = state
    g.d.defects = false
    g.d.aliveList = [1:g.size;]
    g.d.defectBools = [false for x in 1:g.size]
    g.d.defectList = []
    g.d.mactive = false
    g.d.mlist = zeros(g.size)
    setGHFunc!(g)
end

# Initialization of state
function initRandomState(size)::Vector{Int8}
    return rand([-1,1],size)
end

""" Continuous """

# mutable struct CIsingGraph <: AbstractIsingGraph
#     # Global graph props to be tracked for performance
#     N::Int32
#     size::Int32
#     # Vertices and edges
#     state::Vector{Float32}
#     adj::Vector{Vector{Conn}}
#     # Energy function
#     d::IsingData
    
#     CIsingGraph(N::Integer; state = initRandomCState(N^2), weighted = false, weightFunc = DefaultIsingWF, selfE = true) = 
#         (   h = new(
#                 N,
#                 N^2,
#                 state,
#                 selfE ? initSqAdjSelf(N, weightFunc = weightFunc) : initSqAdj(N, weightFunc = weightFunc)
#             );
#             h.d = IsingData(h,weighted);
#             return h
#         )

# end

# # Copy graph data to new one
# CIsingGraph(g::CIsingGraph) = deepcopy(g)

# reInitGraph!(g::CIsingGraph) = reInitGraph!(g, initRandomCState(g.size))


# Initialization of state
function initRandomCState(size)::Vector{Float32}
    return 2 .* (rand(Float32,size) .- .5)
end

"""
Methods
"""
    # Matrix Coordinates to vector Coordinates
    @inline  function coordToIdx(i,j,N)
        return (i-1)*N+j
    end

    coordToIdx((i,j),N) = coordToIdx(i,j,N)
    # Go from idx to lattice coordinates
    @inline function idxToCoord(idx::Integer,N::Integer)
        return ((idx-1)÷N+1,(idx-1)%N+1)
    end

 

    # Returns an iterator over the ising lattice
    # If there are no defects, returns whole range
    # Otherwise it returns all alive spins
    function ising_it(g::AbstractIsingGraph)
        if !g.d.defects
            it::UnitRange{Int32} = 1:g.size
            return it
        else
            return g.d.aliveList
        end

    end

    # Get weight of connection
    @inline function connIdx(conn::Conn)
        conn[1]
    end

    # Get weight of connection
    @inline function connW(conn::Conn)
        conn[2]
    end

# Initialization of adjacency matrix for a given ND
function initSqAdj(N; weightFunc = defaultIsingWF)
    adj = Vector{Vector{Conn}}(undef,N^2)
    fillAdjList!(adj,N, weightFunc)
    return adj
end

function initSqAdjSelf(N; selfWeights = -1 .* ones(N^2), weightFunc = defaultIsingWF)
    adj = Vector{Vector{Conn}}(undef,N^2)
    fillAdjList!(adj,N, weightFunc)
    for (idx,el) in enumerate(adj)
        append!(el,[(idx,selfWeights[idx])])
    end
    return adj
end


""" Setting Elements and defects """



"""Removing and adding defects, and clamping"""
    
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
     
"""Setting Elements"""

    """Backend """
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

    """ User Functions """
        # Set points either to element or defect
        # Implement clamping
        function setSpins!(g::IsingGraph{Int8}, idxs , brush, clamp = false)
            if brush != 0 && !clamp
                setNormal!(g,idxs,brush)
            else
                setClamp!(g,idxs,brush)
            end
        end

        function setSpins!(g::IsingGraph{Float32}, idxs , brush, clamp = false)
            if !clamp
                println("Here")
                setNormal!(g,idxs,brush)
            else
                println("Or Here")
                setClamp!(g,idxs,brush)
            end
        end

        setSpin!(g::AbstractIsingGraph,i::Integer,j::Integer, brush::Real, clamp::Bool = false) = setSpins!(g, [coordToIdx(i,j,g.N)], brush, clamp)
        
        setSpin!(g::AbstractIsingGraph, coord::Integer, brush::Real, clamp::Bool = false) = setSpins!(g,[coord],brush,clamp)

        setSpins!(g::AbstractIsingGraph, tupls::Vector{Tuple{Int32,Int32}}, brush, clamp) = setSpins!(g, coordToIdx.(tupls,g.N), brush, clamp)
        


""" Hamiltonians"""

# No weights
function HFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor
end

# No weights but magfield
function HMagFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor -g.d.mlist[idx]
end

# When there's weights
function HWeightedFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor
end

# Weights and magfield
function HWMagFunc(g::AbstractIsingGraph,idx,state = g.state[idx])::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor -g.d.mlist[idx]
end

function setGHFunc!(g, shouldRun::Observable,isRunning::Ref)
    if !g.d.weighted
        if !g.d.mactive
            g.d.hFuncRef = Ref(HFunc)
            println("Set HFunc")
        else
            g.d.hFuncRef = Ref(HMagFunc)
            println("Set HMagFunc")
        end
    else
        if !g.d.mactive
            g.d.hFuncRef = Ref(HWeightedFunc)
            println("Set HWeightedFunc")
        else
            g.d.hFuncRef = Ref(HWMagFunc)
            println("Set HWMagFunc")
        end
    end

    branchSim(shouldRun,isRunning)
end

""" Changing E functions """

function setMIdxs!(g,idxs,strengths,shouldRun::Observable,isRunning::Ref)
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    shouldRun[] = false
    g.d.mactive = true
    g.d.mlist[idxs] = strengths
    # setGHFunc!(g)
    g.d.hFuncRef = Ref(HWMagFunc)
    while isRunning[]
        sleep(.1)
    end
    shouldRun[] = true
end

function branchSim(shouldRun::Observable,isRunning::Ref)
    shouldRun[] = false 
    while isRunning[]
        sleep(.1)
    end
    shouldRun[] = true
end


"""Helper Functions"""


    function sortedPair(idx1::Integer,idx2::Integer):: Pair{Integer,Integer}
        if idx1 < idx2
            return idx1 => idx2
        else
            return idx2 => idx1
        end
    end

    # Searches backwards from idx in list and removes item
    # This is because spin idx can only be before it's own index in aliveList
    function revRemoveSpin!(list,spin_idx)
        init = min(spin_idx, length(list)) #Initial search index
        for offset in 0:(init-1)
            @inbounds if list[init-offset] == spin_idx
                deleteat!(list,init-offset)
                return init-offset # Returns index where element was found
            end
        end
    end

    # Zip together two ordered lists into a new ordered list    
    function zipOrderedLists(vec1::Vector{T},vec2::Vector{T}) where T
        # result::Vector{T} = zeros(length(vec1)+length(vec2))
        result = Vector{T}(undef, length(vec1)+length(vec2))

        ofs1 = 1
        ofs2 = 1
        while ofs1 <= length(vec1) && ofs2 <= length(vec2)
            @inbounds el1 = vec1[ofs1]
            @inbounds el2 = vec2[ofs2]
            if el1 < el2
                @inbounds result[ofs1+ofs2-1] = el1
                ofs1 += 1
            else
                @inbounds result[ofs1+ofs2-1] = el2
                ofs2 += 1
            end
        end

        if ofs1 <= length(vec1)
            @inbounds result[ofs1+ofs2-1:end] = vec1[ofs1:end]
        else
            @inbounds result[ofs1+ofs2-1:end] = vec2[ofs2:end]
        end
        return result
    end

    # Deletes els from vec
    # Assumes that els are in vec!
    function remOrdEls(vec::Vector{T}, els::Vector{T}) where T
        # result::Vector{T} = zeros(length(vec)-length(els))
        result = Vector{T}(undef, length(vec)-length(els))
        it_idx = 1
        num_del = 0
        for el in els
                while el != vec[it_idx]
            
                result[it_idx - num_del] = vec[it_idx]
                it_idx +=1
            end
                num_del +=1
                it_idx += 1
        end
            result[(it_idx - num_del):end] = vec[it_idx:end]
        return result
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


end


""" Old


macro varname(arg)
    string(arg)
end

function getHMField!(g,idxs,strengths)
    gname = @varname(g)
    try
        setMIdxs!(g,idxs,strengths)
    catch
        return
    end
    

    if g.d.weighted
        println("Hamiltonian set to weighted + magnetic field")
        eval(Meta.parse("DOLLARSIGN(gname).getE(g,idx,state = g.state[idx]) = HWeightedFunc(g,idx,state) + HMagFunc(g,idx,state)"))
    else
        println("Hamiltonian set to unweighted + magnetic field")
        eval(Meta.parse("DOLLARSIGN(gname).getE(g,idx,state = g.state[idx]) = HFunc(g,idx,state) + HMagFunc(g,idx,state)"))
    end

    
end

function setMIdxs!(g,idxs,strengths)
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    for idx in idxs
        g.d.mlist[idx] = strengths[idx]
    end
end


"""