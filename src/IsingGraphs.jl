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
    (   
        h = new{type}(
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
IsingGraph(N::Integer; continuous = true, weighted = false, weightFunc = defaultIsingWF, selfE = true) =
    let adjfunc = continuous ? (selfE ? initSqAdjSelf : initSqAdj) : initSqAdj,
        type = continuous ? Float32 : Int8
        IsingGraph(
            type, 
            N, 
            state = initRandomState(N^2), 
            adj = adjfunc(N, weightFunc = weightFunc)
            ;weighted
        )
    end

# Initialize continuous ising graph
# CIsingGraph(N::Integer; weighted = false, weightFunc = defaultIsingWF, selfE = true) =  
#     IsingGraph(
#         Float32,
#         N, 
#         state = initRandomCState(N^2), 
#         adj = selfE ? initSqAdjSelf(N, weightFunc = weightFunc) : initSqAdj(N, weightFunc = weightFunc)
#         ;weighted
#     )

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


