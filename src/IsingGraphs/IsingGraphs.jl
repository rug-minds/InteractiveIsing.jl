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
    const defectBools::Vector{Bool}
    defectList::Vector{Vert}
    
    # Magnetic field
    const mlist::Vector{Float32}

    #Beta clamp
    clampparam::Float32
    #clamps
    const clamps::Vector{Float32}
end

# IsingData(g, weighted = false) = IsingData( weighted, Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}() , false, zeros(Float32, g.size), weighted ? HFunc : HWeightedFunc)
IsingData(g) = IsingData(Int32.([1:g.size;]), false, [false for x in 1:g.size], Vector{Int32}(), zeros(Float32, g.size), 0 , zeros(Float32, g.size))

function reinitIsingData!(data, g)
    data.aliveList = Int32.([1:g.size;])
    data.defects = false
    data.defectBools .= false
    data.defectList .= Vector{Int32}()
    data.mlist .= 0
    data.clamps .= 0
end

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

@setterGetter IsingGraph
@forward IsingGraph IsingData d

# Minimal Initialization using N and optional args
IsingGraph(N::Integer; continuous = true, weighted = false, weightFunc = defaultIsingWF, selfE = true) =
    let adjfunc = continuous ? (selfE ? initSqAdjSelf : initSqAdj) : initSqAdj,
        type = continuous ? Float32 : Int8
        IsingGraph(
            type, 
            N, 
            state = initRandomState(continuous ? Float32 : Int8, N^2), 
            adj = adjfunc(N, weightFunc = weightFunc)
            ;weighted
        )
    end

# Copy graph data to new one
IsingGraph(g::IsingGraph) = deepcopy(g)

function reinitIsingGraph!(g::IsingGraph)
    g.state .= initRandomState(g)
    g.htype = HType(:Weighted => getHParam(g.htype, :Weighted))
    reinitIsingData!(g.d,g)
end


export initRandomState
""" 
Initialize a random discrete state 
"""
function initRandomState(g::IsingGraph{Int8})::Vector{Int8}
    return rand([-1,1],size(g))
end
""" 
Initialize a random continuous state 
"""
function initRandomState(g::IsingGraph{Float32})::Vector{Float32}
    return 2 .* (rand(Float32,size(g)) .- .5)
end

function initRandomState(type, size)
    if type == Int8
        return rand([-1,1],size)
    elseif type == Float32
        return 2 .* (rand(Float32,size) .- .5)
    end
end
#=
Methods
=#

""" 
Returns an iterator over the ising lattice
If there are no defects, returns whole range
Otherwise it returns all alive spins
"""

@generated function ising_it(g::IsingGraph, htype::HType{Symbs,Params}) where {Symbs,Params}
    # Assumes :Defects will be found
    defects = getHParamType(htype, :Defects)

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
function initSqAdj(N; weightFunc = defaultIsingWF, self = false, selfWeights = -1 .* ones(N^2))
    adj = Vector{Vector{Conn}}(undef,N^2)
    fillAdjList!(adj,N, weightFunc)
    if self
        for (idx,el) in enumerate(adj)
            append!(el,[(idx,selfWeights[idx])])
        end
    end

    return adj
end

"""
Initialization of adjacency matrix for a given N
and using a weightfunc with a self energy
"""
function initSqAdjSelf(N; selfWeights = -1 .* ones(N^2), weightFunc = defaultIsingWF)
    return initSqAdj(N; weightFunc, self = true, selfWeights)
end

export continuous
continuous(g::IsingGraph{T}) where T = T <: Integer ? false : true