# Ising Graph Representation and functions
export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end

mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph{T}
    # Global graph props to be tracked for performance
    size::Tuple{Int32,Int32}

    Nstates::Int32

    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    htype::HType
    d::GraphData

    IsingGraph(type::DataType; length, width, state, adj, weighted = false) = 
    (   
        h = new{type}(
            (length,width),
            length*width,
            state,
            adj,
            generateHType(weighted,false)
        );
        
        h.d = GraphData(h);
        return h
    )

    
end

@setterGetter IsingGraph
@forward IsingGraph GraphData d

import Base: length
glength(g::IsingGraph) = size(g)[1]
gwidth(g::IsingGraph) = size(g)[2]


# Minimal Initialization using N and optional args
IsingGraph(length, width; continuous = true, weighted = false, weightFunc = defaultIsingWF, selfE = true) =
    let adjfunc = continuous ? (selfE ? initSqAdjSelf : initSqAdj) : initSqAdj,
        type = continuous ? Float32 : Int8
        IsingGraph(
            type;
            length,
            width, 
            state = initRandomState(continuous ? Float32 : Int8, length, width), 
            adj = adjfunc(length, width, weightFunc = weightFunc),
            weighted
        )
    end

IsingGraph(N; continuous = true, weighted = false, weightFunc = defaultIsingWF, selfE = true) = 
    IsingGraph(N, N; continuous, weighted, weightFunc, selfE)

# Copy graph data to new one
IsingGraph(g::IsingGraph) = deepcopy(g)

function reinitIsingGraph!(g::IsingGraph)
    state(g) .= initRandomState(g)
    currentlyWeighted = getHParam(g.htype, :Weighted)
    g.htype = HType(:Weighted => currentlyWeighted)
    reinitGraphData!(g.d,g)
end


export initRandomState
""" 
Initialize from a graph
"""
(initRandomState(g::IsingGraph{type})::Vector{type}) where type = initRandomState(type, glength(g), gwidth(g))
    

function initRandomState(type, length, width)::Vector{type}
    if type == Int8
        return rand([-1,1], length*width)
    elseif type == Float32
        return 2 .* rand(Float32, length*width) .- .5
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
        return Expr(:block, :(return it::UnitRange{Int32} = 1:Nstates(g)) )
    else
        return Expr(:block, :(return aliveList(g)))
    end

end

"""
Get index of connection
"""
@inline function connIdx(conn::Conn)::Vert
    conn[1]
end

"""
Get weight of connection
"""
@inline function connW(conn::Conn)::Weight
    conn[2]
end

"""
Initialization of adjacency Vector for a given N
and using a weightfunc
"""
function initSqAdj(length, width; weightFunc = defaultIsingWF, self = false, selfWeights = -1 .* ones(length*width))::Vector{Vector{Conn}}
    adj = Vector{Vector{Conn}}(undef,length*width)
    fillAdjList!(adj, length, width, weightFunc)
    if self
        for (idx,el) in enumerate(adj)
            append!(el,[(idx,selfWeights[idx])])
        end
    end
    return adj
end

"""
Initialization of adjacency Vector for a given N
and using a weightfunc with a self energy
"""
function initSqAdjSelf(length, width; selfWeights = -1 .* ones(length*width), weightFunc = defaultIsingWF)
    return initSqAdj(length, width; weightFunc, self = true, selfWeights)
end

export continuous
continuous(g::IsingGraph{T}) where T = T <: Integer ? false : true