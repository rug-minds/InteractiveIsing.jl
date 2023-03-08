# Ising Graph Representation and functions
export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end

mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph{T}
    # Global graph props to be tracked for performance
    size::Tuple{Int32,Int32}

    nStates::Int32

    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    htype::HType
    d::GraphData

    IsingGraph(type::DataType; length, width, state, adj, weighted = false) = 
    (   
        h = new{type}(
            (length*width,1),
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

@inline glength(g::IsingGraph) = size(g)[1]
@inline gwidth(g::IsingGraph) = size(g)[2]

@inline graph(g::IsingGraph) = g


# Minimal Initialization using N and optional args
IsingGraph(length, width; continuous = true, weighted = false, weightFunc = defaultIsingWF, selfE = true) =
        let type = continuous ? Float32 : Int32
        IsingGraph(
            type;
            length,
            width, 
            state = initRandomState(continuous ? Float32 : Int32, length, width), 
            adj = initSqAdj(length, width, weightFunc = weightFunc),
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
initRandomState(type, glength, gwidth)::Vector{type} = initRandomState(type, glength*gwidth)

function initRandomState(type, nstates)::Vector{type}
    if type == Int32
        return rand([-1,1], nstates)
    elseif type == Float32
        return 2 .* rand(Float32, nstates) .- .5
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
        return Expr(:block, :(return it::UnitRange{Int32} = 1:nStates(g)) )
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
Is a pointer to function in SquareAdj.jl for compatibility
"""
initSqAdj(len, wid; weightFunc = defaultIsingWF) = createSqAdj(len, wid, weightFunc)

# """
# Initialization of adjacency Vector for a given N
# and using a weightfunc with a self energy
# """
# function initSqAdjSelf(len, wid; selfweight = -1 .* ones(len*wid), weightFunc = defaultIsingWF)
#     return initSqAdj(len, wid; weightFunc, self = true, selfweight)
# end

export continuous
continuous(g::IsingGraph{T}) where T = T <: Integer ? false : true

"""
resize Graph to new size with random states and no connections for the new states
"""
function resizeG!(g::IsingGraph{T}, nstates) where T
    oldlength = nStates(g)
    newlength = nStates(g) + nstates

    randomstate = initRandomState(T, nstates)
    state(g, expand(state(g), newlength, randomstate ))
    adj(g, expand(adj(g), newlength, [[] for _ in 1:nstates]))

    nStates(g, newlength)
    size(g, (newlength,1))

    # Data
    aLLength = length(aliveList(g))
    newaLLength = aLLength + nstates
    aliveList(g, expand(aliveList(g), newaLLength, [(oldlength+1):newlength;]))
    defectBools(g, expand(defectBools(g), newlength, false ))
    mlist(g, expand(mlist(g), newlength, 0) ) 
    clamps(g, expand(clamps(g), newlength, 0) )

    return
end

export resizeG!