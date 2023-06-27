abstract type StateType end
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end


# Ising Graph Representation and functions
mutable struct IsingGraph{T <: Real, Sim <: IsingSim} <: AbstractIsingGraph{T}
    sim::Sim
    nStates::Int32

    # Vertices and edges
    state::Vector{T}
    const adj::Vector{Vector{Conn}}
    adjlist::AdjList{Connections}
    # Depcrecated
    htype::HType

    stype::SType
    
    layers::Vector{IsingLayer}
    layeridxs::Vector{Int32}

    continuous::StateType
    # Connection between layers, I don't think it's neccesary to track this
    layerconns::Dict{Set, Int32}
    defects::GraphDefects
    d::GraphData

    function IsingGraph(sim, length, width; periodic = nothing, weightfunc::Union{Nothing,WeightFunc} = nothing, continuous = false, weighted = false)
      
        type = continuous ? Float32 : Int8
        g = new{type, typeof(sim)}(
            sim,
            0,
            type[],
            # Adj
            Vector{Vector{Conn}}[],
            # AdjList
            AdjList(length*width),
            HType(weighted, false),
            SType(:Weighted => weighted),
            #Layers
            Vector{IsingLayer}[],
            Int32[],
            #ContinuityType
            continuous ? ContinuousState() : DiscreteState(),
            Dict{Pair, Int32}()
        )

        # TODO: Later remove this when we have a better way to do this

        # Default Weightfun 
        # if isnothing(weightfunc) || !weighted
        #     weightfunc = defaultIsingWF
        #     if continuous
        #         setSelf(weightfunc, (i,j) -> -1)
        #     end
        # end

        #For performance, don't know why
        # g.adj = deepcopy(g.adj)
        g.defects = GraphDefects(g)
        g.d = GraphData(g)

        addLayer!(g, length, width; periodic, weightfunc, type)

        return g
    end
    
end

#extend show to print out the graph, showing the length of the state, and the layers
function Base.show(io::IO, g::IsingGraph)
    println(io, "IsingGraph with $(nStates(g)) states")
    println(io, "Layers:")
    for (idx, layer) in enumerate(g.layers)
        Base.show(io, layer)
        if idx != length(g.layers)
            print(io, "\n")
        end
    end
end

Base.show(io::IO, graphtype::Type{IsingGraph}) = print(io, "IsingGraph")


coords(g::IsingGraph) = VSI(layers(g), :coords)
export coords

@setterGetter IsingGraph
# @inline htype(g::IsingGraph) = g.htype[]
# @inline htype(g::IsingGraph, htype) = g.htype[] = htype
export htype

# @inline simulation(g::IsingGraph) = g.sim
# export simulation

@forward IsingGraph GraphData d
@forward IsingGraph GraphDefects defects

@inline glength(g::IsingGraph)::Int32 = size(g)[1]
@inline gwidth(g::IsingGraph)::Int32 = size(g)[2]

@inline graph(g::IsingGraph) = g

# Access the layers
@inline layer(g::IsingGraph, idx) = g.layers[idx]
@inline Base.getindex(g::IsingGraph, idx) = layer(g, layeridxs(g)[idx])
#TODO: Give new idx
@inline function layerIdx!(g, oldidx, newidx)
    newidxs = Vector{Int32}(undef, length(g.layeridxs))
    l_idx = g.layeridxs[oldidx]

    newidxs[1:l_idx-1] .= g.layeridxs[1:l_idx-1]
    newidxs[l_idx] = newidx
    newidxs[l_idx+1:end] .= g.layeridxs[l_idx+1:end] .+ 1

    g.layeridxs = newidxs
end
export layerIdx!

IsingGraph(g::IsingGraph) = deepcopy(g)

@inline layerdefects(g::IsingGraph) = layerdefects(defects(g))
export layerdefects

@inline size(g::IsingGraph)::Tuple{Int32,Int32} = (nStates(g), 1)

function reset!(g::IsingGraph)
    state(g) .= initRandomState(g)
    currentlyWeighted = getSParam(stype(g), :Weighted)
    stype(g,SType(:Weighted => currentlyWeighted))
    reset!(defects(g))
    reset!(d(g))
end
 
export initRandomState
""" 
Initialize from a graph
"""
(initRandomState(g::IsingGraph{type})::Vector{type}) where type = initRandomState(type, glength(g), gwidth(g))
initRandomState(type, glength, gwidth)::Vector{type} = initRandomState(type, glength*gwidth)

function initRandomState(type, nstates)::Vector{type}
    if type == Int8
        return rand([-1,1], nstates)
    elseif type == Float32
        return 2.f0 .* rand(Float32, nstates) .- 1.f0
    else
        return rand(type[-1,1], nstates)
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

@generated function ising_it(g::IsingGraph, stype::SType = stype(g))
    # Assumes :Defects will be found
    defects = getSParam(stype, :Defects)

    if !defects
        return Expr(:block, :(return UnitRange{Int32}(1:nStates(g)) ))
        # return Expr(:block, :(return Base.OneTo(nStates(g)) ))
    else
        return Expr(:block, :(return aliveList(g)))
    end

end

export ising_it

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
and using a weightFunc
Is a pointer to function in SquareAdj.jl for compatibility
"""
initSqAdj(len, wid; weightfunc = defaultIsingWF) = createSqAdj(len, wid, weightfunc)

# """
# Initialization of adjacency Vector for a given N
# and using a weightFunc with a self energy
# """
# function initSqAdjSelf(len, wid; selfweight = -1 .* ones(len*wid), weightFunc = defaultIsingWF)
#     return initSqAdj(len, wid; weightFunc, self = true, selfweight)
# end

export continuous
continuous(g::IsingGraph{T}) where T = T <: Integer ? false : true

setdefect(g::IsingGraph, val, idx) = defects(g)[idx] = val

"""
Resize Graph to new size with random states and no connections for the new states
Need to be bigger than original size?
"""
function resize!(g::IsingGraph{T}, newlength, layeridx = 0) where T
    oldlength = nStates(g)
    sizediff = newlength - oldlength

    # If making smaller save states and copy them
    # if sizediff < 0
    #     savedstates = state(g)[(endidx(layer(g,layeridx))+1):end]
    #     
    # end

    

    if sizediff > 0
        resize!(state(g), newlength)
        resize!(adj(g), newlength)
        randomstate = initRandomState(T, sizediff)
        state(g)[oldlength+1:newlength] .= randomstate
        adj(g)[oldlength+1:newlength] .= [Vector{Conn}[] for _ in oldlength+1:newlength]
    else # if making smaller
        d_layer = layer(g,layeridx)
        # idxs to be removed
        g_idxs = graphidxs(d_layer) 

            
        # Delete everything from adj
        for idx in g_idxs
            Threads.@threads for conn in adj(g)[idx]
                conn_idx = connIdx(conn)
                removeWeightDirected!(adj(g), conn_idx, idx)
            end
        end

        #Shift all the leftovers from adj
        for idx in g_idxs[end]:length(adj(g))
            adj(g)[idx] = shiftWeight.(adj(g)[idx], -nStates(d_layer))
        end

        deleteat!(adj(g), g_idxs)
        deleteat!(state(g), g_idxs)
    end

    nStates(g, newlength)

    # Resize Data
    resize!(d(g), newlength)

    return
end

#Remove states from state(graph) and return new vector
function removeStates(state, startidx, endidx)
    #initialize new vec
    newstate = Vector{eltype(state)}(undef, length(state) - (endidx - startidx + 1))
    #copy old vec
    newstate[1:startidx-1] = state[1:startidx-1]
    newstate[startidx:end] = state[endidx+1:end]

    return newstate
end

export resize!

function addLayer!(g::IsingGraph, llength, lwidth; weightfunc = nothing, periodic = true, type = typeof(state(g)))
    glayers = layers(g)

    # Resize underlying graphs 
    resize!(g, nStates(g) + llength*lwidth)

    # If this is not the first, regenerate the views because state now points to new memory
    # And find the starting idx of the layer
    if length(glayers) != 0
        startidx = start(glayers[end]) + glength(glayers[end])*gwidth(glayers[end])
    else
        startidx = 1
    end

    #Make the new layer
    newlayer = IsingLayer(type, g, length(glayers)+1 , startidx, llength, lwidth; periodic)
    # Push it to layers
    push!(glayers, newlayer)
    # Give it an idx by placing it at the end
    push!(layeridxs(g), length(layers(g)))

    # Set the adjacency matrix
    if weightfunc != nothing 
        setAdj!(newlayer, weightfunc)
    end

    # Add Layer to defects
    addLayer!(defects(g), newlayer)
    
    return
end


function removeLayer!(g::IsingGraph, lidx::Integer)
    #if only one layer error
    if length(layers(g)) <= 1
        error("Cannot remove last layer")
    end

    layervec = layers(g)
    layer = layervec[lidx]

    # Resize the graph
    resize!(g, nStates(g) - glength(layer)*gwidth(layer), lidx)

    # Remove the layer from the graph defects
    removeLayer!(defects(g), lidx)

    # Remove the layer from the graph
    deleteat!(layervec, lidx)
    
    #Update the layer idxs
    updateLayerIdxs!(g)


    #Fix the layers to point towards the right spins
    for i in (lidx):length(layervec)
        llayer = layervec[i]
        layeridx(llayer, layeridx(llayer)-1)
        start(llayer, start(llayer) - glength(layer)*gwidth(layer))
        regenerateViews(llayer)
    end

end

removeLayer!(g::IsingGraph, layer::IsingLayer) = removeLayer!(g, layeridx(layer))

function updateLayerIdxs!(g::IsingGraph)
    for (i, layer) in enumerate(layers(g))
        layeridx(layer, i)
    end
end

# Set the SType
"""
Set the SType of the graph g
Only changes the pairs that are given
"""
function setSType!(g::IsingGraph, pairs::Pair...; refresh::Bool = true, force_refresh = false)
    oldstype = stype(g)
    newstype = changeSParam(oldstype, pairs...)
    if oldstype != newstype
        stype(g, newstype)
        if refresh && !force_refresh
            refreshSim(sim(g))
        end
    end
    if force_refresh
        refreshSim(sim(g))
    end
end
"""
Set the SType of the graph g to the given type
"""
function setSType!(g::IsingGraph, st::SType; refresh::Bool = true)
    oldstype = stype(g)
    if oldstype != st
        stype(g, st)
        if refresh
            refreshSim(sim(g))
        end
    end
end


export setSType!