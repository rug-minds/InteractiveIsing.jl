# Ising Graph Representation and functions
mutable struct IsingGraph{T <: Real, Sim <: IsingSim} <: AbstractIsingGraph{T}
    sim::Sim
    nStates::Int32

    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    htype::HType
    layers::Vector{IsingLayer{T}}
    layerconns::Dict{Set, Int32}
    defects::GraphDefects
    d::GraphData

    IsingGraph(sim, length, width; weightFunc::WeightFunc, continuous = false, weighted = false) = 
    (   
        type = continuous ? Float32 : Int8;
        g = new{type, typeof(sim)}(
            sim,
            0,
            type[],
            Vector{Vector{Conn}}[],
            HType(weighted, false),
            IsingLayer[],
            Dict{Pair, Int32}()
        );
        g.defects = GraphDefects(g);
        g.d = GraphData(g);
        addLayer!(g, length, width, periodic = periodic(weightFunc); weightFunc, );
        #For performance, don't know why
        g.adj = deepcopy(g.adj);
        return g
    )
    
end

#extend show to print out the graph, showing the length of the state, and the layers
function Base.show(io::IO, g::IsingGraph)
    println(io, "IsingGraph with $(nStates(g)) states")
    println(io, "Layers:")
    for (idx, layer) in enumerate(g.layers)
        show(io, layer)
        if idx != length(g.layers)
            print(io, "\n")
        end
    end
end


coords(g::IsingGraph) = VSI(layers(g), :coords)
export coords
layerdefects(g::IsingGraph) = VSI(layers(g), :defects)
export layerdefects

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

@inline layer(g::IsingGraph, idx) = g.layers[idx]

IsingGraph(g::IsingGraph) = deepcopy(g)

@inline layerdefects(g::IsingGraph) = layerdefects(defects(g))
@inline size(g::IsingGraph)::Tuple{Int32,Int32} = (nStates(g), 1)

function reset!(g::IsingGraph)
    state(g) .= initRandomState(g)
    currentlyWeighted = getHParam(g.htype, :Weighted)
    g.htype = HType(:Weighted => currentlyWeighted)
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
        return 2 .* rand(Float32, nstates) .- .5
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

@generated function ising_it(g::IsingGraph, htype::HType{Symbs,Params} = htype(g)) where {Symbs,Params}
    # Assumes :Defects will be found
    defects = getHParamType(htype, :Defects)

    if !defects
        return Expr(:block, :(return it::UnitRange{Int32} = 1:nStates(g)) )
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
initSqAdj(len, wid; weightFunc = defaultIsingWF) = createSqAdj(len, wid, weightFunc)

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

function addLayer!(g::IsingGraph, llength, lwidth; weightFunc = defaultIsingWF, periodic = true)
    glayers = layers(g)

    # Resize underlying graphs 
    resize!(g, nStates(g) + llength*lwidth)

    # If this is not the first, regenerate the views because state now points to new memory
    # And find the starting idx of the layer
    if length(glayers) != 0
        startidx = start(glayers[end]) + glength(glayers[end])*gwidth(glayers[end])
        # for layer in glayers
        #     regenerateViews(layer)
        # end
    else
        startidx = 1
    end

    #Make the new layer
    newlayer = IsingLayer(g, length(glayers)+1 , startidx, llength, lwidth; periodic)
    # Push it to layers
    push!(glayers, newlayer)

    setAdj!(newlayer, weightFunc)

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
