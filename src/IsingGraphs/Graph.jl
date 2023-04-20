# Ising Graph Representation and functions
mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph{T}
    nStates::Int32

    # Vertices and edges
    state::Vector{T}
    adj::Vector{Vector{Conn}}
    htype::HType
    layers::Vector{IsingLayer{T}}
    defects::GraphDefects
    d::GraphData



    # IsingGraph(type::DataType; length, width, state, adj, weighted = false) = 
    # (   
    #     g = new{type}(
    #         (length*width,1),
    #         length*width,
    #         state,
    #         adj,
    #         HType(weighted,false)
    #     );
        
    #     g.d = GraphData(g);
    #     g.defects = GraphDefects(nStates(g));
    #     # g.layers = [IsingLayer(g, 1, 1, length, width)];
    #     g.layers = IsingLayer[];
    #     return g
    # )

    IsingGraph(length, width; weightFunc::WeightFunc, continuous = false, weighted = false) = 
    (   
        type = continuous ? Float32 : Int8;
        g = new{type}(
            0,
            type[],
            Vector{Vector{Conn}}[],
            HType(weighted, false),
            IsingLayer[],
        );
        g.defects = GraphDefects(g);
        g.d = GraphData(g);
        addLayer!(g, length, width, weightfunc = weightFunc);
        #For performance for some reason
        g.adj = deepcopy(g.adj);
        return g
    )
    
end

#extend show to print out the graph, showing the length of the state, and the layers
function Base.show(io::IO, g::IsingGraph)
    println(io, "IsingGraph with $(nStates(g)) states")
    println(io, "Layers:")
    for layer in g.layers
        show(io, layer)
    end
end


coords(g::IsingGraph) = VSI(layers(g), :coords)
export coords
layerdefects(g::IsingGraph) = VSI(layers(g), :defects)
export layerdefects

@setterGetter IsingGraph
@forward IsingGraph GraphData d
@forward IsingGraph GraphDefects defects

@inline glength(g::IsingGraph) = size(g)[1]
@inline gwidth(g::IsingGraph) = size(g)[2]

@inline graph(g::IsingGraph) = g

@inline layer(g::IsingGraph, idx) = g.layers[idx]

IsingGraph(g::IsingGraph) = deepcopy(g)

@inline layerdefects(g::IsingGraph) = layerdefects(defects(g))
@inline size(g::IsingGraph) = (nStates(g), 1)

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

@generated function ising_it(g::IsingGraph, htype::HType{Symbs,Params}) where {Symbs,Params}
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

setdefect(g::IsingGraph, val, idx) = defects(g)[idx] = val

"""
Resize Graph to new size with random states and no connections for the new states
Need to be bigger than original size?
"""
function resize!(g::IsingGraph{T}, newlength, layeridx = 0) where T
    oldlength = nStates(g)
    sizediff = newlength - oldlength

    if sizediff < 0
        savedstates = state(g)[(endidx(layer(g,layeridx))+1):end]
        oldstart = startidx(layer(g,layeridx))
    end

    resize!(state(g), newlength)
    resize!(adj(g), newlength)

    if sizediff > 0
        randomstate = initRandomState(T, sizediff)
        state(g)[oldlength+1:newlength] .= randomstate
        adj(g)[oldlength+1:newlength] .= [Vector{Conn}[] for _ in oldlength+1:newlength]
    else
        state(g)[oldstart:end] .= savedstates
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

function addLayer!(g::IsingGraph, llength, lwidth; weightfunc = defaultIsingWF)
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
    newlayer = IsingLayer(g, length(glayers)+1 , startidx, llength, lwidth)
    # Push it to layers
    push!(glayers, newlayer)

    setAdj!(newlayer, weightfunc)

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