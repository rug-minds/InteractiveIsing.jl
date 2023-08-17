abstract type StateType end
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end



# Ising Graph Representation and functions
mutable struct IsingGraph{T <: Real} <: AbstractIsingGraph{T}
    sim::Union{Nothing, IsingSim}
    nStates::Int32

    # Vertices and edges
    state::Vector{T}
    const adj::Vector{Vector{Conn}}
    sp_adj::SparseMatrixCSC{Float32,Int32}
    htype::HType

    stype::SType
    
    layers::ShuffleVec{IsingLayer}

    continuous::StateType
    # Connection between layers, I don't think it's neccesary to track this
    layerconns::Dict{Set, Int32}
    defects::GraphDefects
    d::GraphData

    function IsingGraph(sim, length, width; periodic = nothing, weightfunc::Union{Nothing,WeightFunc} = nothing, continuous = false, weighted = false)
      
        type = continuous ? Float32 : Int8
        g = new{type}(
            sim,
            0,
            type[],
            # Adj
            Vector{Vector{Conn}}[],
            SparseMatrixCSC{Float32,Int32}(undef,0,0),
            HType(weighted, false),
            SType(:Weighted => weighted),
            #Layers
            ShuffleVec{IsingLayer}(),
            #ContinuityType
            continuous ? ContinuousState() : DiscreteState(),
            Dict{Pair, Int32}()
        )

        g.defects = GraphDefects(g)
        g.d = GraphData(g)

        # addLayer!(g, length, width; periodic, weightfunc, type)
        return g
    end
    
end

function tuples2sparse(adj)
    colidx_len = 0
    for col in adj
        colidx_len += length(col)
    end
    colidx = Vector{Int32}(undef, colidx_len)
    colidxidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            colidx[colidxidx] = Int32(idx)
            colidxidx += 1
        end
    end

    rowidx = Vector{Int32}(undef, colidx_len)
    rowidxidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            rowidx[rowidxidx] = Int32(adj[idx][i][1])
            rowidxidx += 1
        end
    end

    vals = Vector{Float32}(undef, colidx_len)
    valsidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            vals[valsidx] = adj[idx][i][2]
            valsidx += 1
        end
    end
    return deepcopy(sparse(rowidx, colidx, vals))
end
export tuples2sparse

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
@inline Base.getindex(g::IsingGraph, idx) = g.layers[idx]
@inline length(g::IsingGraph) = length(g.layers)
Base.view(g::IsingGraph, idx) = view(g.layers, idx)

#TODO: Give new idx
@inline function layerIdx!(g, oldidx, newidx)
    shuffle!(g.layers, oldidx, newidx)
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

"""
Add a layer to graph g.
addLayer(g::IsingGraph, length, width)

Give keyword argument weightfunc to set a weightfunc.
If weightfunc = :Default, uses default weightfunc for the Ising Model
"""
function addLayer!(g::IsingGraph, llength, lwidth; weightfunc = nothing, periodic = true, type = eltype(state(g)))
    glayers = layers(g)

    newlayer = 0

    @tryLockPause sim(g) begin
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

        # Set the adjacency matrix
        if weightfunc != nothing 
            setAdj!(newlayer, weightfunc)
            genSPAdj!(newlayer, weightfunc)
        end

        # Add Layer to defects
        addLayer!(defects(g), newlayer)

        # Update the layer idxs
        nlayers(sim(g))[] += 1
    end

    if weightfunc == :Default
        println("No weightfunc given, using default")
        genAdj!(newlayer, wg_isingdefault)
    end

    return
end


function removeLayer!(g::IsingGraph, lidx::Integer)
    #if only one layer error
    if length(layers(g)) <= 1
        error("Cannot remove last layer")
    end


    # lockPause(sim(g))
    @tryLockPause sim(g)  begin
        # If the slected layer is after the layer to be removed, decrement layerIdx
        if layerIdx(sim(g))[] >= lidx && layerIdx(sim(g))[] > 1
            layerIdx(sim(g))[] -= 1
        end

        layervec = layers(g)
        layer = layervec[lidx]
        nstates_layer = glength(layer)*gwidth(layer)

        i_idx = internal_idx(layer)

        # Resize the graph
        resize!(g, nStates(g) - nstates_layer, lidx)

        # Remove the layer from the graph defects
        removeLayer!(defects(g), lidx)

        # Remove the layer from the graph
        deleteat!(layervec, lidx) do layer, newidx
            internal_idx(layer, newidx)
            start(layer, start(llayer) - nstates_layer)
            regenerateViews(layer)
        end

        nlayers(sim(g))[] -= 1
    end

    # unpause sim
    # unlockPause(sim(g))


    return

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