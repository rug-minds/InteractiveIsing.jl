#TODO: SHouldn't have the same supertype as layer statetype
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end


getIntType(::Float64) = Int64
getFloatType(::Float64) = Float64
isarchitecturetype(::Any) = @show false
isarchitecturetype(t::Tuple{A,B,C}) where {A,B,C} = (A<:Integer && B<:Integer && t[3]<:StateType)
# Ising Graph Representation and functions
mutable struct IsingGraph{T <: AbstractFloat} <: AbstractIsingGraph{T}
    # Simulation
    sim::Union{Nothing, IsingSim}
    # Vertices and edges
    state::Vector{T}
    sp_adj::SparseMatrixCSC{T,Int32}
    temp::T

    default_algorithm::Function
    stype::SType
    
    layers::ShuffleVec{IsingLayer}

    continuous::StateType
    # Connection between layers, Could be useful to track for faster removing of layers
    layerconns::Dict{Set, Int32}
    params::Tuple


    defects::GraphDefects
    d::GraphData


    # Default Initializer for IsingGraph
    function IsingGraph(sim = nothing, length = nothing, width=nothing; periodic = nothing, weights::Union{Nothing,WeightGenerator} = nothing, type = Continuous, weighted = false, precision = Float32, kwargs...)
        architecture = searchkey(kwargs, :architecture, fallback = nothing)
        @assert (!isnothing(length) && !isnothing(width)) || !isnothing(architecture) "Either give length and width or architecture"
        
        println("Architecture: $architecture")

        if isnothing(architecture)
            architecture = [(length, width, type)]
        end
        for idx in eachindex(architecture)
            if !isarchitecturetype(architecture[idx])
                architecture[idx] = (architecture[idx][1], architecture[idx][2], Continuous)
            end
        end
        println("Architecture: $architecture")

        g = new{precision}(
            sim,
            precision[],
            SparseMatrixCSC{precision,Int32}(undef,0,0),
            #Temp            
            1f0,
            # Default algorithm
            updateMetropolis,
            SType(:Weighted => weighted),
            #Layers
            ShuffleVec{IsingLayer}(relocate = relocate!),
            #ContinuityType
            ContinuousState(),
            Dict{Pair, Int32}(),
            ()
        )

        g.defects = GraphDefects(g)
        # Couple the shufflevec and the defects
        internalcouple!(g.layers, g.defects, (layer) -> Int32(0), push = addLayer!, insert = (obj, idx, item) -> addLayer!(obj, item), deleteat = removeLayer!)

        g.d = GraphData(precision, g)
        for arc in architecture
            _addLayer!(g, arc[1], arc[2]; weights, periodic, type = arc[3], kwargs...)
        end
        return g
    end
    
    # Constructor for copying from other graph or savedata.
    function IsingGraph(
        state,
        sp_adj,
        stype,
        layers,
        continuous,
        defects,
        data
        )
        return new{eltype(state)}(
            # Sim
            nothing,
            #state
            state,
            # Adjacency
            sp_adj,
            #Temp
            1f0,
            # Default algorithm
            updateMetropolis,
            # stype
            stype,
            # Layers
            layers,
            # Continuous
            continuous,
            # Connections between layers
            Dict{Pair, Int32}(),
            #params
            (),
            # Defects
            defects,
            # Data
            data
        )
    end
end

Base.eltype(::IsingGraph{T}) where T = T
Base.eltype(::Type{IsingGraph{T}}) where T = T

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

function destructor(g::IsingGraph)
    destructor.(layers(g))
end

Base.show(io::IO, graphtype::Type{IsingGraph}) = print(io, "IsingGraph")

coords(g::IsingGraph) = VSI(layers(g), :coords)
export coords

@inline setrange!(g::AbstractIsingGraph, clamp, idxs) = setrange!(defects(g), clamp, idxs)

@setterGetter IsingGraph sp_adj
@inline nStates(g) = length(state(g))
@inline sp_adj(g::IsingGraph) = g.sp_adj
@inline function sp_adj(g::IsingGraph, sp_adj)
    @assert sp_adj.m == sp_adj.n == length(state(g))
    g.sp_adj = sp_adj
    # Add callbacks field to graph, which is a Dict{typeof(<:Function), Vector{Function}}
    # And create a setterGetter macro that includes the callbacks
    restart(g)
    return sp_adj
end
set_sp_adj!(g::IsingGraph, vecs::Tuple) = sp_adj(g, sparse(vecs..., nStates(g), nStates(g)))
export sp_adj

@forward IsingGraph GraphData d
@forward IsingGraph GraphDefects defects

@inline glength(g::IsingGraph)::Int32 = size(g)[1]
@inline gwidth(g::IsingGraph)::Int32 = size(g)[2]

@inline graph(g::IsingGraph) = g

### Access the layer ###
@inline function spinidx2layer(g::IsingGraph, idx)::IsingLayer
    @assert idx <= nStates(g) "Index out of bounds"
    for layer in unshuffled(layers(g))
        if idx ∈ layer
            return layer
        end
    end
    return g[1]
end
layeridxs(g::IsingGraph) = UnitRange{Int32}[graphidxs(unshuffled(layers(g))[i]) for i in 1:length(g)]
@inline spinidx2layer_i_index(g, idx) = internal_idx(spinidx2layer(g, idx))
@inline layer(g::IsingGraph, idx) = g.layers[idx]
@inline Base.getindex(g::IsingGraph, idx) = g.layers[idx]
@inline Base.getindex(g::IsingGraph) = g.layers[1]
@inline length(g::IsingGraph) = length(g.layers)
@inline Base.lastindex(g::IsingGraph) = length(g)
Base.view(g::IsingGraph, idx) = view(g.layers, idx)

# Base.deleteat!(layervec::ShuffleVec{IsingLayer}, lidx::Integer) = deleteat!(layervec, lidx) do layer, newidx
#     internal_idx(layer, newidx)
#     start(layer, start(layer) - nstates_layer)
# end

function processes(g::IsingGraph)
    return processes(sim(g))[map(process -> process.objectref === g, processes(sim(g)))]
end


#TODO: Give new idx
@inline function layerIdx!(g, oldidx, newidx)
    shuffle!(g.layers, oldidx, newidx)
end
export layerIdx!

IsingGraph(g::IsingGraph) = deepcopy(g)

# @inline layerdefects(g::IsingGraph) = layerdefects(defects(g))
# export layerdefects

@inline size(g::IsingGraph)::Tuple{Int32,Int32} = (nStates(g), 1)

function closetimers(g::IsingGraph)
    for layer in layers(g)
        close.(timers(layer))
        deleteat!(timers(layer), 1:length(timers(layer)))
    end
end

function reset!(g::IsingGraph)
    state(g) .= initRandomState(g)
    currentlyWeighted = getSParam(stype(g), :Weighted)
    stype(g,SType(:Weighted => currentlyWeighted))
    reset!(defects(g))
    reset!(d(g))
    closetimers(g)
end
 
export initRandomState
""" 
Initialize from a graph
"""
# (initRandomState(g::IsingGraph{type})::Vector{type}) where type = initRandomState(type, glength(g), gwidth(g))
# initRandomState(type, glength, gwidth)::Vector{type} = initRandomState(type, glength*gwidth)
function initRandomState(g)
    _state = similar(state(g))
    for layer in unshuffled(layers(g))
        _state[graphidxs(layer)] .= rand(layer, length(graphidxs(layer)))
    end
    return _state
end

#=
Methods
=#
function stateiterator(g::IsingGraph)
    if hasDefects(g)
        return aliveList(g)
    else
        return UnitRange{Int32}(1:nStates(g))
    end
end
# Doesn't need to use multiple dispatch
""" 
Returns in iterator which can be used to choose a random index among alive spins
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

"""
Initialization of adjacency Vector for a given N
and using a weightFunc
Is a pointer to function in SquareAdj.jl for compatibility
"""
# initSqAdj(len, wid; weights = defaultIsingWF) = createSqAdj(len, wid, weights)

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
function Base.resize!(g::IsingGraph{T}, newlength, startidx = length(state(g))) where T
    oldlength = nStates(g)
    sizediff = newlength - oldlength

    if sizediff > 0
        resize!(state(g), newlength)
        randomstate = rand(T, sizediff)
        state(g)[oldlength+1:newlength] .= randomstate
        idxs_to_add = startidx:(startidx + sizediff - 1)
        g.sp_adj = insertrowcol(sp_adj(g), idxs_to_add)
        resize!(d(g), newlength)
    else # if making smaller
        idxs_to_remove = startidx:(startidx + abs(sizediff) - 1)
        deleteat!(state(g), idxs_to_remove)
        g.sp_adj = deleterowcol(sp_adj(g), idxs_to_remove)
        
        # Resize data
        resize!(d(g), newlength, idxs_to_remove)

    end
    
    return g
end

export resize!

export addLayer!

function addLayer!(g::IsingGraph, llength, lwidth; weights = nothing, periodic = true, type = default_ltype(g), set = convert.(eltype(g),(-1,1)), rangebegin = set[1], rangeend = set[2], kwargs...)
    @tryLockPause sim(g) begin 
        _addLayer!(g, llength, lwidth; set, weights, periodic, type, kwargs...)
        # Update the layer idxs
        nlayers(sim(g))[] += 1
    end
    return layers(g)
end

addLayer!(g::IsingGraph, llength, lwidth, wg; kwargs...) = addLayer!(g, llength, lwidth; weights = wg, kwargs...)

function addLayer!(g, dims::Vector, wgs...; kwargs...)
    for (dimidx,dim) in enumerate(dims)
        addLayer!(g, dim[1], dim[2], wgs[dimidx]; kwargs...)
    end
    return layers(g)
end

"""
Add a layer to graph g.
addLayer(g::IsingGraph, length, width)

Give keyword argument weightfunc to set a weightfunc.
If weightfunc = :Default, uses default weightfunc for the Ising Model

When layer needs to be inserted, layers are shifted around
This is handled by the relocate! function automatically in the shufflevec
Because the shufflevec knows then internal data is being pushed around
Not sure if this is the most transparent way to do it since resizing is not done within the shufflevec
"""
function _addLayer!(g::IsingGraph{T}, llength, lwidth; weights = nothing, periodic = true, type = nothing, kwargs...) where T
    if isnothing(type)
        type = default_ltype(g)
    end

    set = searchkey(kwargs, :set, fallback = convert.(eltype(g),(-1,1)))
    set = T.(set) # Safeness
   
    # Function that makes the new layer based on the insertidx
    # Found by the shufflevec
    # Maybe I should make an insert function for the layers?
    make_newlayer(idx) = begin
        _layers = unshuffled(layers(g))

        extra_states = llength*lwidth
        # Resize the old state

        # Find the startidx of the new layer
        # Based on the insertidx found by the shufflevec
        if !isempty(_layers)
            _startidx = endidx(_layers[idx-1]) + 1
        else
            _startidx = 1
        end

        resize!(g, nStates(g) + extra_states, _startidx)


        return IsingLayer(type, g, idx , _startidx, llength, lwidth; periodic, set)
    end
    layertype =  IsingLayer{type, set, typeof(g)}
    push!(layers(g), make_newlayer, layertype)
    newlayer = layers(g)[end]

    # If layer of different type, change default algorithm to layered updateMetropolis
    if hasmixedtypes(layers(g)) && g.default_algorithm == updateMetropolis
        g.default_algorithm = layeredMetropolis
    end

    # Generate the adjacency matrix from the weightfunc
    if !isnothing(weights)
        genAdj!(newlayer, weights)
    elseif weights == :Default
        println("No weightgenerator given, using default")
        genAdj!(newlayer, wg_isingdefault)
    end

    # SET COORDS
    setcoords!(g[end], z = length(g)-1)

    # Init the state
    initstate!(newlayer)

    return layers(g)
end

function _removeLayer!(g::IsingGraph, lidx::Integer)
    #if only one layer error
    if length(layers(g)) <= 1
        error("Cannot remove last layer")
    end

    # Remove the layer from the graph
    layervec = layers(g)
    layer = layervec[lidx]

    # Remove the layer from the graph
    deleteat!(layervec, lidx)

    resize!(g, nStates(g) - nStates(layer), start(layer))

    return layers(g)

end

function removeLayer!(g::IsingGraph, lidx::Integer)
    @tryLockPause sim(g) begin 
        _removeLayer!(g, lidx) 
         # If the slected layer is after the layer to be removed, decrement layerIdx
        if layerIdx(sim(g))[] >= lidx && layerIdx(sim(g))[] > 1
            layerIdx(sim(g))[] -= 1
        else
            notify(layerIdx(sim(g)))
        end
        nlayers(sim(g))[] -= 1 
    end
    return layers(g)
end
removeLayer!(layer::IsingLayer) = removeLayer!(graph(layer), layer)
export removeLayer!

function removeLayer!(g, idxs::Vector{Int}) 
    _layers = layers(g)
    # Sort by internal storage order from last to first, this causes minimal relocations
    sort!(idxs, lt = (x,y) -> internalidx(_layers, x) > internalidx(_layers, y))
    @tryLockPause sim(g) for idx in idxs
        _removeLayer!(g, idx)
        nlayers(sim(g))[] -= 1
    end
    return layers(g)
end
removeLayer!(g::IsingGraph, layer::IsingLayer) = removeLayer!(g, layeridx(layer))

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
            restart(g)
        end
    end
    
    if force_refresh
        restart(g)
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
            restart(g)
        end
    end
end


export setSType!





### ETC ###

### Old adjacency list stuff ###
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