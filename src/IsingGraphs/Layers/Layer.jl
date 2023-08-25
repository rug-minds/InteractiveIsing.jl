"""
ndefects not tracking correctly FIX
"""

# Coordinates that can be left uninitialized
mutable struct Coords{T}
    cs :: Union{Nothing,T}
end

Coords(;y = 0, x = 0, z = 0) = Coords{Tuple{Int32,Int32,Int32}}((Int32(y), Int32(x), Int32(z)))
Coords(n::Nothing) = Coords{Tuple{Int32,Int32,Int32}}(nothing)
Coords(val::Integer) = Coords{Tuple{Int32,Int32,Int32}}((Int32(val), Int32(val), Int32(val)))
export Coords
mutable struct IsingLayer{T, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    graph::Union{Nothing, IsingGraphType}
    name::String
    # Internal idx of layer in shufflevec of graph
    internal_idx::Int32
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}

    connections::Dict{Pair{Int32,Int32}, WeightGenerator} 

    timers::Vector{Timer}

    defects::LayerDefects
    top::LayerTopology


    function IsingLayer(LayerType, g::GraphType, idx, start, length, width; name = "$(length)x$(width) Layer", coords = Coords(nothing), connections = Dict{Pair{Int32,Int32}, WeightGenerator}(), olddefects = 0, periodic::Union{Nothing,Bool} = true) where GraphType <: IsingGraph
        lsize = tuple(Int32(length), Int32(width))
        layer = new{LayerType, GraphType}(
            # Graph
            g,
            # Name
            name,
            # Layer idx
            Int32(idx),
            # Start idx
            Int32(start),
            # Size
            lsize,
            # Number of states
            Int32(lsize[1]*lsize[2]),
            #Coordinates
            coords,
            # Connections
            connections,
            # Timers
            Vector{Timer}(),
            # Layer data
            # LayerData(d(g), start, length, width)
        )
        # TODO: What's olddefects for
        layer.defects = LayerDefects(layer, olddefects)
        layer.top = LayerTopology(layer, [1,0], [0,1]; periodic)

        return layer
    end
end
export IsingLayer

# IsingLayer(g, layer::AbstractIsingLayer) = IsingLayer(g, layeridx(layer), start(layer), glength(layer), gwidth(layer), olddefects = ndefects(layer))

mutable struct IsingLayerCopy{T, IsingGraphType <: AbstractIsingGraph} <: AbstractIsingLayer{T}
    const graph::IsingGraphType
    layeridx::Int32
    state::Matrix{T}
    adj::Matrix{Vector{Tuple{Int32, Float32}}}
    start::Int32
    const size::Tuple{Int32,Int32}
    const nstates::Int32
    coords::Coords{Tuple{Int32,Int32,Int32}}
    # const d::LayerData
    defects::LayerDefects
    top::LayerTopology

    function IsingLayerCopy(layer::IsingLayer{A,B}) where {A,B}
        
        new{A, B}(
            # Graph
            layer.graph,
            # Layer idx at the time of copying
            layeridx(layer),
            # State
            copy(state(layer)),
            # Adj
            copy(adj(layer)),
            # Start idx
            layer.start,
            # Size
            layer.size,
            # Number of states
            layer.nstates,
            # Coordinates
            layer.coords,
            # Layer data
            # layer.d,
            layer.defects,
            layer.top
        )
    end
end



@setterGetter IsingLayer coords size layeridx graph
@setterGetter IsingLayerCopy coords size

# Extend show for IsingLayer, showing the layer idx, and the size of the layer
function Base.show(io::IO, layer::AbstractIsingLayer)
    showstr = "IsingLayer $(layeridx(layer)) with size $(size(layer))"
    if coords(layer) != nothing
        showstr *= " \t at coordinates $(coords(layer))"
    end
    print(io, showstr)
end

Base.show(io::IO, layertype::Type{<:AbstractIsingLayer}) = print(io, "IsingLayer")

@inline state(l::IsingLayer) = reshape((@view state(graph(l))[graphidxs(l)]), glength(l), gwidth(l))
@inline adj(l::IsingLayer) = reshape((@view adj(graph(l))[graphidxs(l)]), glength(l), gwidth(l))
@inline sp_adj(l::IsingLayer) = @view sp_adj(graph(l))[:, graphidxs(l)] 
export state, adj

# Get Graph
@inline function graph(layer::IsingLayer{T,G})::G where {T,G}
    g::G = layer.graph
    return g
end

@inline graph(layer::IsingLayer, g::IsingGraph) = layer.graph = g 


# Get current layeridx through graph
@inline layeridx(layer::IsingLayer) = externalidx(layers(graph(layer)), layer.internal_idx)
@inline idx(layer::IsingLayer) = internal_idx(layer)

@inline coords(layer::AbstractIsingLayer) = layer.coords.cs
# Move to user folder
@inline setcoords!(layer::AbstractIsingLayer; x = 0, y = 0, z = 0) = (layer.coords.cs = Int32.((y,x,z)))
@inline setcoords!(layer::AbstractIsingLayer, val) = (layer.coords.cs = Int32.((val,val,val)))
export setcoords!

@inline reladj(layer::AbstractIsingLayer) = adjGToL(layer.adj, layer)
# @forward IsingLayer LayerData d
@forward IsingLayer LayerDefects defects

# @forward IsingLayerCopy LayerData
@forward IsingLayerCopy LayerDefects defects

# Setters and getters
# @forward IsingLayer IsingGraph g
@inline size(layer::AbstractIsingLayer)::Tuple{Int32,Int32} = layer.size
@inline glength(layer::AbstractIsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[1]::Int32
@inline gwidth(layer::AbstractIsingLayer)::Int32 = (size(layer)::Tuple{Int32,Int32})[2]::Int32
@inline maxdist(layer::AbstractIsingLayer) = maxdist(layer, periodic(layer))
@inline maxdist(layer::AbstractIsingLayer, ::Type) = max(size(layer)...)
@inline function maxdist(layer::AbstractIsingLayer, ::Type{Periodic})
    l, w = size(layer)
    maxdist = dist(1,1, 1 + l÷2, 1 + w÷2, top(layer))
    return maxdist
end
export maxdist

@inline coordToIdx(i,j,layer::AbstractIsingLayer) = coordToIdx(latmod(i, glength(layer)), latmod(j, gwidth(layer)), glength(layer))
@inline idxToCoord(idx, layer::AbstractIsingLayer) = idxToCoord(idx, glength(layer))

@inline startidx(layer::AbstractIsingLayer) = start(layer)
@inline endidx(layer::AbstractIsingLayer) = start(layer) + glength(layer)*gwidth(layer) - 1

@inline getindex(layer::AbstractIsingLayer, idx) = state(layer)[idx]
@inline setindex!(layer::AbstractIsingLayer, val, idx) = state(layer)[idx] = val
@inline Base.in(idx::Integer, layer::IsingLayer) = idx ∈ graphidxs(layer)


"""
Range of idx of layer for underlying graph
"""
@inline graphidxs(layer::AbstractIsingLayer) = UnitRange{Int32}(start(layer):endidx(layer))
export graphidxs

bfield(layer::AbstractIsingLayer) = reshape((@view bfield(graph(layer))[graphidxs(layer)]), glength(layer), gwidth(layer))
clamps(layer::AbstractIsingLayer) = reshape((@view clamps(graph(layer))[graphidxs(layer)]), glength(layer), gwidth(layer))

# Inherited from Graph
@inline htype(layer::AbstractIsingLayer) = layer.graph.htype
@inline nStates(layer::AbstractIsingLayer) = length(graphidxs(layer))
@inline sim(layer::AbstractIsingLayer) = sim(graph(layer))

#forward alivelist
aliveList(layer::AbstractIsingLayer) = aliveList(defects(layer))
#forward defectList
defectList(layer::AbstractIsingLayer) = defectList(defects(layer))

@inline hasDefects(layer::AbstractIsingLayer) = ndefects(defects(layer)) > 0
@inline setdefect(layer::AbstractIsingLayer, val, idx) = defects(layer)[idx] = val

iterator(layer::AbstractIsingLayer) = start(layer):endidx(layer)
iterator(g::IsingGraph) = 1:(nStates(g))

# LayerTopology
@inline periodic(layer::AbstractIsingLayer) = periodic(top(layer))
@inline setPeriodic!(layer::AbstractIsingLayer, periodic) = top!(layer, LayerTopology(top(layer); periodic))
@inline dist(idx1::Integer, idx2::Integer, layer::AbstractIsingLayer) = dist(idxToCoord(idx1, glength(layer))..., idxToCoord(idx2, glength(layer))..., top(layer))
@inline idxToCoord(idx::Integer, layer::AbstractIsingLayer) = idxToCoord(idx, glength(layer))

export setPeriodic!


# Forward Graph Data
# @inline bfield(layer::IsingLayer) = @view bfield(graph(layer))[start(layer):endidx(layer)]

"""
Go from a local idx of layer to idx of the underlying graph
"""
@inline function idxLToG(layer::IsingLayer, idx::Integer)::Int32
    return Int32(start(layer) + idx - 1)
end

"""
Go from a local matrix indexing of layer to idx of the underlying graph
"""
@inline function idxLToG(layer::IsingLayer, i::Integer, j::Integer)::Int32
    return Int32(start(layer) + coordToIdx(i,j, glength(layer)) - 1)
end

@inline idxLToG(i::Integer, j::Integer, layer::IsingLayer) = idxLToG(layer, i, j)
idxLToG(layer, tup::Tuple) = idxLToG(layer, tup[1], tup[2])

"""
Go from graph idx to idx of layer
"""
@inline function idxGToL(layer::IsingLayer, idx::Integer)
    return Int32(idx + 1 - start(layer))
end
@inline function idxGToL(idx::Integer, layer::IsingLayer)
    return Int32(idx + 1 - start(layer))
end


"""
Convert adjacency matrix of a layer to adjacency matrix with idxs of underlying graph
"""
function adjLToG(adj, layer)
    for entry in adj
        for idx in 1:length(entry)
            entry[idx] = tuple(idxLToG(layer, connIdx(entry[idx])), connW(entry[idx]))
        end
    end
end
# debug
export adjLToG

# What is this used for?
function adjGToL(adj::AbstractArray{T}, layer) where T
    newadj = Vector{T}(undef, length(adj))
    for adjidx in 1:length(adj)
        entry = adj[adjidx]
        newentry = typeof(entry)(undef, length(entry))
        for entryidx in 1:length(entry)
            conn = entry[entryidx]
            newentry[entryidx] = tuple(idxGToL(layer, connIdx(conn)), connW(conn))
        end
        newadj[adjidx] = newentry
    end
    return newadj
end
export adjGToL

function setAdj!(layer::IsingLayer, wf = defaultIsingWF)
    
    Threads.@threads for idx in eachindex(adj(layer))
    # for idx in eachindex(adj(layer))
        idxs_weights = getUniqueConnIdxs(wf, idx, glength(layer), gwidth(layer), wt(wf))
        
        for idx_weight in idxs_weights
            addWeight!(layer, idx, idx_weight[1], idx_weight[2])
        end
    end
end

# Set the SType
setSType!(layer::AbstractIsingLayer, varargs...; refresh = true) = setSType!(graph(layer), varargs...; refresh = refresh)
