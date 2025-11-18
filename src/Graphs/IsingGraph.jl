#TODO: SHouldn't have the same supertype as layer statetype
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end

intprecision(::Type{Float16}) = Int16
intprecision(::Type{Float32}) = Int32
intprecision(::Type{Float64}) = Int64

# Ising Graph Representation and functions
mutable struct IsingGraph{T <: AbstractFloat, M <: AbstractMatrix{T}, Layers} <: AbstractIsingGraph{T}
    # Vertices and edges
    state::Vector{T}
    # Adjacency Matrix
    adj::M
    self::ParamVal{Vector{T}}
    
    temp::T

    default_algorithm::MCAlgorithm
    hamiltonian::Hamiltonian
    

    # Connection between layers, Could be useful to track for faster removing of layers
    layerconns::Dict{Set, Int32}
    # params::Parameters #TODO: Make this a custom type?

    # For notifying simulations or other things
    # emitter::Emitter

    defects::GraphDefects
    # d::GraphData{T} #Other stuff. Maybe just make this a dict?
    addons::Dict{Symbol, Any}

    layers::Layers
end

function IsingGraph(dims::Int...; 
                    periodic = nothing, 
                    sets = nothing, 
                    weights::Union{Nothing,WeightGenerator} = nothing,
                    type = Continuous,
                    kwargs...)
    single_layer = Layer(dims...; stateset = sets, stype = type, weights, periodic, kwargs...)
    IsingGraph(single_layer; kwargs...)
end


"""
Default initializer for IsingGraphs
    Give layers in the following format:
    (Layer1, Layer2, ...)
    or
    (Layer1, Weightgenerator12, Layer2, Layer3, WeightGenerator34)
"""
function IsingGraph(layers_or_wgs::Union{AbstractLayerProperties, WeightGenerator}...; precision = Float32, kwargs...)
    layers_props = tuple((layers_or_wgs[i] for i in eachindex(layers_or_wgs) if isa(layers_or_wgs[i], LayerProperties))...)

    # Fill the weight generator list
    layer_idx = 0
    wgs = []
    for l_or_wgs in layers_or_wgs
        if l_or_wgs isa WeightGenerator
            push!(wgs, (layer_idx => layer_idx + 1 , l_or_wgs))
        else
            layer_idx += 1
        end 
    end

    # Initialize layers 
    l_startidx = 1
    layers = ntuple(i -> begin
        l = IsingLayer(nothing, i, l_startidx, layers_props[i])
        l_datalen = datalen(layers_props[i])
        l_startidx += l_datalen
        l
    end,
    length(layers_props))
    

    _datalen = l_startidx - 1
    

    self = StaticParam(0f0, _datalen, description = "Self Connections")


    g = IsingGraph{precision, SparseMatrixCSC{precision,Int32}, typeof(layers)}(
        # sim,
        zeros(precision, _datalen),
        SparseMatrixCSC{precision,intprecision(precision)}(undef,_datalen,_datalen),
        self,
        #Temp            
        1f0,
        # Default algorithm
        LayeredMetropolis(),
        #Hamiltonians
        Ising(precision, _datalen),
        #Layers
        Dict{Pair, Int32}(),
        #Emitter
        # Emitter(Observable[]),
        #Defects
        GraphDefects(nothing),
        Dict{Symbol, Any}(),
        layers
    )


    # Set Graph refs
    g.defects.graph = g
    for layer in g.layers
        layer.graph = g
        # println("Making weights for layer ", layer)
        genAdj!(layer, get_weightgenerator(layer))
    end

    initRandomState(g)
    # cb = x -> layerIdx(sim(x))
    # set_listener_callback!(g, cb)
    # println("cb: ", cb)

    prepare(g.default_algorithm, (;g))
    return g
end

function makelayers(g, arcs)
    return ntuple(i -> IsingLayer(g, arcs[i]), length(arcs))
end

nlayers(g::IsingGraph) = length(g.layers)

"""
The user gives:
    [(length, width, height, type), ...]
    where type = Continuous or Discrete
    this puts the data in the correct format
"""
function decode_architecture(arcs)
    num_layers = length(arcs)
    architecture = []
    # for layer in arcs
    #     # If last is a state type, just push
    #     if layer[end] isa Type && layer[end] <: StateType
    #         if length(layer) == 3
    #             push!(architecture, (layer[1:2]..., nothing, layer[3]))
    #         else
    #             push!(architecture, (layer..., Continuous()))
    #         end
    #     else # add default state type
    #         if length(layer) == 2
    #             push!(architecture, (layer..., nothing, Continuous()))
    #         else
    #             push!(architecture, (layer..., Continuous()))
    #         end
    #     end
    # end
    return architecture
end

"""
User gives, for discrete: (s1,s2,s3,...)
For continuous: (s1,s2) which are the intervals
"""
function decode_statesets(sets, numlayers, precision)
     # Create the sets for each layer
    if isnothing(sets) # Just make some sets
        sets = repeat([convert.(precision,(-1,1))], numlayers)
    else # Correct the given sets
        sets = map(x->convert.(precision, x), sets)
        if length(sets) < numlayers
            lengthdiff = numlayers - length(sets)
            for _ in 1:lengthdiff # If less sets than layers, add default set
                push!(sets, convert.(precision,(-1,1)))
            end
        else
            sets = sets[1:numlayers]
        end
    end
    return sets
end

function arch_to_datalen(architecture, idx = length(architecture))
    total = 0
    for layer in architecture[1:idx]
        prod = 1
        for dim in 1:3
            if !isnothing(layer[dim]) && layer[dim] isa Real
                prod *= layer[dim]
            end
        end
        total += prod
    end
    return total
end

function arch_to_startidxs(architecture)
    startidxs = Int32[1]
    total = 1
    for layer in architecture
        prod = 1
        for dim in 1:3
            if !isnothing(layer[dim]) && layer[dim] isa Real
                prod *= layer[dim]
            end
        end
        total += prod
        push!(startidxs, total)
    end
    return startidxs
end

Base.eltype(::IsingGraph{T}) where T = T
Base.eltype(::Type{IsingGraph{T}}) where T = T

#extend show to print out the graph, showing the length of the state, and the layers
function Base.show(io::IO, g::IsingGraph)
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

@inline clamprange!(g::IsingGraph, clamp, idxs) = setrange!(defects(g), clamp, idxs)
export clamprange!

@setterGetter IsingGraph adj params
# @inline params(g::IsingGraph) = g.params
export params
@inline nStates(g::IsingGraph) = length(state(g))

@inline nstates(g) = length(state(g))
export nstates

@inline adj(g::IsingGraph) = g.adj
@inline function adj(g::IsingGraph, adj)
    @assert adj.m == adj.n == length(state(g))
    g.adj = adj
    # Add callbacks field to graph, which is a Dict{typeof(<:Function), Vector{Function}}
    # And create a setterGetter macro that includes the callbacks
    refresh(g)
    return adj
end
set_adj!(g::IsingGraph, vecs::Tuple) = adj(g, sparse(vecs..., nStates(g), nStates(g)))
export adj

# @forwardfields IsingGraph GraphData d
@forwardfields IsingGraph GraphDefects defects graph

@inline glength(g::IsingGraph)::Int32 = size(g)[1]
@inline gwidth(g::IsingGraph)::Int32 = size(g)[2]
@inline graph(g::IsingGraph) = g

### Access the layer ###
@inline function spinidx2layer(g::IsingGraph, idx)
    @assert idx <= nStates(g) "Index out of bounds"
    # for layer in unshuffled(layers(g))
    for layer in layers(g)
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
@inline Base.length(g::IsingGraph) = length(g.layers)
@inline Base.lastindex(g::IsingGraph) = length(g)
Base.view(g::IsingGraph, idx) = view(g.layers, idx)
@inline graphidxs(g::IsingGraph) = Int32(1):Int32(nStates(g))
Base.get!(g::IsingGraph, s, d) = get!(g.addons, s, d)
Base.get(g::IsingGraph, s) = get(g.addons, s)
Base.get(g::IsingGraph, s, d) = get(g.addons, s, d)

# function set_listener_callback!(g, f::Function)
#     if !haskey(g.addons, :listener_callback)
#         g.addons[:listener_callback] = Function[f]
#     else
#         push!(g.addons[:listener_callback], f)
#     end
# end

# function listener_callback(g)
#     for f in get(g.addons, :listener_callback, Function[])
#         f(g)
#     end
# end


function Base.convert(::Type{<:AbstractIsingLayer}, g::IsingGraph)
    @assert length(g.layers) == 1 "Graph has more than one layer, ambiguous"
    return g.layers[1]
end 

function processes(g::IsingGraph)
    get!(g, :processes, Process[])::Vector{Process}
end
processes(::Nothing) = nothing
# Get the first process
function process(g::IsingGraph)
    ps = get!(g, :processes, Process[])
    if isempty(ps)
        return nothing
    end
    return ps[end]
end
export process


#TODO: Give new idx
@inline function layerIdx!(g, oldidx, newidx)
    shuffle!(g.layers, oldidx, newidx)
end
export layerIdx!

IsingGraph(g::IsingGraph) = deepcopy(g)

@inline Base.size(g::IsingGraph)::Tuple{Int32,Int32} = (nStates(g), 1)

function closetimers(g::IsingGraph)
    for layer in layers(g)
        close.(timers(layer))
        deleteat!(timers(layer), 1:length(timers(layer)))
    end
end

function reset!(g::IsingGraph)
    state(g) .= initRandomState(g)
    closetimers(g)
end
 
statelen(g::IsingGraph) = length(state(g))
export statelen
export initRandomState
""" 
Initialize from a graph
"""
function initRandomState(g)
    _state = similar(state(g))
    # for layer in unshuffled(layers(g))
    for layer in layers(g)
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
function ising_it(g)
    defects = hasDefects(g)
    if !defects
        return UnitRange{Int32}(1:nStates(g))
    else
        return aliveList(g)
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

### ARCHITECTURE
function getarchitecture(g)
    architecture = []
    for layer in layers(g)
        push!(architecture, (glength(layer), gwidth(layer), statetype(layer)))
    end
    return architecture
end

function compare_architecture_sizes(architecture1, architecture2)
    if length(architecture1) != length(architecture2)
        return false
    else
        for (idx, layer) in enumerate(architecture1)
            if layer[1] != architecture2[idx][1] || layer[2] != architecture2[idx][2]
                return false
            end
        end
    end
    return true
end

### SELF ENERGY
@inline function activateself!(g)
    g.self = activate(g.self) # Ensure self is active
    refresh(g)
end
@inline function disableself!(g)
    g.self = deactivate(g.self) # Ensure self is inactive
    refresh(g)
end
@inline function homogeneousself!(g, val = 1)
    g.self = sethomogeneousval(g.self, val) # Set self to zero
    refresh(g)
end
export activateself!, disableself!, homogeneousself!


#### SAVE

# Constructor for copying from other graph or savedata.
# function IsingGraph(
#     state,
#     adj,
#     layers,
#     defects,
#     data,
#     Hamiltonians = Ising(g)
#     )
# return IsingGraph(
# # Sim
# nothing,
# #state
# state,
# # Adjacency
# adj,            
# #Temp
# 1f0,
# # Default algorithm
# updateMetropolis,
# #Hamiltonians
# Hamiltonians,
# # Layers
# layers,
# # Connections between layers
# Dict{Pair, Int32}(),
# #params
# # (;self = ParamVal(zeros(eltype(state), length(state)), 0, "Self Connections", false)),
# # For notifying simulations or other things
# Emitter(Observable[]),
# # Defects
# defects,
# # Data
# data,
# # Processes
# Vector{Process}()
# )
# end