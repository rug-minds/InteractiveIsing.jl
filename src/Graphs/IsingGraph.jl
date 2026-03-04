#TODO: SHouldn't have the same supertype as layer statetype
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end

intprecision(::Type{Float16}) = Int16
intprecision(::Type{Float32}) = Int32
intprecision(::Type{Float64}) = Int64

# Ising Graph Representation and functions
mutable struct IsingGraph{T <: AbstractFloat, M, Layers, N} <: AbstractIsingGraph{T}
    # Vertices and edges
    state::AbstractArray{T,N}
    # Adjacency Matrix
    adj::M
    # self::AbstractArray{T,1} # Diagonal of adj stored as a separate array for efficiency
    
    temp::T

    default_algorithm::ProcessAlgorithm
    hamiltonian::Hamiltonian
    
    # Connection between layers, Could be useful to track for faster removing of layers
    # layerconns::Dict{Set, Int32}

    defects::GraphDefects
    # d::GraphData{T} #Other stuff. Maybe just make this a dict?
    addons::Dict{Symbol, Any}

    layers::Layers
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

@setterGetter IsingGraph adj params layers
# @inline params(g::IsingGraph) = g.params
export params
@inline nStates(g::IsingGraph) = length(state(g))::Int

@inline nstates(g) = length(state(g))::Int
export nstates

@inline adj(g::IsingGraph) = g.adj
@inline function adj!(g::IsingGraph, adj)
    @assert size(adj, 1) == length(state(g)) && size(adj, 2) == length(state(g))
    g.adj = adj
    # Add callbacks field to graph, which is a Dict{typeof(<:Function), Vector{Function}}
    # And create a setterGetter macro that includes the callbacks
    reinit(g)
    return adj
end
set_adj!(g::IsingGraph, vecs::Tuple) = adj!(g, UndirectedAdjacency(g.adj, vecs...))
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
@inline Base.getindex(g::IsingGraph, idx) = IsingLayer(g, idx)
@inline Base.getindex(g::IsingGraph) = IsingLayer(g, 1)
@inline Base.length(g::IsingGraph) = length(g.layers)
@inline Base.lastindex(g::IsingGraph) = length(g)
Base.view(g::IsingGraph, idx) = view(g.layers, idx)
@inline graphidxs(g::IsingGraph) = Int32(1):Int32(nStates(g))
Base.get!(g::IsingGraph, s, d) = get!(g.addons, s, d)
Base.get(g::IsingGraph, s) = get(g.addons, s)
Base.get(g::IsingGraph, s, d) = get(g.addons, s, d)

@inline layers(g::IsingGraph) = ntuple(i -> IsingLayer(g, i), length(g.layers))

# Don't overload!
@inline getstate(g::IsingGraph) = g.state

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

Processes.context(g::IsingGraph) = context(process(g))
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
    reinit(g)
end

@inline function disableself!(g)
    g.self = deactivate(g.self) # Ensure self is inactive
    reinit(g)
end
@inline function homogeneousself!(g, val = 1)
    g.self = sethomogeneoustensor(g.self, val) # Set self to zero
    reinit(g)
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
# # (;self = ParamTensor(zeros(eltype(state), length(state)), 0, "Self Connections", false)),
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