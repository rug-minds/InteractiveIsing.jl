#TODO: SHouldn't have the same supertype as layer statetype
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end

# TODO::FINISH
struct Emitter
    obs::Vector{Observable}
end
Observables.notify(emitter::Emitter) = notify.(emitter.obs)
Observables.notify(g::IsingGraph) = notify(g.emitter)

intprecision(::Type{Float16}) = Int16
intprecision(::Type{Float32}) = Int32
intprecision(::Type{Float64}) = Int64


getIntType(::Float64) = Int64
getFloatType(::Float64) = Float64
isarchitecturetype(::Any) = false
isarchitecturetype(t::Tuple{A,B,C}) where {A,B,C} = (A<:Integer && B<:Integer && t[3]<:StateType)
isarchitecturetype(t::Tuple{A,B,C,D}) where {A,B,C,D} = (A<:Integer && B<:Integer && D<:Integer && t[4]<:StateType)

# Ising Graph Representation and functions
mutable struct IsingGraph{T <: AbstractFloat, M <: AbstractMatrix{T}, Layers} <: AbstractIsingGraph{T}
    # Simulation
    sim::Union{Nothing, IsingSim}
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
    emitter::Emitter

    defects::GraphDefects
    # d::GraphData{T} #Other stuff. Maybe just make this a dict?
    addons::Dict{Symbol, Any}

    layers::Layers
end

# Default Initializer for IsingGraph
function IsingGraph(dims...; sim = nothing,  periodic = nothing, sets = nothing, weights::Union{Nothing,WeightGenerator} = nothing, type = Continuous, weighted = true, precision = Float32, kwargs...)
    # architecture = searchkey(kwargs, :architecture, fallback = LProps(dims...; stype = type, stateset = sets, periodic))
    if isnothing(sets) || length(sets) == 1
        length(sets) == 1 && (sets = sets[1])
        ls = (LProps(dims...;stype = type, stateset = sets, periodic),)
    end
    @assert (isnothing(glength) && isnothing(gwidth) && isnothing(architecture)) || (!isnothing(glength) && !isnothing(gwidth)) || !isnothing(architecture) "Either give length and width or architecture"

    # @assert (isempty(dims) && !isnothing(architecture)) || (!isempty(dims) && isnothing(architecture)) "Either give dims or architecture, not both"

    # layers = makelayers(g, architecture)
    # # Create the architecture
    # if isnothing(architecture) && !isnothing(glength) && !isnothing(gwidth)
    #     architecture = [(glength, gwidth, gheight, type)]
    # else
    #     architecture = decode_architecture(architecture)
    # end

    # sets = decode_statesets(sets, length(architecture), precision)

    self = ParamVal(precision[], 0, "Self Connections", false)

    # datalen = arch_to_datalen(architecture)
    datalen = reduce(*, nStates.(ls))
    # startidxs = arch_to_startidxs(architecture)
    # println("Arc: ", architecture)

    # ls = tuple((
    #     IsingLayer(architecture[x][end], nothing, x, startidxs[x], architecture[x][1:end-1]..., set = sets[x]; precision, periodic, adjtype = SparseMatrixCSC{precision,Int32})
    #     for x in 1:length(architecture))...)

    # println("Layers: ", ls)
    # println("Layertype : ", typeof(ls))
    g = IsingGraph{precision, SparseMatrixCSC{precision,Int32}, typeof(ls)}(
        sim,
        zeros(precision, datalen),
        SparseMatrixCSC{precision,intprecision(precision)}(undef,datalen,datalen),
        self,
        #Temp            
        1f0,
        # Default algorithm
        LayeredMetropolis(),
        #Hamiltonians
        Ising(precision, datalen),
        #Layers
        Dict{Pair, Int32}(),
        #Params
        # Parameters(self = ParamVal(precision[], 0, "Self Connections", false)),
        #Emitter
        Emitter(Observable[]),
        #Defects
        GraphDefects(nothing),
        Dict{Symbol, Any}(),
        ls
    )

    g.defects.graph = g
    for layer in g.layers
        layer.graph = g
    end
    

    # Couple the shufflevec and the defects
    # internalcouple!(g.layers, g.defects, (layer) -> Int32(0), push = addLayer!, insert = (obj, idx, item) -> addLayer!(obj, item), deleteat = removeLayer!)

    # println("Graph architecture: ", architecture)
    # println("State sets: ", sets)
    # if !isnothing(architecture)
    #     for (arc_idx,arc) in enumerate(architecture)
    #         height = arc[3] isa Real ? arc[3] : nothing
    #         _addLayer!(g, arc[1], arc[2], height; weights, periodic, type = arc[end], set = sets[arc_idx], kwargs...)
    #     end
    # end
    initRandomState(g)
    cb = x -> layerIdx(sim(x))
    set_listener_callback!(g, cb)
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
@inline function spinidx2layer(g::IsingGraph, idx)::IsingLayer
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

function set_listener_callback!(g, f::Function)
    if !haskey(g.addons, :listener_callback)
        g.addons[:listener_callback] = Function[f]
    else
        push!(g.addons[:listener_callback], f)
    end
end

function listener_callback(g)
    for f in get(g.addons, :listener_callback, Function[])
        f(g)
    end
end


function Base.convert(::Type{<:IsingLayer}, g::IsingGraph)
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
        println("Initializing random state: ", layer)
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

# """
# Resize Graph to new size with random states and no connections for the new states
# Need to be bigger than original size?
# """
# function Base.resize!(g::IsingGraph{T}, newlength, startidx = length(state(g))) where T
#     oldlength = nStates(g)
#     sizediff = newlength - oldlength

#     #TODO RESIZE GPARAMS

#     if sizediff > 0
#         # randomstate = rand(T, sizediff)
#         idxs_to_add = startidx:(startidx + sizediff - 1)

#         # Resize state
#         # splice!(state(g), startidx:startidx-1, zeros(T, sizediff))
#         resize!(state(g), newlength)
#         # Fill copy state to the new indices


#         # Resize adjacency
#         g.adj = insertrowcol(adj(g), idxs_to_add)
#         # Resize self
#         splice!(g.self, startidx:startidx-1, zeros(T, sizediff))
#     else # if making smaller
#         idxs_to_remove = startidx:(startidx + abs(sizediff) - 1)
#         deleteat!(state(g), idxs_to_remove)
#         g.adj = deleterowcol(adj(g), idxs_to_remove)
#     end
    
#     return g
# end



#export resize!

# export addLayer!
# nlayers(::Nothing) = Observable(0)
# function addLayer!(g::IsingGraph, llength, lwidth, lheight = nothing; weights = nothing, periodic = true, type = default_ltype(g), set = convert.(eltype(g),(-1,1)), rangebegin = set[1], rangeend = set[2], kwargs...)
#     newlayer = nothing
#     Processes.@tryLockPause g begin 
#         newlayer = _addLayer!(g, llength, lwidth, lheight; set, weights, periodic, type, kwargs...)
#         # Update the layer idxs
#         nlayers(sim(g))[] += 1
#     end
#     return newlayer
# end

# # addLayer!(g::IsingGraph, llength, lwidth, wg; kwargs...) = addLayer!(g, llength, lwidth; weights = wg, kwargs...)

# function addLayer!(g, dims::Vector, wgs...; kwargs...)
#     for (dimidx,dim) in enumerate(dims)
#         addLayer!(g, dim[1], dim[2], wgs[dimidx]; kwargs...)
#     end
#     return layers(g)
# end

# """
# Add a layer to graph g.
# addLayer(g::IsingGraph, length, width)

# Give keyword argument weightfunc to set a weightfunc.
# If weightfunc = :Default, uses default weightfunc for the Ising Model

# When layer needs to be inserted, layers are shifted around
# This is handled by the relocate! function automatically in the shufflevec
# Because the shufflevec knows then internal data is being pushed around
# Not sure if this is the most transparent way to do it since resizing is not done within the shufflevec
# """
# function _addLayer!(g::IsingGraph{T}, llength, lwidth, lheight = nothing; weights = nothing, periodic = true, type = nothing, kwargs...) where T
#     if isnothing(type)
#         type = default_ltype(g)
#     end
#     # Look if a stateset is given, otherwise give the default and convert to the graph type  
#     set = T.(searchkey(kwargs, :set, fallback = convert.(eltype(g),(-1,1))))
   
#     # Function that makes the new layer based on the insertidx
#     # Found by the shufflevec
#     # TODO: Maybe I should make an insert function for the layers?
#     make_newlayer(idx) = begin
#         _layers = unshuffled(layers(g))

#         extra_states = llength*lwidth
#         if !isnothing(lheight) && lheight isa Real
#             extra_states *= lheight
#         end
#         # Resize the old state

#         # Find the startidx of the new layer
#         # Based on the insertidx found by the shufflevec
#         if !isempty(_layers) && idx > 1
#             _startidx = endidx(_layers[idx-1]) + 1
#         else
#             _startidx = 1
#         end

#         resize!(g, nStates(g) + extra_states, _startidx)


#         return IsingLayer(type, g, idx , _startidx, llength, lwidth, lheight; periodic, set)
#     end
    
#     layertype =  IsingLayer{type, set}
#     push!(layers(g), make_newlayer, layertype)
#     newlayer = layers(g)[end]
#     # g.layers.data = remake_type.(g.layers.data)

#     # Generate the adjacency matrix from the weightfunc
#     if !isnothing(weights)
#         genAdj!(newlayer, weights)
#     elseif weights == :Default
#         println("No weightgenerator given, using default")
#         genAdj!(newlayer, wg_isingdefault)
#     end

#     # SET COORDS
#     setcoords!(g[end], z = length(g)-1)

#     # Init the state
#     initstate!(newlayer)

#     return newlayer
# end

# function _removeLayer!(g::IsingGraph, lidx::Integer)
#     #if only one layer error
#     if length(layers(g)) <= 1
#         error("Cannot remove last layer")
#     end

#     # Remove the layer from the graph
#     layervec = layers(g)
#     layer = layervec[lidx]

#     # Remove the layer from the graph
#     deleteat!(layervec, lidx)

#     resize!(g, nStates(g) - nStates(layer), start(layer))

#     return layers(g)
# end

# function removeLayer!(g::IsingGraph, lidx::Integer)
#     @tryLockPause sim(g) begin 
#         _removeLayer!(g, lidx) 
#          # If the slected layer is after the layer to be removed, decrement layerIdx
#         if layerIdx(sim(g))[] >= lidx && layerIdx(sim(g))[] > 1
#             layerIdx(sim(g))[] -= 1
#         else
#             notify(layerIdx(sim(g)))
#         end
#         nlayers(sim(g))[] -= 1 
#     end
#     return layers(g)
# end
# removeLayer!(layer::IsingLayer) = removeLayer!(graph(layer), layer)
# export removeLayer!

# function removeLayer!(g, idxs::Vector{Int}) 
#     _layers = layers(g)
#     # Sort by internal storage order from last to first, this causes minimal relocations
#     sort!(idxs, lt = (x,y) -> internalidx(_layers, x) > internalidx(_layers, y))
#     @tryLockPause sim(g) for idx in idxs
#         _removeLayer!(g, idx)
#         nlayers(sim(g))[] -= 1
#     end
#     return layers(g)
# end
# removeLayer!(g::IsingGraph, layer::IsingLayer) = removeLayer!(g, layeridx(layer))

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
    g.self = sethomogenousval(g.self, val) # Set self to zero
    refresh(g)
end
export activateself!, disableself!, homogeneousself!


#### SAVE

# Constructor for copying from other graph or savedata.
function IsingGraph(
    state,
    adj,
    layers,
    defects,
    data,
    Hamiltonians = Ising(g)
    )
return IsingGraph(
# Sim
nothing,
#state
state,
# Adjacency
adj,            
#Temp
1f0,
# Default algorithm
updateMetropolis,
#Hamiltonians
Hamiltonians,
# Layers
layers,
# Connections between layers
Dict{Pair, Int32}(),
#params
# (;self = ParamVal(zeros(eltype(state), length(state)), 0, "Self Connections", false)),
# For notifying simulations or other things
Emitter(Observable[]),
# Defects
defects,
# Data
data,
# Processes
Vector{Process}()
)
end