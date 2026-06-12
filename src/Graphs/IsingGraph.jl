export adj, state
export statelen, initRandomState, initState
export processes, process
# Layer forwards
export layer_idxs
export nstates, nspins, nnodes
# export continuous


#TODO: SHouldn't have the same supertype as layer statetype
struct ContinuousState <: StateType end
struct DiscreteState <: StateType end
struct MixedState <: StateType end

intprecision(::Type{Float16}) = Int16
intprecision(::Type{Float32}) = Int32
intprecision(::Type{Float64}) = Int64

# Ising Graph Representation and functions
mutable struct IsingGraph{T <: AbstractFloat, M, Layers, A, N, I} <: AbstractIsingGraph{T}
    # Vertices and edges
    state::A
    # Adjacency Matrix
    adj::M
    # self::AbstractArray{T,1} # Diagonal of adj stored as a separate array for efficiency
    
    temp::T

    proposer::AbstractProposer
    default_algorithm::ProcessAlgorithm
    hamiltonian::Hamiltonian
    
    # Connection between layers, Could be useful to track for faster removing of layers
    # layerconns::Dict{Set, Int32}

    index_set::I
    # d::GraphData{T} #Other stuff. Maybe just make this a dict?
    addons::Dict{Symbol, Any}

    layers::Layers

    function IsingGraph(
        state,
        adj,
        temp,
        proposer,
        default_algorithm,
        hamiltonian,
        index_set,
        addons,
        layers
    )
        new{eltype(state), typeof(adj), typeof(layers), typeof(state), length(layers), typeof(index_set)}(
            state,
            adj,
            temp,
            proposer,
            default_algorithm,
            hamiltonian,
            index_set,
            addons,
            layers
        )
    end
end


Base.eltype(::IsingGraph{T}) where T = T
Base.eltype(::Type{IsingGraph{T}}) where T = T
Base.eltype(::Type{<:IsingGraph{T}}) where T = T
Base.eltype(::AbstractIsingGraph{T}) where T = T
Base.eltype(::Type{<:AbstractIsingGraph{T}}) where T = T

#extend show to print out the graph, showing the length of the state, and the layers
function Base.show(io::IO, g::IsingGraph)
    for (idx, layer) in enumerate(layers(g))
        Base.show(io, layer)
        if idx != length(g)
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

@inline clamprange!(g::IsingGraph, clamp, idxs) = setrange!(index_set(g), clamp, idxs)
export clamprange!

@Setter!Getter IsingGraph adj params layers temp
# @inline params(g::IsingGraph) = g.params
export params

@inline temp(g::G) where {G<:IsingGraph} = getproperty(g, :temp)

"""
    temp!(g, value)

Set graph temperature. Plain real values are stored directly in graph-internal
units. Unitful values dispatch to the physical conversion path and are stored
as plain graph-precision numbers.
"""
@inline function temp!(g::G, value::T) where {G<:IsingGraph,T<:Real}
    setproperty!(g, :temp, convert(eltype(g), value))
end

@inline function temp!(g::G, value::T) where {G<:IsingGraph,T<:Unitful.AbstractQuantity}
    scales = physicalscales(g)
    converted = internalvalue(
        value,
        _temperature_unit_spec(value, scales),
        scales,
        g;
        parameter = :temperature,
    )
    converted isa Real ||
        throw(ArgumentError("Graph temperature must convert to Real; got $(typeof(converted))."))
    setproperty!(g, :temp, convert(eltype(g), converted))
end

function temp!(g::G, value::T) where {G<:IsingGraph,T}
    throw(ArgumentError("Graph temperature must be Real or Unitful.AbstractQuantity; got $(T)."))
end

export temp, temp!
@inline nStates(g::IsingGraph) = length(state(g))::Int

@inline nstates(g) = length(state(g))::Int
@inline nspins(g::IsingGraph) = length(state(g))::Int
@inline nnodes(g::IsingGraph) = length(state(g))::Int
export nstates

@inline adj(g::IsingGraph) = getfield(g, :adj)
@inline function adj(g::IsingGraph, newadj)
    @assert size(newadj, 1) == length(state(g)) && size(newadj, 2) == length(state(g))
    g.adj = newadj
    # Add callbacks field to graph, which is a Dict{typeof(<:Function), Vector{Function}}
    # And create a setterGetter macro that includes the callbacks
    reinit(g)
    return newadj
end
@inline adj!(g::IsingGraph, newadj) = adj(g, newadj)

"""
    instantiate_adjacency_like(previous_adj, vecs)

Rebuild generated sparse triplets using the same adjacency storage family as
`previous_adj`.
"""
@inline function instantiate_adjacency_like(previous_adj::UA, vecs::Tuple) where {UA <: UndirectedAdjacency}
    return UndirectedAdjacency(previous_adj, vecs...)
end

function instantiate_adjacency_like(previous_adj::SP, vecs::Tuple) where {SP <: AbstractSparseMatrix}
    rows, cols, vals = vecs
    return instantiate_adjacency_from_triplets(typeof(previous_adj), rows, cols, vals, size(previous_adj, 1))
end

set_adj!(g::IsingGraph, vecs::Tuple) = adj(g, instantiate_adjacency_like(adj(g), vecs))

# @forwardfields IsingGraph GraphData d
hasDefects(::AbstractRange{<:Integer}) = false
hasDefects(::AbstractVector{<:Integer}) = false
hasDefects(::AbstractSet{<:Integer}) = false

"""
The indices that may be sampled from
"""
sampling_indices(idxs::AbstractRange{<:Integer}) = idxs
sampling_indices(idxs::AbstractVector{<:Integer}) = idxs
sampling_indices(idxs::AbstractSet{<:Integer}) = idxs
sampling_indices(gd::GraphDefectsNew) = aliveindices(gd)

@inline sampling_indices(g::IsingGraph) = sampling_indices(index_set(g))
@inline consume_changed!(g::IsingGraph) = consume_changed!(index_set(g))

hasDefects(g::IsingGraph) = hasDefects(index_set(g))
aliveList(g::IsingGraph) = aliveList(index_set(g))
defectList(g::IsingGraph) = defectList(index_set(g))
layerdefects(g::IsingGraph) = layerdefects(index_set(g))
isDefect(g::IsingGraph) = isDefect(index_set(g))
export hasDefects, aliveList, defectList, layerdefects, isDefect

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

@inline layer_idxs(g::IsingGraph) = range.(layers(g))
@inline graphstate(g::IsingGraph) = getfield(g, :state)
@inline layer(g::IsingGraph, idx) = getfield(g, :layers)[idx]
@inline Base.getindex(g::IsingGraph, idx) = IsingLayer(g, idx)
@inline Base.getindex(g::IsingGraph) = IsingLayer(g, 1)
@inline Base.length(g::IsingGraph) = length(getfield(g, :layers))
@inline Base.lastindex(g::IsingGraph) = length(g)
Base.view(g::IsingGraph, idx) = view(getfield(g, :layers), idx)
@inline graphidxs(g::IsingGraph) = Int32(1):Int32(nStates(g))
Base.get!(g::IsingGraph, s, d) = get!(addons(g), s, d)
Base.get(g::IsingGraph, s) = get(addons(g), s)
Base.get(g::IsingGraph, s, d) = get(addons(g), s, d)

"""
    InteractiveGraphVarSpec(target, varname; value = nothing, range = nothing, label = string(varname))

Graph addon specification for one interactively controlled process variable.

`target` is the lifecycle target passed to `StatefulAlgorithms.Interactive` and
`StatefulAlgorithms.Override`, for example `LocalLangevin` or a concrete keyed
algorithm instance. `value` is the persistent graph-side value used to seed new
processes. `range` configures the UI slider when present; otherwise the slider
panel infers a range heuristically from the current value.
"""
struct InteractiveGraphVarSpec{Target,ValueType,RangeType,LabelType}
    target::Target
    varname::Symbol
    value::ValueType
    range::RangeType
    label::LabelType
end

function InteractiveGraphVarSpec(
    target,
    varname::Symbol;
    value = nothing,
    range = nothing,
    label = string(varname),
)
    isnothing(value) || value isa Real ||
        throw(ArgumentError("Interactive graph variable `$(varname)` must have a Real `value` or `nothing`, got $(typeof(value))."))
    return InteractiveGraphVarSpec{typeof(target),typeof(value),typeof(range),typeof(label)}(
        target,
        varname,
        value,
        range,
        label,
    )
end

"""
    interactivevars(g::IsingGraph)

Return the graph-level interactive variable specs stored in `g.addons`.
"""
@inline interactivevars(g::G) where {G<:IsingGraph} = get(g, :interactive_vars, ())

@inline function _replace_interactive_graph_var_spec(spec::InteractiveGraphVarSpec, value)
    return InteractiveGraphVarSpec(spec.target, spec.varname; value, range = spec.range, label = spec.label)
end

"""
    interactivevar!(g, target, varname; value = nothing, range = nothing, label = string(varname))

Register or replace one interactively controlled process variable on `g`.
Stored values persist on the graph addon and seed future `createProcess` calls
through `StatefulAlgorithms.Override`.
"""
function interactivevar!(
    g::G,
    target,
    varname::Symbol;
    value = nothing,
    range = nothing,
    label = string(varname),
) where {G<:IsingGraph}
    spec = InteractiveGraphVarSpec(target, varname; value, range, label)
    specs = interactivevars(g)
    updated = ()
    replaced = false
    for oldspec in specs
        if isequal(oldspec.target, spec.target) && oldspec.varname === varname
            updated = (updated..., spec)
            replaced = true
        else
            updated = (updated..., oldspec)
        end
    end
    replaced || (updated = (updated..., spec))
    g.addons[:interactive_vars] = updated
    return spec
end

@inline function _set_interactive_graph_var_value!(g::G, target, varname::Symbol, value) where {G<:IsingGraph}
    specs = interactivevars(g)
    updated = ()
    replaced = false
    for spec in specs
        if isequal(spec.target, target) && spec.varname === varname
            updated = (updated..., _replace_interactive_graph_var_spec(spec, value))
            replaced = true
        else
            updated = (updated..., spec)
        end
    end
    replaced || error("Interactive graph variable $(varname) for target $(target) is not registered on this graph.")
    g.addons[:interactive_vars] = updated
    return value
end
export InteractiveGraphVarSpec, interactivevars, interactivevar!

@inline layers(g::IsingGraph) = ntuple(i -> IsingLayer(g, i), length(g))

# Don't overload!
@inline getstate(g::IsingGraph) = state(g)

function Base.convert(::Type{<:AbstractIsingLayer}, g::IsingGraph)
    @assert length(g) == 1 "Graph has more than one layer, ambiguous"
    return layer(g, 1)
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

StatefulAlgorithms.context(g::IsingGraph) = context(process(g))
export process


#TODO: Give new idx
@inline function layerIdx!(g, oldidx, newidx)
    shuffle!(g.layers, oldidx, newidx)
end
export layerIdx!

IsingGraph(g::IsingGraph) = deepcopy(g)

@inline Base.size(g::IsingGraph)::Tuple{Int32,Int32} = (nStates(g), 1)

function reset!(g::IsingGraph)
    state(g) .= initRandomState(g)
end
 
statelen(g::IsingGraph) = length(state(g))

setdefect(g::IsingGraph, val, idx) = index_set(g)[idx] = val
