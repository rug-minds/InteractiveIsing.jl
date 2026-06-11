# AI Generated
export XYLayer, XYGraph

"""
    _xy_periodic_stateset(stateset)

Normalize XY constructor state-set input to a `PeriodicStateSet`.
"""
@inline function _xy_periodic_stateset(stateset::PeriodicStateSet{S}) where {S}
    return stateset
end

"""
    _xy_periodic_stateset(bounds)

Normalize a tuple of angle bounds to a `PeriodicStateSet`.
"""
@inline function _xy_periodic_stateset(bounds::B) where {B<:Tuple}
    return PeriodicStateSet(bounds)
end

"""
    _xy_defaulted_args(args, hamiltonian, proposer)

Append XY defaults unless the caller already supplied a Hamiltonian or proposer.
"""
function _xy_defaulted_args(args::A, hamiltonian::H, proposer::P) where {A<:Tuple,H,P}
    # Keep caller-provided Hamiltonians/proposers authoritative while giving
    # plain XY constructors a cosine interaction and local angular proposals.
    graph_args = any(arg -> arg isa AbstractProposer, args) ? args : (args..., proposer)
    return any(arg -> arg isa Hamiltonian, graph_args) ? graph_args : (graph_args..., hamiltonian)
end

"""
    XYLayer(args...; angle_set = AngleStateSet(), kwargs...)

Construct a continuous layer whose state values wrap around `angle_set`.
"""
function XYLayer(args::Vararg{Any,N}; angle_set = AngleStateSet(), kwargs...) where {N}
    periodic_set = @inline _xy_periodic_stateset(angle_set)
    return Layer(args..., Continuous(), periodic_set; kwargs...)
end

"""
    XYGraph(size1::Int, args...; angle_set = AngleStateSet(), hamiltonian = CosineInteraction(), proposer = LocalProposer(0.1), kwargs...)

Construct a single-layer XY graph by forwarding to `IsingGraph` with a periodic
angle state set, `Continuous()` states, and cosine-interaction defaults.
"""
function XYGraph(
    size1::I,
    args::Vararg{Any,N};
    angle_set = AngleStateSet(),
    hamiltonian = CosineInteraction(),
    proposer = LocalProposer(0.1),
    kwargs...,
) where {I<:Integer,N}
    periodic_set = @inline _xy_periodic_stateset(angle_set)
    graph_args = @inline _xy_defaulted_args(args, hamiltonian, proposer)
    return IsingGraph(size1, graph_args..., Continuous(), periodic_set; kwargs...)
end

"""
    XYGraph(layer::IsingLayerData, args...; hamiltonian = CosineInteraction(), proposer = LocalProposer(0.1), kwargs...)

Construct an XY graph from explicit layers, typically created with `XYLayer`.
"""
function XYGraph(
    layer::L,
    args::Vararg{Any,N};
    hamiltonian = CosineInteraction(),
    proposer = LocalProposer(0.1),
    kwargs...,
) where {L<:IsingLayerData,N}
    graph_args = @inline _xy_defaulted_args(args, hamiltonian, proposer)
    return IsingGraph(layer, graph_args...; kwargs...)
end
