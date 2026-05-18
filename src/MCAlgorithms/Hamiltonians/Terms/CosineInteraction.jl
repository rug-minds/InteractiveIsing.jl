export CosineInteraction

"""
    CosineInteraction(; J = nothing, adj = nothing, phase = nothing,
        edge_phase = nothing, edge_phase_orientation = :upper,
        period = :stateset, turns = 1)

Connected-spin phase interaction

    H = -1/2 * sum_{i,j} J_ij *
        cos((theta_i - phase_i) - (theta_j - phase_j) - A_ij)

where `theta` is derived from each spin's layer `StateSet`. Continuous layers
use affine range normalization. Discrete layers use q-state clock normalization.
"""
struct CosineInteraction{P,I} <: HamiltonianTerm
    parameters::P
    internal::I
end

struct CosineInteractionInternal{T} <: InternalImplementation
    origin::Vector{T}
    scale::Vector{T}
    clock_indices::Vector{Int}
    clock_sets::Vector{Vector{T}}
    edge_phase_orientation::Symbol
end

_default_cosine_adjacency(g) = adj(g)

function _ensure_optional_adjacency(input, default, model)
    isnothing(input) && return nothing
    !(input isa Type) && applicable(input, model) &&
        return _ensure_optional_adjacency(input(model), default, model)

    n = statelen(model)
    if input isa Number
        sp = copy(sparse(adj(model)))
        fill!(nonzeros(sp), convert(eltype(model), input))
        return sp
    elseif input isa AbstractMatrix
        size(input, 1) == n && size(input, 2) == n ||
            throw(DimensionMismatch("Edge phase matrix size must match graph state length; expected $(n)x$(n), got $(size(input))."))
    end

    return input
end

function CosineInteraction(;
    adj = nothing,
    J = nothing,
    phase = nothing,
    edge_phase = nothing,
    edge_phase_orientation = :upper,
    period = :stateset,
    turns = 1,
)
    isnothing(J) || isnothing(adj) ||
        throw(ArgumentError("Pass either `J` or `adj`, not both."))
    J = isnothing(J) ? adj : J
    params = Parameters(
        parameter(;
            J,
            type = AbstractMatrix,
            default = _default_cosine_adjacency,
            ensure = ensure_isinggraph_adjacency,
            info = "Cosine coupling matrix J_ij",
        ),
        parameter(;
            phase,
            type = AbstractArray,
            default = ConstFill(0),
            default_type = UniformArray,
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local phase offsets phi_i in radians",
        ),
        parameter(;
            edge_phase,
            type = Union{Nothing,AbstractMatrix},
            default = nothing,
            ensure = _ensure_optional_adjacency,
            info = "Directed edge phase offsets A_ij in radians",
        ),
    )
    internal = InternalPlan((; period, turns, edge_phase_orientation)) do plan, g
        return _cosine_interaction_internal(g, plan.values)
    end
    return CosineInteraction(params, internal)
end

@inline CosineInteraction(g::AbstractIsingGraph; kwargs...) =
    instantiate(CosineInteraction(; kwargs...), g)

function instantiate(term::CosineInteraction, g::AbstractIsingGraph)
    params = instantiate(parameters(term), g)
    internals = instantiate(internal(term), g)
    instantiated = CosineInteraction(params, internals)
    _validate_cosine_edge_phase(instantiated)
    return instantiated
end

function _cosine_interaction_internal(g::AbstractIsingGraph, config)
    config.period === :stateset ||
        throw(ArgumentError("CosineInteraction currently supports `period = :stateset`; got $(config.period)."))
    config.edge_phase_orientation in (:upper, :antisymmetric, :raw) ||
        throw(ArgumentError("edge_phase_orientation must be one of :upper, :antisymmetric, or :raw; got $(config.edge_phase_orientation)."))

    T = eltype(g)
    origin = zeros(T, statelen(g))
    scale = zeros(T, statelen(g))
    clock_indices = zeros(Int, statelen(g))
    clock_sets = Vector{Vector{T}}()
    turn = T(config.turns)
    twopi = T(2) * T(π)

    for layer in layers(g)
        states = collect(T, stateset(layer))
        isempty(states) && throw(ArgumentError("CosineInteraction requires non-empty StateSet values."))
        step = twopi * turn
        if statetype(layer) isa Continuous
            low = first(states)
            high = last(states)
            isfinite(low) && isfinite(high) && high > low ||
                throw(ArgumentError("CosineInteraction requires finite positive continuous StateSet ranges; got $(stateset(layer))."))
            layer_scale = step / (high - low)
            for idx in graphidxs(layer)
                origin[idx] = low
                scale[idx] = layer_scale
            end
        elseif statetype(layer) isa Discrete
            length(states) > 0 ||
                throw(ArgumentError("CosineInteraction requires at least one discrete state."))
            push!(clock_sets, states)
            clock_idx = length(clock_sets)
            layer_scale = step / length(states)
            for idx in graphidxs(layer)
                scale[idx] = layer_scale
                clock_indices[idx] = clock_idx
            end
        else
            throw(ArgumentError("CosineInteraction supports Continuous and Discrete layers; got $(statetype(layer))."))
        end
    end

    return CosineInteractionInternal(origin, scale, clock_indices, clock_sets, config.edge_phase_orientation)
end

function _validate_cosine_edge_phase(hterm::CosineInteraction)
    isnothing(hterm.edge_phase) && return nothing
    hterm.edge_phase_orientation === :antisymmetric || return nothing

    J = hterm.J
    A = hterm.edge_phase
    for (i, j) in _cosine_index_pairs(J)
        i == j && continue
        isapprox(A[i, j], -A[j, i]; atol = sqrt(eps(float(eltype(J))))) ||
            throw(ArgumentError("edge_phase_orientation = :antisymmetric requires edge_phase[$i, $j] == -edge_phase[$j, $i]."))
    end
    return nothing
end

@inline function _clock_position(states, value)
    @inbounds for pos in eachindex(states)
        states[pos] == value && return pos
    end
    throw(ArgumentError("Discrete spin value $(value) is not in its layer StateSet $(Tuple(states))."))
end

@inline function _cosine_theta(hterm::CosineInteraction, value, idx::Integer)
    clock_idx = @inbounds hterm.clock_indices[idx]
    if clock_idx == 0
        return @inbounds hterm.scale[idx] * (value - hterm.origin[idx])
    else
        states = @inbounds hterm.clock_sets[clock_idx]
        pos = @inline _clock_position(states, value)
        return @inbounds hterm.scale[idx] * (pos - 1)
    end
end

@inline function _cosine_theta_derivative(hterm::CosineInteraction, idx::Integer)
    @inbounds hterm.clock_indices[idx] == 0 ||
        throw(ArgumentError("CosineInteraction d_iH is not defined for discrete clock-normalized spins."))
    return @inbounds hterm.scale[idx]
end

@inline function _site_phase(hterm::CosineInteraction, spins, idx::Integer)
    return (@inline _cosine_theta(hterm, @inbounds(spins[idx]), idx)) - @inbounds(hterm.phase[idx])
end

@inline function _cosine_index_pairs(J::SparseArrays.AbstractSparseMatrix)
    return (
        (SparseArrays.rowvals(J)[ptr], j)
        for j in axes(J, 2)
        for ptr in SparseArrays.nzrange(J, j)
    )
end

@inline function _cosine_index_pairs(J::UndirectedAdjacency)
    return index_pairs_iterator(J, false)
end

@inline function _cosine_index_pairs(J::AbstractMatrix)
    return (
        (i, j)
        for j in axes(J, 2)
        for i in axes(J, 1)
        if !iszero(@inbounds J[i, j])
    )
end

@inline function _cosine_column_indices(J::SparseArrays.AbstractSparseMatrix, j::Integer)
    return (SparseArrays.rowvals(J)[ptr] for ptr in SparseArrays.nzrange(J, Int(j)))
end

@inline function _cosine_column_indices(J::UndirectedAdjacency, j::Integer)
    return _cosine_column_indices(sparse(J), j)
end

@inline function _cosine_column_indices(J::AbstractMatrix, j::Integer)
    return (i for i in axes(J, 1) if !iszero(@inbounds J[i, j]))
end

@inline function _edge_phase(hterm::CosineInteraction, i::Integer, j::Integer)
    A = hterm.edge_phase
    isnothing(A) && return zero(eltype(hterm.phase))
    orientation = hterm.edge_phase_orientation
    if orientation === :upper
        i == j && return zero(eltype(A))
        return i < j ? @inbounds(A[i, j]) : -@inbounds(A[j, i])
    else
        return @inbounds A[i, j]
    end
end

@inline function _directed_cosine_energy(hterm::CosineInteraction, psi_i, psi_j, i::Integer, j::Integer)
    return -oftype(psi_i, 0.5) * hterm.J[i, j] *
           cos(psi_i - psi_j - (@inline _edge_phase(hterm, i, j)))
end

@inline function calculate(::H, hterm::CosineInteraction, model::S) where {S<:AbstractIsingGraph}
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    for (i, j) in _cosine_index_pairs(hterm.J)
        i == j && continue
        psi_i = @inline _site_phase(hterm, spins, i)
        psi_j = @inline _site_phase(hterm, spins, j)
        total += @inline _directed_cosine_energy(hterm, psi_i, psi_j, i, j)
    end
    return total
end

@inline function calculate(::ΔH, hterm::CosineInteraction, model::S, proposal) where {S<:AbstractIsingGraph}
    spins = @inline graphstate(model)
    J = hterm.J
    j = at_idx(proposal)
    old_psi_j = (@inline _cosine_theta(hterm, @inbounds(spins[j]), j)) - @inbounds(hterm.phase[j])
    new_psi_j = (@inline _cosine_theta(hterm, to_val(proposal), j)) - @inbounds(hterm.phase[j])
    total = zero(eltype(model))

    for i in _cosine_column_indices(J, j)
        i == j && continue
        psi_i = @inline _site_phase(hterm, spins, i)
        old_energy =
            (@inline _directed_cosine_energy(hterm, old_psi_j, psi_i, j, i)) +
            (@inline _directed_cosine_energy(hterm, psi_i, old_psi_j, i, j))
        new_energy =
            (@inline _directed_cosine_energy(hterm, new_psi_j, psi_i, j, i)) +
            (@inline _directed_cosine_energy(hterm, psi_i, new_psi_j, i, j))
        total += new_energy - old_energy
    end
    return total
end

@inline function calculate(::d_iH, hterm::CosineInteraction, model::S, s_idx) where {S<:AbstractIsingGraph}
    spins = @inline graphstate(model)
    J = hterm.J
    j = Int(s_idx)
    psi_j = @inline _site_phase(hterm, spins, j)
    derivative = zero(eltype(model))

    for i in _cosine_column_indices(J, j)
        i == j && continue
        psi_i = @inline _site_phase(hterm, spins, i)
        derivative +=
            oftype(psi_j, 0.5) * J[j, i] *
            sin(psi_j - psi_i - (@inline _edge_phase(hterm, j, i)))
        derivative -=
            oftype(psi_j, 0.5) * J[i, j] *
            sin(psi_i - psi_j - (@inline _edge_phase(hterm, i, j)))
    end

    return derivative * (@inline _cosine_theta_derivative(hterm, j))
end

function _cosine_derivative_buffer(J)
    return similar(sparse(J))
end

function _cosine_derivative_buffer(J::AbstractArray)
    return similar(J)
end

@inline function parameter_derivative(
    hterm::CosineInteraction,
    state::AbstractVector;
    dJ = _cosine_derivative_buffer(hterm.J),
    buffermode::BufferMode = OverwriteBuffer(),
)
    if buffermode isa OverwriteBuffer
        fill!(dJ, zero(eltype(dJ)))
    end

    for (i, j) in _cosine_index_pairs(hterm.J)
        i == j && continue
        psi_i = (@inline _cosine_theta(hterm, @inbounds(state[i]), i)) - @inbounds(hterm.phase[i])
        psi_j = (@inline _cosine_theta(hterm, @inbounds(state[j]), j)) - @inbounds(hterm.phase[j])
        val = -oftype(psi_i, 0.5) * cos(psi_i - psi_j - (@inline _edge_phase(hterm, i, j)))
        if buffermode isa OverwriteBuffer
            dJ[i, j] = val
        else
            dJ[i, j] += sign(buffermode) * val
        end
    end
    return (; dJ)
end
