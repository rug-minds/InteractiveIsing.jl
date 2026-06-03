export VectorExchange, VectorField, VectorMagnitudePenalty, VectorSpin

"""
    VectorExchange(; J = nothing, adj = nothing)

Exchange interaction for vector-spin models.

The energy convention matches scalar `Bilinear`:
`E = -1/2 * sum(i,j) J_ij * dot(s_i, s_j)`.
"""
struct VectorExchange{J} <: HamiltonianTerm
    J::J
end

VectorExchange(; J = nothing, adj = nothing) = begin
    isnothing(J) || isnothing(adj) ||
        throw(ArgumentError("Pass either `J` or `adj`, not both."))
    VectorExchange(isnothing(J) ? adj : J)
end

"""
    VectorField(; h = nothing, c = 1)

Local vector-field term `E = -c * sum_i dot(h_i, s_i)`.
"""
struct VectorField{H,C} <: LocalPotential
    h::H
    c::C
end

VectorField(; h = nothing, c = 1) = VectorField(h, c)

"""
    VectorMagnitudePenalty(; c = 1, target = 1)

Local radial penalty `E = c * sum_i (norm(s_i) - target)^2`.

This is useful for bounded-component vector models where exchange should align
vectors without forcing every spin to the largest magnitude allowed by its
component bounds.
"""
struct VectorMagnitudePenalty{C,R} <: LocalPotential
    c::C
    target::R
end

VectorMagnitudePenalty(; c = 1, target = 1) = VectorMagnitudePenalty(c, target)

@inline _vector_spin_zero(::Type{SVector{D,T}}) where {D,T} = zero(SVector{D,T})
@inline _vector_spin_zero(model::G) where {G<:AbstractVectorSpinGraph} = _vector_spin_zero(spin_state_type(model))

@inline function _default_vector_field(model::G) where {G<:AbstractVectorSpinGraph}
    return ConstFill(_vector_spin_zero(model), nstates(model))
end

@inline function _convert_vector_spin(::Type{SVector{D,T}}, value::StaticVector) where {D,T}
    length(value) == D ||
        throw(ArgumentError("Vector field dimension $(length(value)) does not match spin dimension $D"))
    return SVector{D,T}(value)
end

@inline function _convert_vector_spin(::Type{SVector{D,T}}, value::AbstractVector) where {D,T}
    length(value) == D ||
        throw(ArgumentError("Vector field dimension $(length(value)) does not match spin dimension $D"))
    return SVector{D,T}(ntuple(i -> convert(T, value[i]), Val(D)))
end

@inline function _convert_vector_spin(::Type{SVector{D,T}}, value::Tuple) where {D,T}
    length(value) == D ||
        throw(ArgumentError("Vector field dimension $(length(value)) does not match spin dimension $D"))
    return SVector{D,T}(ntuple(i -> convert(T, value[i]), Val(D)))
end

function _instantiate_vector_field_values(h, model::G) where {G<:AbstractVectorSpinGraph}
    S = spin_state_type(model)
    isnothing(h) && return _default_vector_field(model)
    applicable(h, model) && return _instantiate_vector_field_values(h(model), model)

    if h isa StaticVector || h isa Tuple
        return ConstFill(_convert_vector_spin(S, h), nstates(model))
    elseif h isa AbstractMatrix
        size(h, 1) == spin_dimension(model) ||
            throw(ArgumentError("Vector field matrix first dimension $(size(h, 1)) does not match spin dimension $(spin_dimension(model))"))
        size(h, 2) == nstates(model) ||
            throw(ArgumentError("Vector field matrix second dimension $(size(h, 2)) does not match graph state length $(nstates(model))"))
        return [_convert_vector_spin(S, view(h, :, i)) for i in 1:nstates(model)]
    elseif h isa AbstractVector
        length(h) == nstates(model) ||
            throw(ArgumentError("Vector field length $(length(h)) does not match graph state length $(nstates(model))"))
        return [_convert_vector_spin(S, h_i) for h_i in h]
    else
        throw(ArgumentError("Unsupported vector field input $(typeof(h)); pass a vector spin, a vector of vector spins, a D-by-N matrix, or a graph function."))
    end
end

function instantiate(hterm::VectorExchange, model::G) where {G<:AbstractVectorSpinGraph}
    J = isnothing(hterm.J) ? adj(model) : hterm.J
    J isa AbstractMatrix ||
        throw(ArgumentError("VectorExchange coupling `J` must be an AbstractMatrix, got $(typeof(J))."))
    return VectorExchange(J)
end

function instantiate(hterm::VectorField, model::G) where {G<:AbstractVectorSpinGraph}
    h = _instantiate_vector_field_values(hterm.h, model)
    c = convert(eltype(model), hterm.c)
    return VectorField(h, c)
end

function instantiate(hterm::VectorMagnitudePenalty, model::G) where {G<:AbstractVectorSpinGraph}
    c = convert(eltype(model), hterm.c)
    target = convert(eltype(model), hterm.target)
    return VectorMagnitudePenalty(c, target)
end

@inline function _weighted_vector_neighbors_sum(node_idx::I, J::UndirectedAdjacency, spins) where {I<:Integer}
    total = zero(eltype(spins))
    sp = J.sp
    rows = SparseArrays.rowvals(sp)
    colptr = SparseArrays.getcolptr(sp)
    nzvals = SparseArrays.nonzeros(sp)

    @inbounds for ptr in colptr[node_idx]:(colptr[node_idx + 1] - 1)
        row = rows[ptr]
        row == node_idx && continue
        total += nzvals[ptr] * spins[row]
    end
    return total
end

@inline function _weighted_vector_neighbors_sum(node_idx::I, J::SparseMatrixCSC, spins) where {I<:Integer}
    total = zero(eltype(spins))
    rows = SparseArrays.rowvals(J)
    colptr = SparseArrays.getcolptr(J)
    nzvals = SparseArrays.nonzeros(J)

    @inbounds for ptr in colptr[node_idx]:(colptr[node_idx + 1] - 1)
        row = rows[ptr]
        row == node_idx && continue
        total += nzvals[ptr] * spins[row]
    end
    return total
end

@inline function _vector_connection_iterator(J::UndirectedAdjacency)
    sp = J.sp
    rows = SparseArrays.rowvals(sp)
    colptr = SparseArrays.getcolptr(sp)
    nzvals = SparseArrays.nonzeros(sp)
    return (
        (rows[ptr], col, nzvals[ptr])
        for col in 1:size(sp, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
        if rows[ptr] != col
    )
end

@inline function _vector_connection_iterator(J::SparseMatrixCSC)
    rows = SparseArrays.rowvals(J)
    colptr = SparseArrays.getcolptr(J)
    nzvals = SparseArrays.nonzeros(J)
    return (
        (rows[ptr], col, nzvals[ptr])
        for col in 1:size(J, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
        if rows[ptr] != col
    )
end

@inline function calculate(::H, hterm::VectorExchange, model::G) where {G<:AbstractVectorSpinGraph}
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    for (i, j, weight) in _vector_connection_iterator(hterm.J)
        @inbounds total += weight * dot(spins[i], spins[j])
    end
    return -eltype(model)(0.5) * total
end

@inline function calculate(::ΔH, hterm::VectorExchange, model::G, proposal::FlipProposal) where {G<:AbstractVectorSpinGraph}
    spins = @inline graphstate(model)
    total = @inline _weighted_vector_neighbors_sum(at_idx(proposal), hterm.J, spins)
    return dot(total, from_val(proposal) - to_val(proposal))
end

@inline function calculate(::d_iH, hterm::VectorExchange, model::G, s_idx::Integer) where {G<:AbstractVectorSpinGraph}
    spins = @inline graphstate(model)
    total = @inline _weighted_vector_neighbors_sum(s_idx, hterm.J, spins)
    return -total
end

@inline function calculate(::H, hterm::VectorField, model::G) where {G<:AbstractVectorSpinGraph}
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    @inbounds for i in eachindex(spins)
        total += dot(hterm.h[i], spins[i])
    end
    return -hterm.c * total
end

@inline function calculate(::ΔH, hterm::VectorField, model::G, proposal::FlipProposal) where {G<:AbstractVectorSpinGraph}
    return -hterm.c * dot(hterm.h[at_idx(proposal)], to_val(proposal) - from_val(proposal))
end

@inline function calculate(::d_iH, hterm::VectorField, model::G, s_idx::Integer) where {G<:AbstractVectorSpinGraph}
    return -hterm.c * hterm.h[s_idx]
end

@inline function _vector_magnitude_penalty_energy(hterm::VectorMagnitudePenalty, spin)
    radial_delta = norm(spin) - hterm.target
    return hterm.c * radial_delta * radial_delta
end

@inline function calculate(::H, hterm::VectorMagnitudePenalty, model::G) where {G<:AbstractVectorSpinGraph}
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    @inbounds for spin in spins
        total += _vector_magnitude_penalty_energy(hterm, spin)
    end
    return total
end

@inline function calculate(::ΔH, hterm::VectorMagnitudePenalty, model::G, proposal::FlipProposal) where {G<:AbstractVectorSpinGraph}
    return _vector_magnitude_penalty_energy(hterm, to_val(proposal)) -
        _vector_magnitude_penalty_energy(hterm, from_val(proposal))
end

@inline function calculate(::d_iH, hterm::VectorMagnitudePenalty, model::G, s_idx::Integer) where {G<:AbstractVectorSpinGraph}
    spin = @inbounds graphstate(model)[s_idx]
    spin_norm = norm(spin)
    iszero(spin_norm) && return zero(spin)
    return 2 * hterm.c * (one(eltype(model)) - hterm.target / spin_norm) * spin
end

"""
    VectorSpin(; J = nothing, adj = nothing, h = nothing)

Default Hamiltonian for `VectorSpinGraph`.

This combines vector exchange and a local vector field:
`VectorExchange(; J, adj) + VectorField(; h)`.
"""
@inline function VectorSpin(; J = nothing, adj = nothing, h = nothing)
    return HamiltonianTerms(VectorExchange(; J, adj), VectorField(; h))
end

@inline function VectorSpin(g::G; J = nothing, adj = nothing, h = nothing) where {G<:AbstractVectorSpinGraph}
    return instantiate(VectorSpin(; J, adj, h), g)
end
