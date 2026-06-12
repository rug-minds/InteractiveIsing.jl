export Clamping

"""
Clamping Hamiltonian for Equilibrium Propagation
H = β/2 * mask_i * (s_i - y_i)^2

Where y_i is the target value for the i-th node and mask_i controls whether
that node is clamped. A zero target is therefore no longer the same thing as
"not clamped"; use mask_i = 0 for unconstrained nodes.
"""
struct Clamping{P} <: HamiltonianTerm
    parameters::P
end

@inline function Clamping(; β = 1f0, y = nothing, mask = nothing)
    params = Parameters(
        parameter(;
            β,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Clamping strength β",
            units = physicalunits(energy = 1, role = :clamping_energy),
        ),
        parameter(;
            y,
            type = AbstractVector,
            default = g -> fill(zero(eltype(g)), statelen(g)),
            default_type = Vector,
            ensure = ensure_isinggraph_state_vector,
            info = "Target state y_i",
            units = physicalunits(state = 1, role = :state),
        ),
        parameter(;
            mask,
            type = AbstractVector,
            default = g -> fill(one(eltype(g)), statelen(g)),
            default_type = Vector,
            ensure = ensure_isinggraph_state_vector,
            info = "Clamping mask m_i; zero entries are unconstrained",
            units = physicalunits(role = :dimensionless),
        ),
    )
    return Clamping(params)
end

@inline Clamping(β::Real, y = nothing, mask = nothing) = Clamping(; β, y, mask)
@inline Clamping(β::AbstractArray, y = nothing, mask = nothing) = Clamping(; β, y, mask)
@inline Clamping(β::NoEnsure, y = nothing, mask = nothing) = Clamping(; β, y, mask)
@inline Clamping(β::Force, y = nothing, mask = nothing) = Clamping(; β, y, mask)

@inline function Clamping(g::AbstractIsingGraph, β = one(eltype(g)), y = nothing, mask = nothing)
    return instantiate(Clamping(β, y, mask), g)
end

params(::Type{Clamping}, GraphType) = GatherHamiltonianParams(
    (:β, GraphType, GraphType(0), "Clamping Factor"),
    (:y, Vector{GraphType}, GraphType(0), "Targets"),
    (:mask, Vector{GraphType}, GraphType(1), "Clamping mask"),
)

@inline function calculate(::H, hterm::Clamping, model)
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    @inbounds for i in eachindex(spins)
        δ = spins[i] - hterm.y[i]
        total += hterm.mask[i] * δ * δ
    end
    return hterm.β[] / 2 * total
end

@inline function calculate(::H_i, hterm::Clamping, model, idx)
    spins = @inline graphstate(model)
    δ = spins[idx] - hterm.y[idx]
    return hterm.β[] * hterm.mask[idx] / 2 * δ * δ
end


# function ΔH(::Clamping, hargs, proposal)
@inline function calculate(::ΔH, hterm::Clamping, model, proposal)
    j = at_idx(proposal)
    iszero(hterm.mask[j]) && return zero(eltype(model))
    newstate = to_val(proposal)
    spins = @inline graphstate(model)
    return hterm.β[] * hterm.mask[j] / 2 * (newstate^2 - spins[j]^2 - 2 * hterm.y[j] * (newstate - spins[j]))
end

@inline function calculate(::d_iH, hterm::Clamping, model, proposal::SingleSpinProposal)
    spins = @inline graphstate(model)
    s_idx = @inline at_idx(proposal)
    state = @inline proposed_value(spins, proposal)
    return hterm.β[] * hterm.mask[s_idx] * (state - hterm.y[s_idx])
end

function clamp!(c::Clamping, layer::AbstractIsingLayer, vals::V) where {V <: AbstractVector}
    @assert length(vals) == length(state(layer)) "Length of vals must match number of states in layer"
    idxs = layerrange(layer)
    c.y[idxs] .= vals
    c.mask[idxs] .= one(eltype(c.mask))
    return
end
