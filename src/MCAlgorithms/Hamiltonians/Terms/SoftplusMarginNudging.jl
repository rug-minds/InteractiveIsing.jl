# AI generated.

export SoftplusMarginNudging

raw"""
    SoftplusMarginNudging(; β = 1f0, y = nothing, mask = nothing, τ = 0.25f0)

Hamiltonian term for smooth margin nudging of clipped continuous Ising output
spins:

```math
H_{nudge}(s, y) =
\frac{β}{2}\sum_i m_i
\left[
    τ \log\left(1 + \exp\left(\frac{1 - y_i s_i}{τ}\right)\right)
\right]^2.
```

The target values `y_i` are expected to be bipolar (`-1` or `+1`) on nudged
output spins. `mask_i = 0` leaves a spin unconstrained, matching the direct
`Clamping` target/mask convention used by MNIST learning code.
"""
struct SoftplusMarginNudging{P} <: HamiltonianTerm
    parameters::P
end

@inline function SoftplusMarginNudging(; β = 1f0, y = nothing, mask = nothing, τ = 0.25f0)
    params = Parameters(
        parameter(;
            β,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Softplus margin nudging strength β",
        ),
        parameter(;
            y,
            type = AbstractVector,
            default = g -> fill(zero(eltype(g)), statelen(g)),
            default_type = Vector,
            ensure = ensure_isinggraph_state_vector,
            info = "Bipolar target state y_i",
        ),
        parameter(;
            mask,
            type = AbstractVector,
            default = g -> fill(one(eltype(g)), statelen(g)),
            default_type = Vector,
            ensure = ensure_isinggraph_state_vector,
            info = "Nudging mask m_i; zero entries are unconstrained",
        ),
        parameter(;
            τ,
            type = AbstractArray,
            default = ConstVal(0.25f0),
            ensure = ensure_isinggraph_scalar,
            info = "Softplus smoothing temperature τ",
        ),
    )
    return SoftplusMarginNudging(params)
end

@inline SoftplusMarginNudging(β::Real, y = nothing, mask = nothing, τ = 0.25f0) =
    SoftplusMarginNudging(; β, y, mask, τ)
@inline SoftplusMarginNudging(β::AbstractArray, y = nothing, mask = nothing, τ = 0.25f0) =
    SoftplusMarginNudging(; β, y, mask, τ)
@inline SoftplusMarginNudging(β::NoEnsure, y = nothing, mask = nothing, τ = 0.25f0) =
    SoftplusMarginNudging(; β, y, mask, τ)
@inline SoftplusMarginNudging(β::Force, y = nothing, mask = nothing, τ = 0.25f0) =
    SoftplusMarginNudging(; β, y, mask, τ)

@inline function SoftplusMarginNudging(
    g::AbstractIsingGraph,
    β = one(eltype(g)),
    y = nothing,
    mask = nothing,
    τ = eltype(g)(0.25),
)
    return instantiate(SoftplusMarginNudging(β, y, mask, τ), g)
end

params(::Type{SoftplusMarginNudging}, GraphType) = GatherHamiltonianParams(
    (:β, GraphType, GraphType(0), "Softplus margin nudging factor"),
    (:y, Vector{GraphType}, GraphType(0), "Bipolar targets"),
    (:mask, Vector{GraphType}, GraphType(1), "Nudging mask"),
    (:τ, GraphType, GraphType(0.25), "Softplus smoothing temperature"),
)

"""
    softplus_margin_residual(z, τ)

Return `τ * log(1 + exp(z))` using branches that avoid overflow for large
positive or negative margins.
"""
@inline function softplus_margin_residual(z::T, τ::T) where {T<:Real}
    if z > T(18)
        return τ * z
    elseif z < -T(18)
        return τ * exp(z)
    end
    return τ * log1p(exp(z))
end

"""
    softplus_margin_sigmoid(z)

Return the logistic factor used by the margin derivative, with stable branches
for large `|z|`.
"""
@inline function softplus_margin_sigmoid(z::T) where {T<:Real}
    if z >= zero(T)
        ez = exp(-z)
        return one(T) / (one(T) + ez)
    end
    ez = exp(z)
    return ez / (one(T) + ez)
end

"""
    softplus_margin_energy(β, y, mask, τ, state)

Return the masked smooth squared-hinge nudging energy contribution for one spin.
"""
@inline function softplus_margin_energy(β::T, y::T, mask::T, τ::T, state::T) where {T<:Real}
    iszero(mask) && return zero(T)
    τ > zero(T) || throw(ArgumentError("SoftplusMarginNudging τ must be positive"))
    z = (one(T) - y * state) / τ
    r = softplus_margin_residual(z, τ)
    return β * mask * r * r / T(2)
end

@inline function calculate(::H, hterm::SoftplusMarginNudging, model)
    spins = @inline graphstate(model)
    total = zero(eltype(model))
    β = hterm.β[]
    τ = hterm.τ[]
    @inbounds for i in eachindex(spins)
        total += softplus_margin_energy(β, hterm.y[i], hterm.mask[i], τ, spins[i])
    end
    return total
end

@inline function calculate(::H_i, hterm::SoftplusMarginNudging, model, idx)
    spins = @inline graphstate(model)
    return softplus_margin_energy(hterm.β[], hterm.y[idx], hterm.mask[idx], hterm.τ[], spins[idx])
end

@inline function calculate(::ΔH, hterm::SoftplusMarginNudging, model, proposal)
    j = at_idx(proposal)
    iszero(hterm.mask[j]) && return zero(eltype(model))
    spins = @inline graphstate(model)
    β = hterm.β[]
    y = hterm.y[j]
    mask = hterm.mask[j]
    τ = hterm.τ[]
    old_energy = softplus_margin_energy(β, y, mask, τ, spins[j])
    new_energy = softplus_margin_energy(β, y, mask, τ, to_val(proposal))
    return new_energy - old_energy
end

@inline function calculate(::d_iH, hterm::SoftplusMarginNudging, model, proposal::SingleSpinProposal)
    s_idx = @inline at_idx(proposal)
    mask = hterm.mask[s_idx]
    iszero(mask) && return zero(eltype(model))

    spins = @inline graphstate(model)
    y = hterm.y[s_idx]
    τ = hterm.τ[]
    τ > zero(τ) || throw(ArgumentError("SoftplusMarginNudging τ must be positive"))

    # dH/ds_i = -β m_i y_i r_i σ((1 - y_i s_i) / τ).
    state = @inline proposed_value(spins, proposal)
    z = (one(τ) - y * state) / τ
    r = softplus_margin_residual(z, τ)
    return -hterm.β[] * mask * y * r * softplus_margin_sigmoid(z)
end

"""
    clamp!(nudge, layer, vals)

Install bipolar target values on one layer and enable softplus-margin nudging
for exactly those layer spins.
"""
function clamp!(c::SoftplusMarginNudging, layer::AbstractIsingLayer, vals::V) where {V<:AbstractVector}
    @assert length(vals) == length(state(layer)) "Length of vals must match number of states in layer"
    idxs = layerrange(layer)
    c.y[idxs] .= vals
    c.mask[idxs] .= one(eltype(c.mask))
    return
end
