export LinearReadoutClamping, ConstantLinearReadoutNudge

raw"""
    LinearReadoutClamping(output_idxs, readout; β = 0, target = 0)

Learning-extension Hamiltonian term for nudging a scalar linear readout of an
output layer:

```math
H_{clamp}(s) = \frac{β}{2}(w^T s_{out} - y)^2
```

This is the right clamping term when the supervised loss is a squared error on
a readout `w' * output_state`, rather than a separate squared error on every
output spin. In that case the output spins are coupled by the loss, so this term
also implements a direct `MultiSpinProposal` `ΔH` instead of using the generic
sequential fallback.
"""
struct LinearReadoutClamping{B,T,I,W} <: InteractiveIsing.HamiltonianTerm
    β::B
    target::T
    output_idxs::I
    readout::W
end

raw"""
    ConstantLinearReadoutNudge(output_idxs, readout; β = 0, target = 0, free_score = 0)

Learning-extension Hamiltonian term for the "constant nudge" used by
Laydevant et al. in binary EP:

```math
H_{nudge}(s) = -β (y - w^T s^*_out) (w^T s_{out})
```

`free_score` must be set from the free-phase equilibrium before the nudged
phase. Unlike `LinearReadoutClamping`, the nudging force is constant during one
nudged relaxation.
"""
struct ConstantLinearReadoutNudge{B,T,F,I,W} <: InteractiveIsing.HamiltonianTerm
    β::B
    target::T
    free_score::F
    output_idxs::I
    readout::W
end

_scalar_value(x::AbstractArray) = x[]
_scalar_value(x) = x
_scalar_storage(x::AbstractArray, ::Type) = x
_scalar_storage(x, ::Type{T}) where {T} = InteractiveIsing.UniformArray(T(x))

function LinearReadoutClamping(output_idxs, readout; β = 0, target = 0)
    output_idxs_vec = collect(Int, output_idxs)
    readout_vec = collect(readout)
    length(output_idxs_vec) == length(readout_vec) ||
        throw(ArgumentError("output_idxs and readout must have the same length"))
    isempty(output_idxs_vec) && throw(ArgumentError("output_idxs cannot be empty"))

    β0 = _scalar_value(β)
    target0 = _scalar_value(target)
    T = promote_type(eltype(readout_vec), typeof(β0), typeof(target0))
    return LinearReadoutClamping(
        _scalar_storage(β, T),
        _scalar_storage(target, T),
        output_idxs_vec,
        T.(readout_vec),
    )
end

function ConstantLinearReadoutNudge(output_idxs, readout; β = 0, target = 0, free_score = 0)
    output_idxs_vec = collect(Int, output_idxs)
    readout_vec = collect(readout)
    length(output_idxs_vec) == length(readout_vec) ||
        throw(ArgumentError("output_idxs and readout must have the same length"))
    isempty(output_idxs_vec) && throw(ArgumentError("output_idxs cannot be empty"))

    β0 = _scalar_value(β)
    target0 = _scalar_value(target)
    free_score0 = _scalar_value(free_score)
    T = promote_type(eltype(readout_vec), typeof(β0), typeof(target0), typeof(free_score0))
    return ConstantLinearReadoutNudge(
        _scalar_storage(β, T),
        _scalar_storage(target, T),
        _scalar_storage(free_score, T),
        output_idxs_vec,
        T.(readout_vec),
    )
end

"""
    readout_score(hterm, state)

Evaluate `w' * state[output_idxs]` for a linear readout clamping term.
"""
function readout_score(hterm::Union{LinearReadoutClamping,ConstantLinearReadoutNudge}, state)
    score = zero(promote_type(eltype(hterm.readout), eltype(state)))
    @inbounds for local_idx in eachindex(hterm.output_idxs, hterm.readout)
        score += hterm.readout[local_idx] * state[hterm.output_idxs[local_idx]]
    end
    return score
end

function _readout_position(hterm::Union{LinearReadoutClamping,ConstantLinearReadoutNudge}, state_idx::Integer)
    return findfirst(==(state_idx), hterm.output_idxs)
end

"""
    calculate(H(), hterm::LinearReadoutClamping, model)

Return the current readout-clamping energy `β/2 * (w' * s_out - target)^2`.
"""
function InteractiveIsing.calculate(
    ::InteractiveIsing.H,
    hterm::LinearReadoutClamping,
    model::InteractiveIsing.AbstractIsingGraph,
)
    state = InteractiveIsing.graphstate(model)
    residual = readout_score(hterm, state) - hterm.target[]
    return hterm.β[] / 2 * residual^2
end

raw"""
    calculate(d_iH(), hterm::LinearReadoutClamping, model, s_idx)

Derivative of the readout-clamping energy with respect to one state value:

```math
\partial H_{clamp}/\partial s_i =
β (w^T s_{out} - y) w_i
```

for output spins, and zero for non-output spins.
"""
function InteractiveIsing.calculate(
    ::InteractiveIsing.d_iH,
    hterm::LinearReadoutClamping,
    model::InteractiveIsing.AbstractIsingGraph,
    s_idx,
)
    pos = _readout_position(hterm, s_idx)
    isnothing(pos) && return zero(eltype(model))

    state = InteractiveIsing.graphstate(model)
    residual = readout_score(hterm, state) - hterm.target[]
    return hterm.β[] * residual * hterm.readout[pos]
end

"""
    calculate(ΔH(), hterm::LinearReadoutClamping, model, proposal)

Energy change for a single-spin proposal. If the proposal does not touch the
readout output indices, the clamping contribution is zero.
"""
function InteractiveIsing.calculate(
    ::InteractiveIsing.ΔH,
    hterm::LinearReadoutClamping,
    model::InteractiveIsing.AbstractIsingGraph,
    proposal::InteractiveIsing.FlipProposal,
)
    pos = _readout_position(hterm, InteractiveIsing.at_idx(proposal))
    isnothing(pos) && return zero(eltype(model))

    state = InteractiveIsing.graphstate(model)
    old_score = readout_score(hterm, state)
    new_score = old_score + hterm.readout[pos] *
        (InteractiveIsing.to_val(proposal) - InteractiveIsing.from_val(proposal))
    target = hterm.target[]
    return hterm.β[] / 2 * ((new_score - target)^2 - (old_score - target)^2)
end

function _constant_readout_error(hterm::ConstantLinearReadoutNudge)
    return hterm.target[] - hterm.free_score[]
end

"""
    calculate(H(), hterm::ConstantLinearReadoutNudge, model)

Return the constant-readout nudging energy
`-β * (target - free_score) * (w' * s_out)`.
"""
function InteractiveIsing.calculate(
    ::InteractiveIsing.H,
    hterm::ConstantLinearReadoutNudge,
    model::InteractiveIsing.AbstractIsingGraph,
)
    state = InteractiveIsing.graphstate(model)
    return -hterm.β[] * _constant_readout_error(hterm) * readout_score(hterm, state)
end

function InteractiveIsing.calculate(
    ::InteractiveIsing.d_iH,
    hterm::ConstantLinearReadoutNudge,
    model::InteractiveIsing.AbstractIsingGraph,
    s_idx,
)
    pos = _readout_position(hterm, s_idx)
    isnothing(pos) && return zero(eltype(model))
    return -hterm.β[] * _constant_readout_error(hterm) * hterm.readout[pos]
end

function InteractiveIsing.calculate(
    ::InteractiveIsing.ΔH,
    hterm::ConstantLinearReadoutNudge,
    model::InteractiveIsing.AbstractIsingGraph,
    proposal::InteractiveIsing.FlipProposal,
)
    pos = _readout_position(hterm, InteractiveIsing.at_idx(proposal))
    isnothing(pos) && return zero(eltype(model))

    Δscore = hterm.readout[pos] *
        (InteractiveIsing.to_val(proposal) - InteractiveIsing.from_val(proposal))
    return -hterm.β[] * _constant_readout_error(hterm) * Δscore
end

function InteractiveIsing.calculate(
    ::InteractiveIsing.ΔH,
    hterm::ConstantLinearReadoutNudge,
    model::InteractiveIsing.AbstractIsingGraph,
    proposal::InteractiveIsing.MultiSpinProposal,
)
    score_delta = zero(eltype(hterm.readout))
    @inbounds for proposal_idx in 1:length(proposal)
        pos = _readout_position(hterm, InteractiveIsing.at_idx(proposal, proposal_idx))
        isnothing(pos) && continue
        score_delta += hterm.readout[pos] * InteractiveIsing.delta(proposal, proposal_idx)
    end
    iszero(score_delta) && return zero(eltype(model))
    return -hterm.β[] * _constant_readout_error(hterm) * score_delta
end

"""
    calculate(ΔH(), hterm::LinearReadoutClamping, model, proposal::MultiSpinProposal)

Energy change for a simultaneous multi-spin proposal. This intentionally does
not use the generic single-spin fallback because the readout loss has cross
terms between output spins.
"""
function InteractiveIsing.calculate(
    ::InteractiveIsing.ΔH,
    hterm::LinearReadoutClamping,
    model::InteractiveIsing.AbstractIsingGraph,
    proposal::InteractiveIsing.MultiSpinProposal,
)
    state = InteractiveIsing.graphstate(model)
    old_score = readout_score(hterm, state)
    score_delta = zero(typeof(old_score))

    @inbounds for proposal_idx in 1:length(proposal)
        pos = _readout_position(hterm, InteractiveIsing.at_idx(proposal, proposal_idx))
        isnothing(pos) && continue
        score_delta += hterm.readout[pos] * InteractiveIsing.delta(proposal, proposal_idx)
    end

    iszero(score_delta) && return zero(eltype(model))
    target = hterm.target[]
    new_score = old_score + score_delta
    return hterm.β[] / 2 * ((new_score - target)^2 - (old_score - target)^2)
end
