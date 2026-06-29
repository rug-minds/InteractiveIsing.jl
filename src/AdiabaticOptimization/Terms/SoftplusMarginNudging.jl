# AI Generated

"""
    _accumulate_whole_state_derivative!(dest, d_sH(), hterm::SoftplusMarginNudging, model, backend, scale)

Accumulate the softplus-margin nudging whole-state derivative into `dest`.
"""
function _accumulate_whole_state_derivative!(
    dest::AbstractVector,
    ds::d_sH,
    hterm::SoftplusMarginNudging,
    model::M,
    backend::B,
    scale::S,
) where {M<:AbstractIsingGraph,B<:AbstractGPUBackend,S<:Real}
    state = backend_state(backend, model)
    y = backend_vector(backend, getproperty(hterm, :y))
    mask = backend_vector(backend, getproperty(hterm, :mask))
    T = eltype(dest)
    β = backend_scalar(getproperty(hterm, :β), T)
    τ = backend_scalar(getproperty(hterm, :τ), T)
    τ > zero(T) || throw(ArgumentError("SoftplusMarginNudging τ must be positive"))

    # Use broadcasted `ifelse` branches so the expression remains GPU-friendly.
    z = @. (one(T) - y * state) / τ
    residual = @. τ * ifelse(z > T(18), z, ifelse(z < -T(18), exp(z), log1p(exp(z))))
    sigmoid = @. ifelse(z >= zero(T), one(T) / (one(T) + exp(-z)), exp(z) / (one(T) + exp(z)))
    @. dest += -T(scale) * β * mask * y * residual * sigmoid
    return dest
end
