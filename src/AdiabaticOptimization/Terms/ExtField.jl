# AI Generated

"""
    _accumulate_whole_state_derivative!(dest, d_sH(), hterm::ExtField, model, backend, scale)

Accumulate the external-field whole-state derivative `-c * b` into `dest`.
"""
function _accumulate_whole_state_derivative!(
    dest::AbstractVector,
    ds::d_sH,
    hterm::ExtField,
    model::M,
    backend::B,
    scale::S,
) where {M<:AbstractIsingGraph,B<:AbstractGPUBackend,S<:Real}
    b = backend_vector(backend, getproperty(hterm, :b))
    T = eltype(dest)
    c = backend_scalar(getproperty(hterm, :c), T)
    @. dest += -T(scale) * c * b
    return dest
end
