# AI Generated

"""
    _accumulate_whole_state_derivative!(dest, d_sH(), hterm::Bilinear, model, backend, scale)

Accumulate the bilinear whole-state derivative `-J * state` into `dest`.
"""
function _accumulate_whole_state_derivative!(
    dest::AbstractVector,
    ds::d_sH,
    hterm::Bilinear,
    model::M,
    backend::B,
    scale::S,
) where {M<:AbstractIsingGraph,B<:AbstractGPUBackend,S<:Real}
    state = backend_state(backend, model)
    J = backend_matrix(backend, getproperty(hterm, :J))
    T = eltype(dest)
    mul!(dest, J, state, -T(scale), one(T))
    return dest
end
