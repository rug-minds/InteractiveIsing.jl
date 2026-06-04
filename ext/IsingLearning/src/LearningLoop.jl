export graph_view,
       apply_sgd_update,
       contrastive_train_step!,
       fit_contrastive!

using ProgressMeter: Progress, ProgressUnknown, next!, finish!

"""
    graph_view(graph, idxs)

Materialize `state(graph)[idxs]` as a dense `Float32` vector for use as a Lux
activation or experiment metric.
"""
function graph_view(graph::G, idxs::I) where {G,I}
    return Float32.(copy(state(graph)[idxs]))
end

"""
    apply_sgd_update(ps, grads; η = 1f-3)

Apply a plain SGD update to a parameter NamedTuple. This is intentionally small;
experiments that need Adam or momentum should use `Optimisers.update`.
"""
function apply_sgd_update(ps::P, grads::G; η::Real = 1f-3) where {P,G}
    ηf = Float32(η)
    updated = (;
        w = ps.w .- ηf .* grads.w,
        b = ps.b .- ηf .* grads.b,
    )
    hasproperty(ps, :α) || return updated
    return merge(updated, (; α = ps.α .- ηf .* grads.α))
end

"""Set every gradient-buffer array to zero before a new sample or minibatch."""
function clear_gradient_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Return a scaled copy of a contrastive-gradient buffer."""
function scaled_gradient(buffer::B, scale::T) where {B,T<:Real}
    grad = (;
        w = copy(buffer.w) .* scale,
        b = copy(buffer.b) .* scale,
    )
    hasproperty(buffer, :α) || return grad
    return merge(grad, (; α = copy(buffer.α) .* scale))
end

"""
    contrastive_train_step!(layer, x, target, ps, st; β = layer.β, η = 1f-3)

Run one single-example contrastive update using the reusable
`st.contrastive_process` prepared by the Lux layer. The process context owns the
graph, sample buffers, state captures, sampler contexts, and gradient buffers.
"""
function contrastive_train_step!(
    layer::L,
    x::X,
    target::Y,
    ps::P,
    st::S;
    β::Real = layer.β,
    η::Real = 1f-3,
    update_rule = apply_sgd_update,
) where {L<:LayeredIsingGraphLayer,X,Y,P,S}
    graph = st.graph
    sync_params!(graph, ps)

    process = st.contrastive_process
    context = StatefulAlgorithms.context(process)._state
    context.x .= x
    context.y .= target
    clear_gradient_buffer!(context.buffers)

    StatefulAlgorithms.reset!(process)
    run(process)
    wait(process)

    scale = inv(Float32(2) * Float32(β))
    grads = scaled_gradient(context.buffers, scale)
    ps_new = update_rule(ps, grads; η)
    output = Float32.(copy(@view context.equilibrium_state[layer.output_layer]))
    return ps_new, st, (; y = output, grads)
end

"""Create a progress tracker that works for sized and unsized iterables."""
function contrastive_progress(data::D; enabled::Bool, desc::AbstractString) where {D}
    enabled || return nothing
    try
        return Progress(length(data); desc)
    catch
        return ProgressUnknown(; desc)
    end
end

"""Advance a progress tracker with the current running mean loss."""
function contrastive_progress_step!(progress, nsteps::Integer, total_loss::Real)
    progress === nothing && return nothing
    mean_loss = total_loss / max(Int(nsteps), 1)
    next!(progress; showvalues = [(:step, Int(nsteps)), (:mean_loss, mean_loss)])
    return nothing
end

"""
    fit_contrastive!(layer, data, ps, st; kwargs...) -> (ps, st, stats)

Run a clean custom contrastive loop over `(x, target)` pairs. This keeps Lux's
parameter/state split but uses explicit process-based learning instead of AD.
"""
function fit_contrastive!(
    layer::L,
    data,
    ps,
    st;
    β::Real = layer.β,
    η::Real = 1f-3,
    update_rule = apply_sgd_update,
    show_progress::Bool = true,
    progress_desc::AbstractString = "Contrastive training",
) where {L<:LayeredIsingGraphLayer}
    total_loss = 0f0
    nsteps = 0
    progress = contrastive_progress(data; enabled = show_progress, desc = progress_desc)

    for (x, target) in data
        ps, st, aux = contrastive_train_step!(
            layer,
            x,
            target,
            ps,
            st;
            β,
            η,
            update_rule,
        )
        total_loss += sum(abs2, aux.y .- Float32.(target))
        nsteps += 1
        contrastive_progress_step!(progress, nsteps, total_loss)
    end

    progress === nothing || finish!(progress)
    mean_loss = nsteps == 0 ? 0f0 : total_loss / nsteps
    return ps, st, (; mean_loss, nsteps)
end
