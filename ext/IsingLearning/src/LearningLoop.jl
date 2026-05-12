export graph_view,
       graph_port_forward!,
       contrastive_gradients,
       apply_sgd_update,
       contrastive_train_step!,
       fit_contrastive!

using ProgressMeter: Progress, ProgressUnknown, next!, finish!

"""
    graph_view(g, idxs) -> Vector{Float32}

Read a slice of the graph state and materialize it as a dense vector.
This is the main bridge between the graph world and ordinary Lux-style
vector activations.
"""
function graph_view(g, idxs)
    return Float32.(copy(state(g)[idxs]))
end

"""
    graph_port_forward!(g, input_idxs, output_idxs, x; clamp_input!, run_dynamics!) -> y

Generic "port" abstraction for a graph-owning Lux layer.

- `x` is treated like the activation coming from a normal Lux layer.
- `clamp_input!` pins that activation into the graph.
- `run_dynamics!` advances the custom Monte Carlo / simulation dynamics.
- The return value is a dense readout vector that can feed into the next
  Lux layer.

This is the practical pattern for composing your graph with standard Lux
primitives:

```julia
model = Lux.Chain(
    encoder,
    graph_layer,  # internally calls graph_port_forward!
    decoder,
)
```

If you need multiple views into the same graph, prefer one graph-owning
layer with multiple helper ports instead of several independent Lux layers.
Independent Lux layers would each allocate their own `st.graph`.
"""
function graph_port_forward!(
    g,
    input_idxs,
    output_idxs,
    x;
    clamp_input!,
    run_dynamics!,
)
    clamp_input!(g, input_idxs, x)
    run_dynamics!(g)
    return graph_view(g, output_idxs)
end

"""
    apply_sgd_update(ps, grads; η) -> ps_new

Minimal parameter update for custom contrastive learning.
Swap this out for `Optimisers.update` later if you want momentum, Adam, etc.
"""
function apply_sgd_update(ps, grads; η::Real = 1f-3)
    ηf = Float32(η)
    return (
        weights = ps.weights .- ηf .* grads.weights,
        biases = ps.biases .- ηf .* grads.biases,
    )
end

function _progress_tracker(data; enabled::Bool, desc::AbstractString)
    enabled || return nothing

    try
        return Progress(length(data); desc = desc)
    catch
    end

    return ProgressUnknown(; desc = desc)
end

function _progress_step!(progress, nsteps::Integer, total_loss::Real)
    progress === nothing && return nothing

    mean_loss = total_loss / max(nsteps, 1)
    next!(progress; showvalues = [(:step, nsteps), (:mean_loss, mean_loss)])
    return nothing
end



"""
    fit_contrastive!(layer, data, ps, st; kwargs...) -> (ps, st, stats)

Run a simple custom epoch over `(x, target)` pairs.

This is intentionally not tied to Lux AD. The graph layer still follows the
Lux parameter/state split, but the learning signal comes from your own
contrastive Monte Carlo procedure.
"""
function fit_contrastive!(
    layer::LayeredIsingGraphLayer,
    data,
    ps,
    st;
    β::Real = layer.β,
    η::Real = 1f-3,
    run_free_phase! = _missing_free_phase!,
    run_nudged_phase! = _missing_nudged_phase!,
    update_rule = apply_sgd_update,
    show_progress::Bool = true,
    progress_desc::AbstractString = "Contrastive training",
)
    total_loss = 0f0
    nsteps = 0
    progress = _progress_tracker(data; enabled = show_progress, desc = progress_desc)

    for (x, target) in data
        ps, st, aux = contrastive_train_step!(
            layer,
            x,
            target,
            ps,
            st;
            β = β,
            η = η,
            run_free_phase! = run_free_phase!,
            run_nudged_phase! = run_nudged_phase!,
            update_rule = update_rule,
        )

        total_loss += sum(abs2, aux.y .- Float32.(target))
        nsteps += 1
        _progress_step!(progress, nsteps, total_loss)
    end

    progress === nothing || finish!(progress)
    mean_loss = nsteps == 0 ? 0f0 : total_loss / nsteps
    return ps, st, (; mean_loss, nsteps)
end

