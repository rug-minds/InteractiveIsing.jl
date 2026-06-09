using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using LinearAlgebra
using Optimisers
using Random

include(joinpath(@__DIR__, "..", "mnist_lcl_6x6_window_adam.jl"))

"""Return the expected flushed input-projection gradient for one manager job."""
function expected_input_gradient(
    x::X,
    free_state::S,
    nudged_state::S,
    β::T,
) where {X<:AbstractVector,S<:AbstractVector,T<:Real}
    patches = LCL_PATCH_INPUT_IDXS[]
    expected = zeros(Float32, length(patches), INPUT_DIM)
    invβ = inv(Float32(β))
    @inbounds for hidden_idx in eachindex(patches)
        delta = (nudged_state[hidden_idx] - free_state[hidden_idx]) * invβ
        for input_idx in patches[hidden_idx]
            expected[hidden_idx, input_idx] = -x[input_idx] * delta
        end
    end
    return expected
end

"""Validate the live manager's input matrix gradient, Adam update, mask, and shared ref."""
function main()
    rng = Random.MersenneTwister(20260607)
    config = updated_config(
        InputFieldMNISTConfig();
        workers = 1,
        epochs = 0,
        batchsize = 1,
        train_per_class = 1,
        test_per_class = 1,
        train_eval_per_class = 0,
        hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2,
        output_replicas = 1,
        β = 0.3f0,
        lr = 1f-3,
        sweeps = 2f0,
        outdir = joinpath(@__DIR__, "tmp_validate_lcl_input_matrix_update"),
    )
    setup = build_layer(config)
    input_hidden_w = Ref(copy(setup.input_hidden_w))
    manager = input_field_manager(setup.layer, setup.graph, config, input_hidden_w)
    worker = first(StatefulAlgorithms.workers(manager))
    ctx = worker_context(worker)

    try
        x = rand(rng, Float32, INPUT_DIM)
        y = fill(-1f0, NCLASSES)
        y[4] = 1f0
        jobs = [InputFieldMNISTJob(copy(x), copy(y))]

        ctx.input_hidden_w === manager.state.input_hidden_w ||
            error("worker does not hold the manager input_hidden_w Ref")
        manager.state.input_hidden_w[] == manager.state.params[].w_input ||
            error("manager input_hidden_w values do not match initial optimizer params")

        clear_manager_buffers!(manager)
        manager.state.nsamples[] = length(jobs)
        StatefulAlgorithms.run!(manager, jobs)

        expected = expected_input_gradient(x, ctx.equilibrium_state, ctx.nudged_state, config.β)
        got = manager.state.batch_gradient.w_input
        mask = LCL_INPUT_MASK[]
        max_gradient_error = maximum(abs.(got .- expected))
        max_mask_leak = maximum(abs.(got[.!mask]))
        grad_norm = norm(got)
        grad_norm > 0 || error("input matrix gradient is zero after one contrastive job")

        before = copy(manager.state.params[].w_input)
        old_ref_value = manager.state.input_hidden_w[]
        manager.state.opt_state, updated = Optimisers.update(
            manager.state.opt_state,
            manager.state.params[],
            manager.state.batch_gradient,
        )
        manager.state.params[] = updated
        sync_after_update!(manager, updated)
        after = manager.state.params[].w_input
        delta = after .- before

        manager.state.input_hidden_w[] === after ||
            error("manager input_hidden_w Ref was not repointed to updated params")
        ref_value(ctx.input_hidden_w) === after ||
            error("worker input_hidden_w Ref does not see updated params")
        old_ref_value === after && error("Adam update unexpectedly reused old input matrix object")

        max_update_leak = maximum(abs.(delta[.!mask]))
        masked_delta_norm = norm(delta[mask])
        descent_dot = dot(vec(delta[mask]), vec(got[mask]))

        println("max_gradient_error = ", max_gradient_error)
        println("max_mask_leak = ", max_mask_leak)
        println("grad_norm = ", grad_norm)
        println("masked_delta_norm = ", masked_delta_norm)
        println("max_update_leak = ", max_update_leak)
        println("descent_dot = ", descent_dot)

        max_gradient_error < 1f-5 || error("input matrix gradient does not match contrastive image-hidden rule")
        max_mask_leak == 0f0 || error("input matrix gradient leaks outside local receptive-field mask")
        masked_delta_norm > 0 || error("Adam did not update masked input weights")
        max_update_leak == 0f0 || error("Adam update changed input weights outside local receptive-field mask")
        descent_dot < 0 || error("Adam input update is not a descent step against the flushed gradient")
        println("LCL input matrix manager update validation passed")
    finally
        close(manager)
    end
end

main()
