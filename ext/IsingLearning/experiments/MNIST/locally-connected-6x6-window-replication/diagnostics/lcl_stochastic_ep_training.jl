include(joinpath(@__DIR__, "..", "mnist_lcl_6x6_window_adam.jl"))

const LCL_STOCHASTIC_SAMPLES = parse(Int, get(ENV, "ISING_MNIST_LCL_STOCHASTIC_SAMPLES", "4"))

"""Accumulate a scaled one-sided observable-gradient contribution for the LCL graph.

This is the stochastic-EP diagnostic estimator. Each nudged rollout contributes
`scale * (dH(s_beta) - dH(s_free))`; the normal manager flush later applies the
usual `1 / (beta * nsamples)` minibatch scaling. For hidden-output couplings this
averages `s_i * s_j` observables directly, not products of averaged states.
"""
function accumulate_lcl_scaled_observable_gradient!(
    isinggraph::G,
    nudged_state::N,
    equilibrium_state::E,
    x::X,
    buffers::B,
    scale::T,
) where {
    G,
    N<:AbstractVector,
    E<:AbstractVector,
    X<:AbstractVector,
    B,
    T<:Real,
}
    signed_scale = FT(scale)
    adjacency = II.adj(isinggraph)

    # Hidden-output graph parameters: dH/dJ = -1/2 * s_i * s_j.
    @inbounds for (ptr, (src, dst)) in enumerate(II.index_pairs_iterator(adjacency, false))
        buffers.w[ptr] += signed_scale * FT(-0.5) *
            (nudged_state[src] * nudged_state[dst] - equilibrium_state[src] * equilibrium_state[dst])
    end

    # Learnable base bias field: dH/db = -s.
    @inbounds for idx in eachindex(buffers.b)
        buffers.b[idx] += -signed_scale * (nudged_state[idx] - equilibrium_state[idx])
    end

    # External LCL input projection: dH/dW_input = -x_i * h_j.
    patches = LCL_PATCH_INPUT_IDXS[]
    length(patches) == size(buffers.w_input, 1) || error("LCL patch table does not match hidden count")
    @inbounds for hidden_idx in eachindex(patches)
        state_delta = nudged_state[hidden_idx] - equilibrium_state[hidden_idx]
        for input_idx in patches[hidden_idx]
            buffers.w_input[hidden_idx, input_idx] += -signed_scale * x[input_idx] * state_delta
        end
    end
    return buffers
end

# ProcessAlgorithm wrapper so the stochastic sampler can update worker-local buffers.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateLCLScaledObservableGradientRef!(
    isinggraph::G,
    nudged_state::AbstractVector,
    equilibrium_state::AbstractVector,
    x,
    buffers,
    sample_scale::Float32,
) where {G}
    accumulate_lcl_scaled_observable_gradient!(
        isinggraph,
        nudged_state,
        equilibrium_state,
        ref_value(x),
        buffers,
        sample_scale,
    )
    return nothing
end

"""Build one stochastic nudged rollout that accumulates averaged observables."""
function lcl_stochastic_nudged_observable_algorithm(
    layer::L,
    sample_scale::T,
) where {L<:IsingLearning.LayeredIsingGraphLayer,T<:Real}
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    default_β = layer.β
    output_idxs = collect(layer.output_layer)
    scaled = FT(sample_scale)

    if LCL_TANGENT_NUDGE
        return StatefulAlgorithms.@Routine begin
            @state x
            @state y
            @state input_hidden_w
            @state input_pattern
            @state phase_beta = default_β
            @state equilibrium_state
            @state nudged_state = zeros(n_units)
            @state output_idxs = output_idxs
            @state buffers
            @state sample_scale = scaled
            @alias dynamics = dynamics_algorithm

            # Restart from the same free equilibrium; stochasticity comes from LocalLangevin noise.
            IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
            ApplyTangentProjectedInputFieldRef!(
                dynamics.model,
                input_hidden_w,
                x,
                y,
                equilibrium_state,
                input_pattern,
                output_idxs,
                phase_beta,
            )
            model = @repeat nudged_steps dynamics()
            IsingLearning.CopyGraphState!(nudged_state, model)
            AccumulateLCLScaledObservableGradientRef!(
                dynamics.model,
                nudged_state,
                equilibrium_state,
                x,
                buffers,
                sample_scale,
            )
        end
    end

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern
        @state phase_beta = default_β
        @state equilibrium_state
        @state nudged_state = zeros(n_units)
        @state buffers
        @state sample_scale = scaled
        @alias dynamics = dynamics_algorithm

        # Clamp-style fallback retained for direct comparison to the main LCL runner.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        ApplyTargetsRef!(dynamics.model, y)
        SetInputFieldClampingBeta!(dynamics.model, phase_beta)
        model = @repeat nudged_steps dynamics()
        IsingLearning.CopyGraphState!(nudged_state, model)
        AccumulateLCLScaledObservableGradientRef!(
            dynamics.model,
            nudged_state,
            equilibrium_state,
            x,
            buffers,
            sample_scale,
        )
        SetInputFieldClampingBeta!(dynamics.model, 0f0)
    end
end

"""Build a one-sided stochastic-EP worker for the scalar LCL diagnostic.

The free phase is the normal single relaxation endpoint. The nudged phase is
repeated `ISING_MNIST_LCL_STOCHASTIC_SAMPLES` times and averages parameter
observables across the resulting noisy nudged trajectories.
"""
function input_field_contrastive_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    default_β = layer.β
    stochastic_samples = max(1, LCL_STOCHASTIC_SAMPLES)
    sample_scale = inv(FT(stochastic_samples))
    free_phase = input_field_free_phase_algorithm(layer)
    nudged_sample = lcl_stochastic_nudged_observable_algorithm(layer, sample_scale)

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @input phase_beta::FT = default_β

        @context free_context = free_phase()
        @context nudged_context = @repeat stochastic_samples nudged_sample()
        @bind x => free_context.x
        @bind x => nudged_context.x
        @bind y => nudged_context.y
        @bind buffers => nudged_context.buffers
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => nudged_context.input_hidden_w
        @merge free_context.input_pattern, nudged_context.input_pattern
        @merge free_context.equilibrium_state, nudged_context.equilibrium_state
        @bind phase_beta => nudged_context.phase_beta
    end
end

"""Write diagnostic-specific stochastic-EP settings next to the normal settings."""
function write_stochastic_settings!(config::C) where {C<:InputFieldMNISTConfig}
    path = joinpath(config.outdir, "stochastic_ep_settings.md")
    open(path, "w") do io
        println(io, "# LCL Stochastic-EP Diagnostic")
        println(io)
        println(io, "- estimator: one-sided nudged observable average")
        println(io, "- stochastic nudged rollouts per example: `$(LCL_STOCHASTIC_SAMPLES)`")
        println(io, "- nudging: `$(LCL_TANGENT_NUDGE ? "tangent fixed free-equilibrium error force" : "baseline clamp-style nudge")`")
        println(io, "- beta/temp/stepsize: `$(config.β)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- sweeps: `$(config.sweeps)`")
        println(io, "- note: hidden-output gradients average `s_i*s_j` observables directly.")
    end
    return path
end

"""Run the stochastic-EP LCL diagnostic with normal environment overrides."""
function main()
    hidden_default = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2
    base = InputFieldMNISTConfig()
    config = updated_config(
        base;
        workers = haskey(ENV, "ISING_MNIST_IF_WORKERS") ? base.workers : 32,
        epochs = haskey(ENV, "ISING_MNIST_IF_EPOCHS") ? base.epochs : 80,
        batchsize = haskey(ENV, "ISING_MNIST_IF_BATCHSIZE") ? base.batchsize : 200,
        hidden = haskey(ENV, "ISING_MNIST_IF_HIDDEN") ? base.hidden : hidden_default,
        output_replicas = haskey(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS") ? base.output_replicas : 1,
        lr = haskey(ENV, "ISING_MNIST_IF_LR") ? base.lr : FT(1f-4),
        β = haskey(ENV, "ISING_MNIST_IF_BETA") ? base.β : FT(0.3),
        train_per_class = haskey(ENV, "ISING_MNIST_IF_TRAIN_PER_CLASS") ? base.train_per_class : 500,
        test_per_class = haskey(ENV, "ISING_MNIST_IF_TEST_PER_CLASS") ? base.test_per_class : 100,
        outdir = haskey(ENV, "ISING_MNIST_IF_OUTDIR") ? base.outdir :
            joinpath(@__DIR__, "..", "experiments", "current", default_run_dirname("lcl_6x6_stochastic_ep_diagnostic")),
    )
    result = run_config!(config)
    write_stochastic_settings!(config)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
