include(joinpath(@__DIR__, "low_beta_diagnostic_common.jl"))

# Accumulate one full graph-state sample into a free-phase average buffer owned
# by the process algorithm context.
StatefulAlgorithms.@ProcessAlgorithm begin
    @config nstates::Int = 1

    function SampleFreeAverageState!(
        isinggraph,
        @managed(sum = zeros(PMNIST_FT, nstates)),
        @managed(count::Int = 0),
    )
        sum .+= II.state(isinggraph)
        return (; count = count + 1)
    end
end

# Accumulate one full graph-state sample into a nudged-phase average buffer
# owned by the process algorithm context.
StatefulAlgorithms.@ProcessAlgorithm begin
    @config nstates::Int = 1

    function SamplePlusAverageState!(
        isinggraph,
        @managed(sum = zeros(PMNIST_FT, nstates)),
        @managed(count::Int = 0),
    )
        sum .+= II.state(isinggraph)
        return (; count = count + 1)
    end
end

# Accumulate one full graph-state sample into a negative-nudge average buffer.
StatefulAlgorithms.@ProcessAlgorithm begin
    @config nstates::Int = 1

    function SampleMinusAverageState!(
        isinggraph,
        @managed(sum = zeros(PMNIST_FT, nstates)),
        @managed(count::Int = 0),
    )
        sum .+= II.state(isinggraph)
        return (; count = count + 1)
    end
end

# Write a state-averager subcontext mean into a persistent phase-owned buffer.
StatefulAlgorithms.@ProcessAlgorithm function WriteAverageState!(
    dest::AbstractVector,
    sum::AbstractVector,
    count::Integer,
)
    count > 0 || error("state averager recorded no samples")
    dest .= sum
    dest ./= PMNIST_FT(count)
    return nothing
end

"""Build a repeated measurement block that steps dynamics then samples state."""
function stateavg_measurement_block(
    dynamics_algorithm::D,
    nstates::I,
    sample_interval::I,
    alias_name::Symbol,
) where {D,I<:Integer}
    if alias_name === :free_state_averager
        return StatefulAlgorithms.@Routine begin
            @alias dynamics = dynamics_algorithm
            @repeat sample_interval dynamics()
            @alias free_state_averager = SampleFreeAverageState!(; nstates = nstates)
            free_state_averager(isinggraph = dynamics.model)
        end
    elseif alias_name === :plus_state_averager
        return StatefulAlgorithms.@Routine begin
            @alias dynamics = dynamics_algorithm
            @repeat sample_interval dynamics()
            @alias plus_state_averager = SamplePlusAverageState!(; nstates = nstates)
            plus_state_averager(isinggraph = dynamics.model)
        end
    elseif alias_name === :minus_state_averager
        return StatefulAlgorithms.@Routine begin
            @alias dynamics = dynamics_algorithm
            @repeat sample_interval dynamics()
            @alias minus_state_averager = SampleMinusAverageState!(; nstates = nstates)
            minus_state_averager(isinggraph = dynamics.model)
        end
    end
    throw(ArgumentError("unknown state averager alias `$alias_name`"))
end

# Install the positive target nudge used by one-sided and symmetric estimators.
StatefulAlgorithms.@ProcessAlgorithm function InstallPositiveNudgedSampleBias!(
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_sample_bias!(learning_model, x, base_bias, sample_buffer, y, learning_model.config.β)
    return nothing
end

# Install the negative target nudge used by the symmetric finite difference.
StatefulAlgorithms.@ProcessAlgorithm function InstallNegativeNudgedSampleBias!(
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_sample_bias!(learning_model, x, base_bias, sample_buffer, y, -learning_model.config.β)
    return nothing
end

"""Install input fields plus a tangent output nudge around the free state."""
function install_tangent_sample_bias!(
    model::M,
    x::X,
    y::Y,
    free_state::F,
    base_bias::B,
    sample_buffer::S,
    beta::Real,
) where {M<:LocalMNISTModel,X<:AbstractVector,Y<:AbstractVector,F<:AbstractVector,B<:AbstractVector,S<:AbstractVector}
    length(x) == length(model.input_idxs) ||
        throw(DimensionMismatch("input length $(length(x)) does not match input layer length $(length(model.input_idxs))"))
    length(y) == length(model.output_idxs) ||
        throw(DimensionMismatch("target length $(length(y)) does not match output layer length $(length(model.output_idxs))"))
    length(base_bias) == length(sample_buffer) == length(base_magfield(model.graph).b) ||
        throw(DimensionMismatch("base/sample/graph field lengths do not match"))

    ensure_input_layer_inactive!(model.graph.index_set)
    fill!(II.state(model.graph[1]), 0f0)
    fill!(sample_buffer, 0f0)

    A = II.adj(model.graph)
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    nz = SparseArrays.nonzeros(A)

    # The tangent target is fixed at the free state: beta * (y - s_free_out).
    @inbounds for (xpos, input_idx) in enumerate(model.input_idxs)
        xval = x[xpos]
        for ptr in colptr[input_idx]:(colptr[input_idx + 1] - 1)
            sample_buffer[rows[ptr]] += nz[ptr] * xval
        end
    end
    β = PMNIST_FT(beta)
    @inbounds for (opos, idx) in enumerate(model.output_idxs)
        sample_buffer[idx] += β * (y[opos] - free_state[idx])
    end
    sample_buffer .= clamp.(sample_buffer, -model.config.applied_bias_clip, model.config.applied_bias_clip)

    combined_b = base_magfield(model.graph).b
    combined_b .= base_bias
    combined_b .+= sample_buffer
    return model
end

# Install a positive tangent nudge, using the free output as the expansion point.
StatefulAlgorithms.@ProcessAlgorithm function InstallPositiveTangentNudgedSampleBias!(
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    free_state::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_tangent_sample_bias!(learning_model, x, y, free_state, base_bias, sample_buffer, learning_model.config.β)
    return nothing
end

# Install a negative tangent nudge for symmetric finite differences.
StatefulAlgorithms.@ProcessAlgorithm function InstallNegativeTangentNudgedSampleBias!(
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    free_state::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_tangent_sample_bias!(learning_model, x, y, free_state, base_bias, sample_buffer, -learning_model.config.β)
    return nothing
end

"""Accumulate one sampled-state observable contribution into a gradient buffer."""
function accumulate_local_observable_gradient!(
    gradient::G,
    model::M,
    x::X,
    state::S,
    scale::T,
) where {G<:NamedTuple,M<:LocalMNISTModel,X<:AbstractVector,S<:AbstractVector,T<:Real}
    config = model.config

    # This samples the stochastic EqProp observable directly. For edges this is
    # E[s_i s_j], not E[s_i]E[s_j]; input edges use the fixed image value.
    edge_layout = model.edge_layout
    edge_kind = edge_layout.kind
    edge_a = edge_layout.a
    edge_b = edge_layout.b
    grad_w = gradient.w
    signed_scale = PMNIST_FT(scale)
    @inbounds for ptr in eachindex(edge_kind)
        kind = edge_kind[ptr]
        kind == EDGE_UNUSED && continue
        if kind == EDGE_INPUT_HIDDEN
            grad_w[ptr] += signed_scale * x[edge_a[ptr]] * state[edge_b[ptr]]
        else
            grad_w[ptr] += signed_scale * state[edge_a[ptr]] * state[edge_b[ptr]]
        end
    end

    grad_b = gradient.b
    @inbounds for idx in model.hidden1_idxs
        grad_b[idx] += signed_scale * state[idx]
    end
    @inbounds for idx in model.hidden2_idxs
        grad_b[idx] += signed_scale * state[idx]
    end
    if config.train_output_bias
        @inbounds for idx in model.output_idxs
            grad_b[idx] += signed_scale * state[idx]
        end
    end
    return gradient
end

# Sample edge/bias observables during a nudged phase and add the correctly
# signed finite-difference contribution directly into the worker gradient.
StatefulAlgorithms.@ProcessAlgorithm begin
    @config scale::Float32 = 1f0

    function AccumulateGradientObservables!(
        gradient::NamedTuple,
        learning_model,
        x::AbstractVector,
        isinggraph,
    )
        accumulate_local_observable_gradient!(gradient, learning_model, x, II.state(isinggraph), scale)
        return nothing
    end
end

"""Accumulate a symmetric `(+beta - -beta) / (2beta)` local gradient."""
function accumulate_local_symmetric_contrastive_gradient!(
    gradient::G,
    model::M,
    x::X,
    plus_state::P,
    minus_state::N,
) where {G<:NamedTuple,M<:LocalMNISTModel,X<:AbstractVector,P<:AbstractVector,N<:AbstractVector}
    config = model.config
    inv2β = one(PMNIST_FT) / (2 * config.β)

    # Walk the trainable CSC nonzeros once; `edge_layout` is aligned to the
    # optimizer parameter vector, so there are no sparse lookups in this loop.
    edge_layout = model.edge_layout
    edge_kind = edge_layout.kind
    edge_a = edge_layout.a
    edge_b = edge_layout.b
    grad_w = gradient.w
    @inbounds for ptr in eachindex(edge_kind)
        kind = edge_kind[ptr]
        kind == EDGE_UNUSED && continue
        if kind == EDGE_INPUT_HIDDEN
            hidden_idx = edge_b[ptr]
            grad_w[ptr] += x[edge_a[ptr]] * (plus_state[hidden_idx] - minus_state[hidden_idx]) * inv2β
        else
            src = edge_a[ptr]
            dst = edge_b[ptr]
            grad_w[ptr] += (plus_state[src] * plus_state[dst] - minus_state[src] * minus_state[dst]) * inv2β
        end
    end

    grad_b = gradient.b
    @inbounds for idx in model.hidden1_idxs
        grad_b[idx] += (plus_state[idx] - minus_state[idx]) * inv2β
    end
    @inbounds for idx in model.hidden2_idxs
        grad_b[idx] += (plus_state[idx] - minus_state[idx]) * inv2β
    end
    if config.train_output_bias
        @inbounds for idx in model.output_idxs
            grad_b[idx] += (plus_state[idx] - minus_state[idx]) * inv2β
        end
    end
    return gradient
end

"""Compute stats from the free state after observable gradients were sampled."""
function finish_symmetric_observable_sample!(
    model::M,
    y::Y,
    free_state::F,
) where {M<:LocalMNISTModel,Y<:AbstractVector,F<:AbstractVector}
    config = model.config
    free_o = @view free_state[model.output_idxs]
    correct = argmax(class_scores(free_o, config.output_replicas)) == argmax(class_scores(y, config.output_replicas))
    loss = sum(abs2, y .- free_o) / 2
    return (; loss, correct, skipped = false)
end

"""Compute stats from the free state and gradients from symmetric nudges."""
function finish_symmetric_contrastive_sample!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    free_state::F,
    plus_state::P,
    minus_state::N,
) where {G<:NamedTuple,M<:LocalMNISTModel,X<:AbstractVector,Y<:AbstractVector,F<:AbstractVector,P<:AbstractVector,N<:AbstractVector}
    config = model.config
    free_o = @view free_state[model.output_idxs]
    correct = argmax(class_scores(free_o, config.output_replicas)) == argmax(class_scores(y, config.output_replicas))
    loss = sum(abs2, y .- free_o) / 2
    if all(free_o .== y)
        return (; loss, correct, skipped = true)
    end
    accumulate_local_symmetric_contrastive_gradient!(gradient, model, x, plus_state, minus_state)
    return (; loss, correct, skipped = false)
end

# Finish one observable-symmetric sample after phase samplers updated gradients.
StatefulAlgorithms.@ProcessAlgorithm function FinishSymmetricObservableSample!(
    learning_model,
    y::AbstractVector,
    free_state::AbstractVector,
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    nskipped::Base.RefValue,
    total_loss::Base.RefValue,
    @managed(loss::Float32 = 0f0),
    @managed(correct::Bool = false),
    @managed(skipped::Bool = false),
)
    stats = finish_symmetric_observable_sample!(learning_model, y, free_state)
    loss = Float32(stats.loss)
    correct = Bool(stats.correct)
    skipped = Bool(stats.skipped)

    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    nskipped[] += skipped ? 1 : 0
    total_loss[] += loss
    return (; loss, correct, skipped)
end

# Finish one symmetric contrastive sample and update worker-local stats.
StatefulAlgorithms.@ProcessAlgorithm function FinishSymmetricContrastiveSample!(
    gradient::NamedTuple,
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    free_state::AbstractVector,
    plus_state::AbstractVector,
    minus_state::AbstractVector,
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    nskipped::Base.RefValue,
    total_loss::Base.RefValue,
    @managed(loss::Float32 = 0f0),
    @managed(correct::Bool = false),
    @managed(skipped::Bool = false),
)
    stats = finish_symmetric_contrastive_sample!(gradient, learning_model, x, y, free_state, plus_state, minus_state)
    loss = Float32(stats.loss)
    correct = Bool(stats.correct)
    skipped = Bool(stats.skipped)

    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    nskipped[] += skipped ? 1 : 0
    total_loss[] += loss
    return (; loss, correct, skipped)
end

"""Build a free phase that returns a time-averaged full state."""
function stateavg_free_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
    burnin_sweeps::J,
    average_sweeps::K,
    sample_every_sweeps::L,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer,K<:Integer,L<:Integer}
    burnin_steps = max(1, Int(burnin_sweeps) * Int(nstates))
    sample_interval = max(1, Int(sample_every_sweeps) * Int(nstates))
    average_count = max(1, cld(Int(average_sweeps), Int(sample_every_sweeps)))
    burnin_temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = burnin_steps)
    burnin_step = free_phase_step_algorithm(dynamics_algorithm, burnin_temperature)
    measurement = stateavg_measurement_block(dynamics_algorithm, nstates, sample_interval, :free_state_averager)
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias burnin_step = burnin_step
        @alias measurement = measurement
        @state mnist_model
        @state x
        @state base_bias
        @state sample_buffer
        @state free_state
        @state rng

        RandomizeGraphState!(dynamics.model, rng)
        InstallSampleBias!(mnist_model, x, base_bias, sample_buffer)
        @repeat burnin_steps burnin_step()
        @context avg_ctx = @repeat average_count measurement()
        WriteAverageState!(free_state, avg_ctx.free_state_averager.sum, avg_ctx.free_state_averager.count)
    end
end

"""Build a positive nudged phase that returns a time-averaged full state."""
function stateavg_nudged_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
    burnin_sweeps::J,
    average_sweeps::K,
    sample_every_sweeps::L,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer,K<:Integer,L<:Integer}
    burnin_steps = max(1, Int(burnin_sweeps) * Int(nstates))
    sample_interval = max(1, Int(sample_every_sweeps) * Int(nstates))
    average_count = max(1, cld(Int(average_sweeps), Int(sample_every_sweeps)))
    burnin_temperature = ReverseAnnealDynamicsTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = burnin_steps)
    burnin_step = nudged_phase_step_algorithm(dynamics_algorithm, burnin_temperature)
    measurement = stateavg_measurement_block(dynamics_algorithm, nstates, sample_interval, :plus_state_averager)
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias burnin_step = burnin_step
        @alias measurement = measurement
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state free_state
        @state nudged_state

        SetGraphState!(dynamics.model, free_state)
        InstallPositiveNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
        @repeat burnin_steps burnin_step()
        @context avg_ctx = @repeat average_count measurement()
        WriteAverageState!(nudged_state, avg_ctx.plus_state_averager.sum, avg_ctx.plus_state_averager.count)
    end
end

"""Build a nudged phase that samples true edge/bias observables."""
function observableavg_nudged_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
    burnin_sweeps::J,
    average_sweeps::K,
    sample_every_sweeps::L,
    beta_sign::T,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer,K<:Integer,L<:Integer,T<:Real}
    burnin_steps = max(1, Int(burnin_sweeps) * Int(nstates))
    sample_interval = max(1, Int(sample_every_sweeps) * Int(nstates))
    average_count = max(1, cld(Int(average_sweeps), Int(sample_every_sweeps)))
    signed_scale = PMNIST_FT(beta_sign) / (2f0 * config.β * PMNIST_FT(average_count))
    tangent_nudge = parse(Bool, lowercase(get(ENV, "ISING_MNIST_TANGENT_NUDGE", "false")))
    burnin_temperature = ReverseAnnealDynamicsTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = burnin_steps)
    burnin_step = nudged_phase_step_algorithm(dynamics_algorithm, burnin_temperature)
    if beta_sign > 0
        observable_step = StatefulAlgorithms.@Routine begin
            @alias dynamics = dynamics_algorithm
            @state gradient
            @state mnist_model
            @state x
            @repeat sample_interval dynamics()
            @alias plus_observable_sampler = AccumulateGradientObservables!(; scale = signed_scale)
            plus_observable_sampler(gradient, mnist_model, x, dynamics.model)
        end
        if tangent_nudge
            return StatefulAlgorithms.@Routine begin
                @alias dynamics = dynamics_algorithm
                @alias burnin_step = burnin_step
                @alias plus_observable_step = observable_step
                @state mnist_model
                @state x
                @state y
                @state base_bias
                @state sample_buffer
                @state free_state

                SetGraphState!(dynamics.model, free_state)
                InstallPositiveTangentNudgedSampleBias!(mnist_model, x, y, free_state, base_bias, sample_buffer)
                @repeat burnin_steps burnin_step()
                @repeat average_count plus_observable_step()
            end
        else
            return StatefulAlgorithms.@Routine begin
                @alias dynamics = dynamics_algorithm
                @alias burnin_step = burnin_step
                @alias plus_observable_step = observable_step
                @state mnist_model
                @state x
                @state y
                @state base_bias
                @state sample_buffer
                @state free_state

                SetGraphState!(dynamics.model, free_state)
                InstallPositiveNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
                @repeat burnin_steps burnin_step()
                @repeat average_count plus_observable_step()
            end
        end
    else
        observable_step = StatefulAlgorithms.@Routine begin
            @alias dynamics = dynamics_algorithm
            @state gradient
            @state mnist_model
            @state x
            @repeat sample_interval dynamics()
            @alias minus_observable_sampler = AccumulateGradientObservables!(; scale = signed_scale)
            minus_observable_sampler(gradient, mnist_model, x, dynamics.model)
        end
        if tangent_nudge
            return StatefulAlgorithms.@Routine begin
                @alias dynamics = dynamics_algorithm
                @alias burnin_step = burnin_step
                @alias minus_observable_step = observable_step
                @state mnist_model
                @state x
                @state y
                @state base_bias
                @state sample_buffer
                @state free_state

                SetGraphState!(dynamics.model, free_state)
                InstallNegativeTangentNudgedSampleBias!(mnist_model, x, y, free_state, base_bias, sample_buffer)
                @repeat burnin_steps burnin_step()
                @repeat average_count minus_observable_step()
            end
        else
            return StatefulAlgorithms.@Routine begin
                @alias dynamics = dynamics_algorithm
                @alias burnin_step = burnin_step
                @alias minus_observable_step = observable_step
                @state mnist_model
                @state x
                @state y
                @state base_bias
                @state sample_buffer
                @state free_state

                SetGraphState!(dynamics.model, free_state)
                InstallNegativeNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
                @repeat burnin_steps burnin_step()
                @repeat average_count minus_observable_step()
            end
        end
    end
end

"""Build a negative nudged phase for the symmetric state-averaged estimator."""
function stateavg_negative_nudged_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
    burnin_sweeps::J,
    average_sweeps::K,
    sample_every_sweeps::L,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer,K<:Integer,L<:Integer}
    burnin_steps = max(1, Int(burnin_sweeps) * Int(nstates))
    sample_interval = max(1, Int(sample_every_sweeps) * Int(nstates))
    average_count = max(1, cld(Int(average_sweeps), Int(sample_every_sweeps)))
    burnin_temperature = ReverseAnnealDynamicsTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = burnin_steps)
    burnin_step = nudged_phase_step_algorithm(dynamics_algorithm, burnin_temperature)
    measurement = stateavg_measurement_block(dynamics_algorithm, nstates, sample_interval, :minus_state_averager)
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias burnin_step = burnin_step
        @alias measurement = measurement
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state free_state

        SetGraphState!(dynamics.model, free_state)
        InstallNegativeNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
        @repeat burnin_steps burnin_step()
        @context avg_ctx = @repeat average_count measurement()
        WriteAverageState!(sample_buffer, avg_ctx.minus_state_averager.sum, avg_ctx.minus_state_averager.count)
    end
end

"""Build a symmetric low-beta worker that averages gradient observables."""
function observableavg_symmetric_contrastive_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    default_burnin_sweeps = get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps))
    default_average_sweeps = get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10")
    free_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_BURNIN_SWEEPS", default_burnin_sweeps))
    free_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_AVERAGE_SWEEPS", default_average_sweeps))
    nudge_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_BURNIN_SWEEPS", default_burnin_sweeps))
    nudge_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_AVERAGE_SWEEPS", default_average_sweeps))
    sample_every_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS", "1"))
    free_phase = stateavg_free_phase_algorithm(dynamics_algorithm, config, nstates, free_burnin_sweeps, free_average_sweeps, sample_every_sweeps)
    plus_phase = observableavg_nudged_phase_algorithm(dynamics_algorithm, config, nstates, nudge_burnin_sweeps, nudge_average_sweeps, sample_every_sweeps, 1f0)
    minus_phase = observableavg_nudged_phase_algorithm(dynamics_algorithm, config, nstates, nudge_burnin_sweeps, nudge_average_sweeps, sample_every_sweeps, -1f0)
    return StatefulAlgorithms.@Routine begin
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state gradient
        @state free_state
        @state rng
        @state nsamples
        @state ncorrect
        @state nskipped
        @state total_loss
        @alias free_phase = free_phase
        @alias plus_phase = plus_phase
        @alias minus_phase = minus_phase

        @context free_phase_ctx = free_phase()
        @bind mnist_model => free_phase_ctx.mnist_model
        @bind x => free_phase_ctx.x
        @bind base_bias => free_phase_ctx.base_bias
        @bind sample_buffer => free_phase_ctx.sample_buffer
        @bind free_state => free_phase_ctx.free_state
        @bind rng => free_phase_ctx.rng

        @context plus_phase_ctx = plus_phase()
        @bind mnist_model => plus_phase_ctx.mnist_model
        @bind x => plus_phase_ctx.x
        @bind y => plus_phase_ctx.y
        @bind base_bias => plus_phase_ctx.base_bias
        @bind sample_buffer => plus_phase_ctx.sample_buffer
        @bind free_state => plus_phase_ctx.free_state
        @bind gradient => plus_phase_ctx.gradient

        @context minus_phase_ctx = minus_phase()
        @bind gradient => minus_phase_ctx.gradient

        FinishSymmetricObservableSample!(
            mnist_model,
            y,
            free_state,
            nsamples,
            ncorrect,
            nskipped,
            total_loss,
        )
    end
end

"""Build a one-sided low-beta EqProp worker using full-state time averages."""
function stateavg_contrastive_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    default_burnin_sweeps = get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps))
    default_average_sweeps = get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10")
    free_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_BURNIN_SWEEPS", default_burnin_sweeps))
    free_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_AVERAGE_SWEEPS", default_average_sweeps))
    nudge_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_BURNIN_SWEEPS", default_burnin_sweeps))
    nudge_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_AVERAGE_SWEEPS", default_average_sweeps))
    sample_every_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS", "1"))
    free_phase = stateavg_free_phase_algorithm(dynamics_algorithm, config, nstates, free_burnin_sweeps, free_average_sweeps, sample_every_sweeps)
    nudged_phase = stateavg_nudged_phase_algorithm(dynamics_algorithm, config, nstates, nudge_burnin_sweeps, nudge_average_sweeps, sample_every_sweeps)
    return StatefulAlgorithms.@Routine begin
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state gradient
        @state free_state
        @state nudged_state
        @state rng
        @state nsamples
        @state ncorrect
        @state nskipped
        @state total_loss
        @alias free_phase = free_phase
        @alias nudged_phase = nudged_phase

        @context free_phase_ctx = free_phase()
        @bind mnist_model => free_phase_ctx.mnist_model
        @bind x => free_phase_ctx.x
        @bind base_bias => free_phase_ctx.base_bias
        @bind sample_buffer => free_phase_ctx.sample_buffer
        @bind free_state => free_phase_ctx.free_state
        @bind rng => free_phase_ctx.rng

        @context nudged_phase_ctx = nudged_phase()
        @bind mnist_model => nudged_phase_ctx.mnist_model
        @bind x => nudged_phase_ctx.x
        @bind y => nudged_phase_ctx.y
        @bind base_bias => nudged_phase_ctx.base_bias
        @bind sample_buffer => nudged_phase_ctx.sample_buffer
        @bind free_state => nudged_phase_ctx.free_state
        @bind nudged_state => nudged_phase_ctx.nudged_state

        FinishContrastiveSample!(
            gradient,
            mnist_model,
            x,
            y,
            free_state,
            nudged_state,
            nsamples,
            ncorrect,
            nskipped,
            total_loss,
        )
    end
end

"""Build a symmetric low-beta EqProp worker using full-state time averages."""
function stateavg_symmetric_contrastive_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    default_burnin_sweeps = get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps))
    default_average_sweeps = get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10")
    free_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_BURNIN_SWEEPS", default_burnin_sweeps))
    free_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_FREE_AVERAGE_SWEEPS", default_average_sweeps))
    nudge_burnin_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_BURNIN_SWEEPS", default_burnin_sweeps))
    nudge_average_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_NUDGE_AVERAGE_SWEEPS", default_average_sweeps))
    sample_every_sweeps = parse(Int, get(ENV, "ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS", "1"))
    free_phase = stateavg_free_phase_algorithm(dynamics_algorithm, config, nstates, free_burnin_sweeps, free_average_sweeps, sample_every_sweeps)
    plus_phase = stateavg_nudged_phase_algorithm(dynamics_algorithm, config, nstates, nudge_burnin_sweeps, nudge_average_sweeps, sample_every_sweeps)
    minus_phase = stateavg_negative_nudged_phase_algorithm(dynamics_algorithm, config, nstates, nudge_burnin_sweeps, nudge_average_sweeps, sample_every_sweeps)
    return StatefulAlgorithms.@Routine begin
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state gradient
        @state free_state
        @state nudged_state
        @state rng
        @state nsamples
        @state ncorrect
        @state nskipped
        @state total_loss
        @alias free_phase = free_phase
        @alias plus_phase = plus_phase
        @alias minus_phase = minus_phase

        @context free_phase_ctx = free_phase()
        @bind mnist_model => free_phase_ctx.mnist_model
        @bind x => free_phase_ctx.x
        @bind base_bias => free_phase_ctx.base_bias
        @bind sample_buffer => free_phase_ctx.sample_buffer
        @bind free_state => free_phase_ctx.free_state
        @bind rng => free_phase_ctx.rng

        @context plus_phase_ctx = plus_phase()
        @bind mnist_model => plus_phase_ctx.mnist_model
        @bind x => plus_phase_ctx.x
        @bind y => plus_phase_ctx.y
        @bind base_bias => plus_phase_ctx.base_bias
        @bind sample_buffer => plus_phase_ctx.sample_buffer
        @bind free_state => plus_phase_ctx.free_state
        @bind nudged_state => plus_phase_ctx.nudged_state

        @context minus_phase_ctx = minus_phase()
        @bind sample_buffer => minus_phase_ctx.sample_buffer

        FinishSymmetricContrastiveSample!(
            gradient,
            mnist_model,
            x,
            y,
            free_state,
            nudged_state,
            sample_buffer,
            nsamples,
            ncorrect,
            nskipped,
            total_loss,
        )
    end
end

"""Select one-sided or symmetric state-averaged worker construction."""
function stateavg_worker_algorithm(dynamics_algorithm::D, config::C, nstates::I) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    estimator = lowercase(get(ENV, "ISING_MNIST_STATEAVG_ESTIMATOR", "one_sided"))
    if estimator in ("observable_symmetric", "observables", "observable")
        return observableavg_symmetric_contrastive_worker_algorithm(dynamics_algorithm, config, nstates)
    elseif estimator in ("symmetric", "plus_minus", "pm")
        return stateavg_symmetric_contrastive_worker_algorithm(dynamics_algorithm, config, nstates)
    elseif estimator in ("one_sided", "onesided")
        return stateavg_contrastive_worker_algorithm(dynamics_algorithm, config, nstates)
    end
    throw(ArgumentError("unknown ISING_MNIST_STATEAVG_ESTIMATOR=`$estimator`"))
end

"""Create the diagnostic manager with state-averaged worker dynamics."""
function stateavg_local_manager(source::M) where {M<:LocalMNISTModel}
    t_manager = time()
    progress_log(source.config, "stateavg manager construction started"; workers = source.config.workers)
    params = trainable_params(source)
    state = LocalMNISTManagerState(
        source,
        gradient_buffer(source),
        gradient_buffer(source),
        Ref(params),
        optimizer_states(source.config, params),
        Ref(zeros(PMNIST_FT, PMNIST_INPUT_DIM, 0)),
        Ref(zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas, 0)),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0f0),
    )
    dynamics_algorithm = mnist_dynamics_algorithm()
    t_resolve = time()
    worker_algorithm = StatefulAlgorithms.resolve(stateavg_worker_algorithm(deepcopy(dynamics_algorithm), source.config, length(II.state(source.graph))))
    progress_log(source.config, "stateavg worker algorithm resolved"; t0 = t_resolve)
    recipe = (;
        makeworker = (idx, manager) -> local_worker(manager.state.model, idx, worker_algorithm),
        loadjob! = load_training_index_job!,
        close! = close_process_worker!,
        sync_to_state! = manager -> flush_manager_buffers!(manager),
    )
    manager = StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = source.config.workers,
        config = source.config,
        state,
        sync_policy = StatefulAlgorithms.SyncAtEnd(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        execution = StatefulAlgorithms.ChannelWorkers(),
        poll_interval = 0.0,
        job_type = Int,
    )
    progress_log(source.config, "stateavg manager constructed"; t0 = t_manager, workers = source.config.workers)
    return manager
end

# Override the training manager only in this diagnostic script.
local_manager(source::M) where {M<:LocalMNISTModel} = stateavg_local_manager(source)

"""Run the low-beta state-averaged diagnostic training configuration."""
function main()
    config = LocalMNISTManagerConfig()
    result = run_config!(config)
    open(joinpath(config.outdir, "stateavg_training_settings.md"), "w") do io
        println(io, "# Low-Beta State-Averaged Training")
        println(io)
        println(io, "- estimator: `$(get(ENV, "ISING_MNIST_STATEAVG_ESTIMATOR", "one_sided"))`")
        println(io, "- tangent nudge: `$(get(ENV, "ISING_MNIST_TANGENT_NUDGE", "false"))`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- burn-in sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps)))`")
        println(io, "- averaging sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10"))`")
        println(io, "- free burn-in sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_FREE_BURNIN_SWEEPS", get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps))))`")
        println(io, "- free averaging sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_FREE_AVERAGE_SWEEPS", get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10")))`")
        println(io, "- nudged burn-in sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_NUDGE_BURNIN_SWEEPS", get(ENV, "ISING_MNIST_STATEAVG_BURNIN_SWEEPS", string(config.free_sweeps))))`")
        println(io, "- nudged averaging sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_NUDGE_AVERAGE_SWEEPS", get(ENV, "ISING_MNIST_STATEAVG_AVERAGE_SWEEPS", "10")))`")
        println(io, "- sample every sweeps: `$(get(ENV, "ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS", "1"))`")
        println(io, "- source script: `low_beta_stateavg_training.jl`")
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
