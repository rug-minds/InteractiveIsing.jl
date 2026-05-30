export SetGraphState!, CopyGraphState!, RandomizeGraphState!,
       GeometricTemperatureSchedule, ReverseAnnealTemperatureSchedule,
       GeometricDynamicsTemperatureSchedule, ReverseAnnealDynamicsTemperatureSchedule,
       ResetRef!, ResetBestEnergyCapture!, CaptureBestEnergyState!,
       CaptureBestEnergyStateFromHamiltonian!, graph_energy,
       InstallSampleBias!, InstallNudgedSampleBias!, FinishContrastiveSample!,
       FinishValidationSample!, install_sample_bias!, install_nudged_sample_bias!,
       finish_contrastive_sample!, finish_validation_sample!,
       free_phase_step_algorithm, nudged_phase_step_algorithm,
       free_phase_algorithm, nudged_phase_algorithm,
       contrastive_worker_algorithm, validation_free_phase_algorithm

"""Install a free-phase sample field for a model-specific learning process."""
function install_sample_bias! end

"""Install a nudged sample field for a model-specific learning process."""
function install_nudged_sample_bias! end

"""Finish one model-specific contrastive sample and return loss/correct/skipped stats."""
function finish_contrastive_sample! end

"""Finish one model-specific validation sample and return loss/correct stats."""
function finish_validation_sample! end

# Process tool that writes a stored state vector into an Ising graph.
@ProcessAlgorithm function SetGraphState!(isinggraph::AbstractIsingGraph, target::AbstractVector)
    state(isinggraph) .= target
    return nothing
end

# Process tool that copies the current graph state into a reusable buffer.
@ProcessAlgorithm function CopyGraphState!(dest::AbstractVector, isinggraph::AbstractIsingGraph)
    dest .= state(isinggraph)
    return nothing
end

# Process tool that initializes every active graph state to a random Ising sign.
@ProcessAlgorithm function RandomizeGraphState!(isinggraph::AbstractIsingGraph, rng::AbstractRNG)
    s = state(isinggraph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(rng, Bool) ? one(eltype(s)) : -one(eltype(s))
    end
    return nothing
end

# Process tool that resets a reusable scalar `Ref` owned by another algorithm.
@ProcessAlgorithm function ResetRef!(target::Base.RefValue, value)
    target[] = convert(typeof(target[]), value)
    return nothing
end

"""Return the graph energy for Bilinear + MagField Ising graphs."""
@inline function graph_energy(isinggraph::G) where {T<:AbstractFloat,G<:AbstractIsingGraph{T}}
    return graph_energy(isinggraph, isinggraph.hamiltonian)
end

"""Return graph energy using an already-initialized concrete Hamiltonian."""
@inline function graph_energy(isinggraph::G, hamiltonian::H) where {T<:AbstractFloat,G<:AbstractIsingGraph{T},H}
    s = state(isinggraph)
    A = adj(isinggraph)
    colptrs = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    nzvals = SparseArrays.nonzeros(A)
    energy = zero(T)
    @inbounds for col in 1:size(A, 2)
        for ptr in colptrs[col]:(colptrs[col + 1] - 1)
            energy -= T(0.5) * nzvals[ptr] * s[rowvals[ptr]] * s[col]
        end
    end
    for hterm in InteractiveIsing.hamiltonians(hamiltonian)
        hterm isa InteractiveIsing.MagField || continue
        b = hterm.b
        @inbounds for idx in eachindex(s)
            energy -= hterm.c * b[idx] * s[idx]
        end
    end
    return energy
end

"""
    GeometricTemperatureSchedule(; start_T, stop_T)

Process scheduler that writes a geometric temperature schedule into a graph.
The schedule counter is ordinary managed state so it can stay scalarized.
"""
@ProcessAlgorithm begin
    @config start_T::Float32 = 5f0
    @config stop_T::Float32 = 1f-2
    @config n_steps::Int = 1

    function GeometricTemperatureSchedule(
        isinggraph::AbstractIsingGraph,
        @managed(step_idx::Int = 0),
        @managed(total_steps::Int = n_steps),
        @managed(current_T::Float32 = start_T),
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = start_T * (stop_T / start_T)^progress
        InteractiveIsing.temp!(isinggraph, current_T)
        next_step = step_idx >= total - 1 ? 0 : step_idx + 1
        return (; step_idx = next_step, current_T)
    end
end

"""
    ReverseAnnealTemperatureSchedule(; cold_T, peak_T)

Process scheduler that warms from `cold_T` to `peak_T`, then cools back down.
The schedule counter is ordinary managed state so it can stay scalarized.
"""
@ProcessAlgorithm begin
    @config cold_T::Float32 = 1f-2
    @config peak_T::Float32 = 1f0
    @config n_steps::Int = 1

    function ReverseAnnealTemperatureSchedule(
        isinggraph::AbstractIsingGraph,
        @managed(step_idx::Int = 0),
        @managed(total_steps::Int = n_steps),
        @managed(current_T::Float32 = cold_T),
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = if progress <= 0.5f0
            cold_T + (progress / 0.5f0) * (peak_T - cold_T)
        else
            peak_T + ((progress - 0.5f0) / 0.5f0) * (cold_T - peak_T)
        end
        InteractiveIsing.temp!(isinggraph, current_T)
        next_step = step_idx >= total - 1 ? 0 : step_idx + 1
        return (; step_idx = next_step, current_T)
    end
end

"""
    GeometricDynamicsTemperatureSchedule(; start_T, stop_T)

Process scheduler that writes a geometric temperature schedule into the
dynamics context's scalar `T` instead of mutating `isinggraph.temp`.
"""
@ProcessAlgorithm begin
    @config start_T::Float32 = 5f0
    @config stop_T::Float32 = 1f-2
    @config n_steps::Int = 1

    function GeometricDynamicsTemperatureSchedule(
        T::AbstractFloat,
        @managed(step_idx::Int = 0),
        @managed(total_steps::Int = n_steps),
        @managed(current_T::Float32 = start_T),
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = start_T * (stop_T / start_T)^progress
        next_step = step_idx >= total - 1 ? 0 : step_idx + 1
        return (; step_idx = next_step, T = typeof(T)(current_T), current_T)
    end
end

"""
    ReverseAnnealDynamicsTemperatureSchedule(; cold_T, peak_T)

Process scheduler that reverse-anneals the dynamics context's scalar `T`.
"""
@ProcessAlgorithm begin
    @config cold_T::Float32 = 1f-2
    @config peak_T::Float32 = 1f0
    @config n_steps::Int = 1

    function ReverseAnnealDynamicsTemperatureSchedule(
        T::AbstractFloat,
        @managed(step_idx::Int = 0),
        @managed(total_steps::Int = n_steps),
        @managed(current_T::Float32 = cold_T),
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = if progress <= 0.5f0
            cold_T + (progress / 0.5f0) * (peak_T - cold_T)
        else
            peak_T + ((progress - 0.5f0) / 0.5f0) * (cold_T - peak_T)
        end
        next_step = step_idx >= total - 1 ? 0 : step_idx + 1
        return (; step_idx = next_step, T = typeof(T)(current_T), current_T)
    end
end

# Process tool that prepares a reusable best-energy state buffer.
@ProcessAlgorithm function ResetBestEnergyCapture!(
    best_energy::Base.RefValue,
    best_state::AbstractVector,
)
    best_energy[] = convert(typeof(best_energy[]), Inf)
    fill!(best_state, zero(eltype(best_state)))
    return nothing
end

# Process tool that stores the current graph state when its energy improves.
@ProcessAlgorithm function CaptureBestEnergyState!(
    isinggraph::AbstractIsingGraph,
    best_energy::Base.RefValue,
    best_state::AbstractVector,
)
    energy = graph_energy(isinggraph)
    if energy < best_energy[]
        best_energy[] = energy
        best_state .= state(isinggraph)
    end
    return nothing
end

# Process tool that uses a dynamics-owned initialized Hamiltonian for capture.
@ProcessAlgorithm function CaptureBestEnergyStateFromHamiltonian!(
    isinggraph::AbstractIsingGraph,
    hamiltonian,
    best_energy::Base.RefValue,
    best_state::AbstractVector,
)
    energy = graph_energy(isinggraph, hamiltonian)
    if energy < best_energy[]
        best_energy[] = energy
        best_state .= state(isinggraph)
    end
    return nothing
end

# Process tool that delegates model-specific free-phase sample-field writes.
@ProcessAlgorithm function InstallSampleBias!(
    learning_model,
    x::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_sample_bias!(learning_model, x, base_bias, sample_buffer)
    return nothing
end

# Process tool that delegates model-specific nudged sample-field writes.
@ProcessAlgorithm function InstallNudgedSampleBias!(
    learning_model,
    x::AbstractVector, 
    y::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    install_nudged_sample_bias!(learning_model, x, y, base_bias, sample_buffer)
    return nothing
end

# Process tool that finishes one contrastive sample and updates worker-local stats.
@ProcessAlgorithm function FinishContrastiveSample!(
    gradient::NamedTuple,
    learning_model,
    x::AbstractVector,
    y::AbstractVector,
    free_state::AbstractVector,
    nudged_state::AbstractVector,
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    nskipped::Base.RefValue,
    total_loss::Base.RefValue,
    @managed(loss::Float32 = 0f0),
    @managed(correct::Bool = false),
    @managed(skipped::Bool = false),
)
    stats = finish_contrastive_sample!(gradient, learning_model, x, y, free_state, nudged_state)
    loss = Float32(stats.loss)
    correct = Bool(stats.correct)
    skipped = Bool(stats.skipped)

    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    nskipped[] += skipped ? 1 : 0
    total_loss[] += loss
    return (; loss, correct, skipped)
end

# Process tool that finishes one validation sample and updates worker-local stats.
@ProcessAlgorithm function FinishValidationSample!(
    learning_model,
    y::AbstractVector,
    free_state::AbstractVector,
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    total_loss::Base.RefValue,
    pred_counts::AbstractVector{Int},
    @managed(loss::Float32 = 0f0),
    @managed(correct::Bool = false),
)
    stats = finish_validation_sample!(learning_model, y, free_state, pred_counts)
    loss = Float32(stats.loss)
    correct = Bool(stats.correct)

    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    total_loss[] += loss
    return (; loss, correct)
end

"""Build one reusable free-phase temperature-scheduled dynamics step."""
function free_phase_step_algorithm(dynamics_algorithm::D, temperature_algorithm::T) where {D,T}
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias free_temperature = temperature_algorithm

        free_temperature(dynamics.T)
        dynamics()
    end
end

"""Build one reusable nudged-phase temperature-scheduled dynamics step."""
function nudged_phase_step_algorithm(dynamics_algorithm::D, temperature_algorithm::T) where {D,T}
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias nudge_temperature = temperature_algorithm

        nudge_temperature(dynamics.T)
        dynamics()
    end
end

"""Build a reusable free phase around model-specific sample-field installation."""
function free_phase_algorithm(
    dynamics_algorithm::D,
    temperature_algorithm::T,
    steps::I,
) where {D,T,I<:Integer}
    phase_step = free_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias phase_step = phase_step
        @state mnist_model
        @state x
        @state base_bias
        @state sample_buffer
        @state free_state
        @state free_best_energy
        @state rng

        RandomizeGraphState!(dynamics.model, rng)
        InstallSampleBias!(mnist_model, x, base_bias, sample_buffer)
        @repeat steps phase_step()
        CaptureBestEnergyStateFromHamiltonian!(dynamics.model, dynamics.hamiltonian, free_best_energy, free_state)
    end
end

"""Build a reusable nudged phase around model-specific target-field installation."""
function nudged_phase_algorithm(
    dynamics_algorithm::D,
    temperature_algorithm::T,
    steps::I,
) where {D,T,I<:Integer}
    phase_step = nudged_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias phase_step = phase_step
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state free_state
        @state nudged_state
        @state nudged_best_energy

        SetGraphState!(dynamics.model, free_state)
        InstallNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
        @repeat steps phase_step()
        CaptureBestEnergyStateFromHamiltonian!(dynamics.model, dynamics.hamiltonian, nudged_best_energy, nudged_state)
    end
end

"""Build a reusable contrastive worker algorithm from free/nudged phase settings."""
function contrastive_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C,I<:Integer}
    free_steps = max(1, config.free_sweeps * Int(nstates))
    nudge_steps = max(1, config.nudge_sweeps * Int(nstates))
    free_reads = max(1, config.free_reads)
    nudge_reads = max(1, config.nudge_reads)
    free_temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    nudge_temperature = ReverseAnnealDynamicsTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = nudge_steps)
    free_phase = free_phase_algorithm(dynamics_algorithm, free_temperature, free_steps)
    nudged_phase = nudged_phase_algorithm(dynamics_algorithm, nudge_temperature, nudge_steps)
    return Processes.@Routine begin
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state gradient
        @state free_state
        @state nudged_state
        @state free_best_energy
        @state nudged_best_energy
        @state rng
        @state nsamples
        @state ncorrect
        @state nskipped
        @state total_loss

        ResetBestEnergyCapture!(free_best_energy, free_state)
        @context free_phase_ctx = @repeat free_reads free_phase()
        @bind mnist_model => free_phase_ctx.mnist_model
        @bind x => free_phase_ctx.x
        @bind base_bias => free_phase_ctx.base_bias
        @bind sample_buffer => free_phase_ctx.sample_buffer
        @bind free_state => free_phase_ctx.free_state
        @bind free_best_energy => free_phase_ctx.free_best_energy
        @bind rng => free_phase_ctx.rng

        ResetBestEnergyCapture!(nudged_best_energy, nudged_state)
        @context nudged_phase_ctx = @repeat nudge_reads nudged_phase()
        @bind mnist_model => nudged_phase_ctx.mnist_model
        @bind x => nudged_phase_ctx.x
        @bind y => nudged_phase_ctx.y
        @bind base_bias => nudged_phase_ctx.base_bias
        @bind sample_buffer => nudged_phase_ctx.sample_buffer
        @bind free_state => nudged_phase_ctx.free_state
        @bind nudged_state => nudged_phase_ctx.nudged_state
        @bind nudged_best_energy => nudged_phase_ctx.nudged_best_energy

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

"""Build a reusable free-phase validation worker algorithm."""
function validation_free_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C,I<:Integer}
    free_steps = max(1, config.free_sweeps * Int(nstates))
    free_reads = max(1, config.free_reads)
    free_temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    free_phase = free_phase_algorithm(dynamics_algorithm, free_temperature, free_steps)
    return Processes.@Routine begin
        @state mnist_model
        @state x
        @state y
        @state base_bias
        @state sample_buffer
        @state free_state
        @state free_best_energy
        @state rng
        @state nsamples
        @state ncorrect
        @state total_loss
        @state pred_counts

        ResetBestEnergyCapture!(free_best_energy, free_state)
        @context free_phase_ctx = @repeat free_reads free_phase()
        @bind mnist_model => free_phase_ctx.mnist_model
        @bind x => free_phase_ctx.x
        @bind base_bias => free_phase_ctx.base_bias
        @bind sample_buffer => free_phase_ctx.sample_buffer
        @bind free_state => free_phase_ctx.free_state
        @bind free_best_energy => free_phase_ctx.free_best_energy
        @bind rng => free_phase_ctx.rng

        FinishValidationSample!(
            mnist_model,
            y,
            free_state,
            nsamples,
            ncorrect,
            total_loss,
            pred_counts,
        )
    end
end
