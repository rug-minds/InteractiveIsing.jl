using Dates
using Random
using SparseArrays

const SINGLE_FIELD_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SINGLE_FIELD_MANAGER_FILE = joinpath(SINGLE_FIELD_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "single_field_bias_timing"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = "1"
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")

include(SINGLE_FIELD_MANAGER_FILE)

"""Print one timestamped single-field diagnostic line."""
function single_field_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Create a normal Process from an algorithm and init specs."""
function normal_process(algorithm::A, inits::Vararg{Any,N}) where {A,N}
    return Processes.Process(algorithm, inits...; repeat = 1)
end

"""Run a Process through the asynchronous `run`/`wait` path."""
@inline function run_async_once!(process::P) where {P<:Processes.Process}
    Processes.reset!(process)
    @inline run(process)
    @inline wait(process)
    return process
end

"""Build a sampled graph with a single writable magnetic-field term."""
function single_field_sampled_graph(
    config::C,
    rng::R;
    shared_adj,
    combined_bias::B,
) where {C<:LocalMNISTManagerConfig,R<:Random.AbstractRNG,B<:AbstractVector}
    output_rows, output_cols = factor_shape(PMNIST_NCLASSES * config.output_replicas)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    input = II.Layer(PMNIST_INPUT_SIDE, PMNIST_INPUT_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    h1 = II.Layer(config.hidden1_side, config.hidden1_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + 2, 0); periodic = false)
    h2 = II.Layer(config.hidden2_side, config.hidden2_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + config.hidden1_side + 4, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + config.hidden1_side + config.hidden2_side + 6, 0); periodic = false)
    graph = II.IsingGraph(
        input,
        zero_wg,
        h1,
        zero_wg,
        h2,
        zero_wg,
        out,
        LocalMNISTFlipProposer(),
        II.Bilinear() + II.MagField(b = II.Force(combined_bias));
        precision = PMNIST_FT,
        adj = shared_adj,
        index_set = local_mnist_active_indices,
    )
    II.temp!(graph, config.cold_temp)
    return graph
end

"""Create a worker model whose Hamiltonian has one combined bias field."""
function single_field_worker_model(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    rng = Random.MersenneTwister(source.config.seed + 30_000 + Int(worker_idx))
    combined_bias = copy(base_magfield(source.graph).b)
    graph = single_field_sampled_graph(source.config, rng; shared_adj = II.adj(source.graph), combined_bias)
    return LocalMNISTModel(
        source.config,
        graph,
        source.edge_layout,
        collect(II.layerrange(graph[1])),
        collect(II.layerrange(graph[2])),
        collect(II.layerrange(graph[3])),
        collect(II.layerrange(graph[4])),
        rng,
    )
end

"""Write `base_bias + clipped(sample_bias)` into the graph's only `MagField`."""
function write_single_field_sample_bias!(
    model::M,
    x::X,
    base_bias::B,
    sample_buffer::S,
    target,
    beta::T,
) where {M<:LocalMNISTModel,X<:AbstractVector,B<:AbstractVector,S<:AbstractVector,T<:Real}
    length(x) == length(model.input_idxs) ||
        throw(DimensionMismatch("input length $(length(x)) does not match input layer length $(length(model.input_idxs))"))
    length(base_bias) == length(sample_buffer) == length(base_magfield(model.graph).b) ||
        throw(DimensionMismatch("base/sample/graph field lengths do not match"))

    ensure_input_layer_inactive!(model.graph.index_set)
    fill!(II.state(model.graph[1]), 0f0)
    fill!(sample_buffer, 0f0)

    A = II.adj(model.graph)
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    nz = SparseArrays.nonzeros(A)

    # The input layer is inactive; its couplings become a per-sample local field.
    @inbounds for (xpos, input_idx) in enumerate(model.input_idxs)
        xval = x[xpos]
        for ptr in colptr[input_idx]:(colptr[input_idx + 1] - 1)
            sample_buffer[rows[ptr]] += nz[ptr] * xval
        end
    end
    if target !== nothing
        @inbounds sample_buffer[model.output_idxs] .+= PMNIST_FT(beta) .* target
    end
    sample_buffer .= clamp.(sample_buffer, -model.config.applied_bias_clip, model.config.applied_bias_clip)

    combined = base_magfield(model.graph).b
    combined .= base_bias
    combined .+= sample_buffer
    return model
end

# Install free-phase sample fields into one combined MagField.
Processes.@ProcessAlgorithm function InstallSingleFieldSampleBias!(
    mnist_model::LocalMNISTModel,
    x::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    write_single_field_sample_bias!(mnist_model, x, base_bias, sample_buffer, nothing, 0)
    return nothing
end

# Install nudged sample fields into one combined MagField.
Processes.@ProcessAlgorithm function InstallSingleFieldNudgedSampleBias!(
    mnist_model::LocalMNISTModel,
    x::AbstractVector,
    y::AbstractVector,
    base_bias::AbstractVector,
    sample_buffer::AbstractVector,
)
    write_single_field_sample_bias!(mnist_model, x, base_bias, sample_buffer, y, mnist_model.config.β)
    return nothing
end

"""Build the free-phase routine for the single-field worker graph."""
function single_field_free_phase_algorithm(
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
        InstallSingleFieldSampleBias!(mnist_model, x, base_bias, sample_buffer)
        @repeat steps phase_step()
        CaptureBestEnergyStateFromHamiltonian!(dynamics.model, dynamics.hamiltonian, free_best_energy, free_state)
    end
end

"""Build the nudged-phase routine for the single-field worker graph."""
function single_field_nudged_phase_algorithm(
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
        InstallSingleFieldNudgedSampleBias!(mnist_model, x, y, base_bias, sample_buffer)
        @repeat steps phase_step()
        CaptureBestEnergyStateFromHamiltonian!(dynamics.model, dynamics.hamiltonian, nudged_best_energy, nudged_state)
    end
end

"""Build one full local-worker sample routine using one combined MagField."""
function single_field_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    free_steps = max(1, config.free_sweeps * Int(nstates))
    nudge_steps = max(1, config.nudge_sweeps * Int(nstates))
    free_reads = max(1, config.free_reads)
    nudge_reads = max(1, config.nudge_reads)
    free_temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    nudge_temperature = ReverseAnnealDynamicsTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = nudge_steps)
    free_phase = single_field_free_phase_algorithm(dynamics_algorithm, free_temperature, free_steps)
    nudged_phase = single_field_nudged_phase_algorithm(dynamics_algorithm, nudge_temperature, nudge_steps)
    return Processes.@Routine begin
        @alias free_phase = free_phase
        @alias nudged_phase = nudged_phase
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
        @repeat free_reads free_phase()

        ResetBestEnergyCapture!(nudged_best_energy, nudged_state)
        @repeat nudge_reads nudged_phase()

        FinishLocalContrastiveSample!(
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

"""Build a Process for one full sample on the single-field worker graph."""
function single_field_worker_process(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    model = single_field_worker_model(source, worker_idx)
    graph_state = II.state(model.graph)
    algorithm = Processes.resolve(single_field_worker_algorithm(mnist_dynamics_algorithm(), source.config, length(graph_state)))
    return normal_process(
        algorithm,
        Processes.Init(:_state;
            mnist_model = model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            y = zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas),
            base_bias = copy(base_magfield(source.graph).b),
            sample_buffer = zeros(PMNIST_FT, length(base_magfield(source.graph).b)),
            gradient = gradient_buffer(model),
            free_state = similar(graph_state),
            nudged_state = similar(graph_state),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            nudged_best_energy = Ref(PMNIST_FT(Inf)),
            rng = model.rng,
            nsamples = Ref(0),
            ncorrect = Ref(0),
            nskipped = Ref(0),
            total_loss = Ref(0f0),
        ),
        Processes.Init(:dynamics; model = model.graph),
    )
end

"""Install sample data into the `_state` subcontext of a full worker."""
@inline function set_full_worker_sample!(context, xtrain::X, ytrain::Y, sample_idx::I) where {X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    st = context._state
    st.x .= view(xtrain, :, Int(sample_idx))
    st.y .= view(ytrain, :, Int(sample_idx))
    return context
end

"""Time a process variant for `nruns` independent repetitions."""
@inline function time_process_variant!(runner::F, process, nruns::I) where {F,I<:Integer}
    @inline runner(process)
    return @elapsed begin
        for _ in 1:Int(nruns)
            @inline runner(process)
        end
    end
end

"""Time a full worker variant over concrete sample indices."""
@inline function time_full_worker_variant!(runner::F, process, xtrain::X, ytrain::Y, nsamples::I) where {F,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    @inline set_full_worker_sample!(Processes.context(process), xtrain, ytrain, 1)
    @inline runner(process)
    return @elapsed begin
        for sample_idx in 1:Int(nsamples)
            @inline set_full_worker_sample!(Processes.context(process), xtrain, ytrain, sample_idx)
            @inline runner(process)
        end
    end
end

"""Print a benchmark row with normalized throughput."""
function print_row(label, path, work_units, seconds, unit_name)
    println(join((
        label,
        path,
        work_units,
        unit_name,
        round(seconds; digits = 6),
        round(seconds / work_units; digits = 9),
        round(work_units / seconds; digits = 3),
    ), ","))
    flush(stdout)
    return nothing
end

"""Run two-field versus single-field worker timings."""
function main()
    nsamples = parse(Int, get(ENV, "ISING_STRIP_NSAMPLES", "2"))
    nrepeats = parse(Int, get(ENV, "ISING_STRIP_REPEATS", "3"))
    config = LocalMNISTManagerConfig(;
        name = "single_field_bias_timing",
        workers = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = @__DIR__,
    )

    source = init_model(config, config.seed)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    nstates = length(II.state(source.graph))
    steps_per_sample = (config.free_reads * config.free_sweeps + config.nudge_reads * config.nudge_sweeps) * nstates
    single_field_log("configured"; nsamples, nrepeats, nstates, steps_per_sample)

    println("label,path,work_units,unit,total_seconds,seconds_per_unit,units_per_second")

    free_steps = config.free_sweeps * nstates
    two_free_algo = Processes.resolve(free_phase_algorithm(
        II.Metropolis(),
        GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps),
        free_steps,
    ))
    two_free_model = worker_model(source, 50)
    two_free_proc = normal_process(
        two_free_algo,
        Processes.Init(:_state;
            mnist_model = two_free_model,
            x = copy(view(xtrain, :, 1)),
            free_state = similar(II.state(two_free_model.graph)),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            rng = two_free_model.rng,
        ),
        Processes.Init(:dynamics; model = two_free_model.graph),
    )
    print_row("two_field_free_phase", "normal_process_run_wait", free_steps * nrepeats, time_process_variant!(run_async_once!, two_free_proc, nrepeats), "steps")

    single_free_algo = Processes.resolve(single_field_free_phase_algorithm(
        II.Metropolis(),
        GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps),
        free_steps,
    ))
    single_free_model = single_field_worker_model(source, 51)
    single_free_proc = normal_process(
        single_free_algo,
        Processes.Init(:_state;
            mnist_model = single_free_model,
            x = copy(view(xtrain, :, 1)),
            base_bias = copy(base_magfield(source.graph).b),
            sample_buffer = zeros(PMNIST_FT, length(base_magfield(source.graph).b)),
            free_state = similar(II.state(single_free_model.graph)),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            rng = single_free_model.rng,
        ),
        Processes.Init(:dynamics; model = single_free_model.graph),
    )
    print_row("single_field_free_phase", "normal_process_run_wait", free_steps * nrepeats, time_process_variant!(run_async_once!, single_free_proc, nrepeats), "steps")

    two_full_proc = local_worker(
        source,
        60,
        Processes.resolve(local_worker_algorithm(mnist_dynamics_algorithm(), config, nstates)),
    )
    print_row("two_field_full_worker", "normal_process_run_wait", nsamples, time_full_worker_variant!(run_async_once!, two_full_proc, xtrain, ytrain, nsamples), "samples")

    single_full_proc = single_field_worker_process(source, 61)
    print_row("single_field_full_worker", "normal_process_run_wait", nsamples, time_full_worker_variant!(run_async_once!, single_full_proc, xtrain, ytrain, nsamples), "samples")
end

main()
