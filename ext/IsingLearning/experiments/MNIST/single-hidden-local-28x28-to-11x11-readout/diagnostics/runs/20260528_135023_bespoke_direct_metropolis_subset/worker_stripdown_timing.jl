using Dates

const STRIP_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const STRIP_MANAGER_FILE = joinpath(STRIP_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "worker_stripdown_timing"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = "1"
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")

include(STRIP_MANAGER_FILE)

using Random
using SparseArrays

"""Print one timestamped strip-down diagnostic line."""
function strip_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Randomize graph state while keeping the inactive input layer at zero."""
function strip_randomize!(graph::G, rng::R) where {G,R<:Random.AbstractRNG}
    state = II.state(graph)
    @inbounds for idx in eachindex(state)
        state[idx] = rand(rng, Bool) ? one(eltype(state)) : -one(eltype(state))
    end
    fill!(II.state(graph[1]), 0f0)
    return graph
end

"""Install one sample into a worker graph using the combined-field path."""
function install_strip_sample_bias!(model::M, source::S, x::X) where {M<:LocalMNISTModel,S<:LocalMNISTModel,X<:AbstractVector}
    sample_buffer = zeros(PMNIST_FT, length(base_magfield(source.graph).b))
    install_sample_bias!(model, x, base_magfield(source.graph).b, sample_buffer)
    return model
end

"""Create a normal Process from an algorithm and init specs."""
function normal_process(algorithm::A, inits::Vararg{Any,N}) where {A,N}
    return StatefulAlgorithms.Process(algorithm, inits...; repeat = 1)
end

"""Create an InlineProcess from the stable post-first-run context of a normal Process."""
function warmed_inline_process(algorithm::A, inits::Vararg{Any,N}) where {A,N}
    warm_process = normal_process(algorithm, inits...)
    run(warm_process)
    wait(warm_process)
    return StatefulAlgorithms.InlineProcess(
        algorithm;
        context = StatefulAlgorithms.context(warm_process),
        repeats = 1,
        threaded = false,
    )
end

"""Run a Process through the asynchronous `run`/`wait` path."""
@inline function run_async_once!(process::P) where {P<:StatefulAlgorithms.Process}
    StatefulAlgorithms.reset!(process)
    @inline run(process)
    @inline wait(process)
    return process
end

"""Run an InlineProcess once in synchronous mode."""
@inline function run_inline_process_once!(process::P) where {P<:StatefulAlgorithms.InlineProcess}
    @inline run(process; threaded = false)
    return process
end

"""Build a process algorithm for raw Metropolis repeated `nsteps` times."""
function raw_metropolis_algorithm(nsteps::I) where {I<:Integer}
    dynamics = II.Metropolis()
    return StatefulAlgorithms.resolve(StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics
        @repeat nsteps dynamics()
    end)
end

"""Build a process algorithm for temperature-scheduled Metropolis repeated `nsteps` times."""
function scheduled_metropolis_algorithm(config::C, nsteps::I) where {C<:LocalMNISTManagerConfig,I<:Integer}
    dynamics = II.Metropolis()
    temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = Int(nsteps))
    phase_step = free_phase_step_algorithm(dynamics, temperature)
    return StatefulAlgorithms.resolve(StatefulAlgorithms.@Routine begin
        @alias phase_step = phase_step
        @repeat nsteps phase_step()
    end)
end

"""Build a process algorithm for the full free phase once."""
function free_phase_once_algorithm(config::C, nstates::I) where {C<:LocalMNISTManagerConfig,I<:Integer}
    free_steps = config.free_sweeps * Int(nstates)
    dynamics = II.Metropolis()
    temperature = GeometricDynamicsTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    return StatefulAlgorithms.resolve(free_phase_algorithm(dynamics, temperature, free_steps))
end

"""Build a process algorithm for the full local worker sample once."""
function full_worker_algorithm(config::C, nstates::I) where {C<:LocalMNISTManagerConfig,I<:Integer}
    return StatefulAlgorithms.resolve(local_worker_algorithm(mnist_dynamics_algorithm(), config, nstates))
end

"""Make a Process running a full worker sample."""
function full_worker_process(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    return local_worker(source, worker_idx, full_worker_algorithm(source.config, length(II.state(source.graph))))
end

"""Make an InlineProcess running a full worker sample."""
function full_worker_inline_process(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    model = worker_model(source, worker_idx)
    graph_state = II.state(model.graph)
    algorithm = full_worker_algorithm(source.config, length(graph_state))
    return warmed_inline_process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            mnist_model = model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            y = zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas),
            base_bias = base_magfield(source.graph).b,
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
        StatefulAlgorithms.Init(:dynamics; model = model.graph),
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
    @inline set_full_worker_sample!(StatefulAlgorithms.context(process), xtrain, ytrain, 1)
    @inline runner(process)
    return @elapsed begin
        for sample_idx in 1:Int(nsamples)
            @inline set_full_worker_sample!(StatefulAlgorithms.context(process), xtrain, ytrain, sample_idx)
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

"""Run strip-down timings for the MNIST worker process stack."""
function main()
    nsteps = parse(Int, get(ENV, "ISING_STRIP_NSTEPS", "100000"))
    nsamples = parse(Int, get(ENV, "ISING_STRIP_NSAMPLES", "3"))
    config = LocalMNISTManagerConfig(;
        name = "worker_stripdown_timing",
        workers = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = @__DIR__,
    )

    source = init_model(config, config.seed)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    nstates = length(II.state(source.graph))
    active = length(II.sampling_indices(source.graph.index_set))
    nnz = length(SparseArrays.nonzeros(II.adj(source.graph)))
    steps_per_sample = (config.free_reads * config.free_sweeps + config.nudge_reads * config.nudge_sweeps) * nstates
    strip_log("configured"; nsteps, nsamples, nstates, active, nnz, steps_per_sample)

    println("label,path,work_units,unit,total_seconds,seconds_per_unit,units_per_second")

    raw_model = worker_model(source, 10)
    strip_randomize!(raw_model.graph, Random.MersenneTwister(10))
    install_strip_sample_bias!(raw_model, source, view(xtrain, :, 1))
    raw_algo = raw_metropolis_algorithm(nsteps)
    raw_async = normal_process(raw_algo, StatefulAlgorithms.Init(:dynamics; model = raw_model.graph))
    nrepeats = 3
    print_row("raw_metropolis", "normal_process_run_wait", nsteps * nrepeats, time_process_variant!(run_async_once!, raw_async, nrepeats), "steps")

    raw_model3 = worker_model(source, 12)
    strip_randomize!(raw_model3.graph, Random.MersenneTwister(12))
    install_strip_sample_bias!(raw_model3, source, view(xtrain, :, 1))
    raw_inline = warmed_inline_process(raw_algo, StatefulAlgorithms.Init(:dynamics; model = raw_model3.graph))
    print_row("raw_metropolis", "inline_process", nsteps * nrepeats, time_process_variant!(run_inline_process_once!, raw_inline, nrepeats), "steps")

    sched_algo = scheduled_metropolis_algorithm(config, nsteps)
    sched_model = worker_model(source, 20)
    strip_randomize!(sched_model.graph, Random.MersenneTwister(20))
    install_strip_sample_bias!(sched_model, source, view(xtrain, :, 1))
    sched_async = normal_process(sched_algo, StatefulAlgorithms.Init(:dynamics; model = sched_model.graph))
    print_row("scheduled_metropolis", "normal_process_run_wait", nsteps * nrepeats, time_process_variant!(run_async_once!, sched_async, nrepeats), "steps")

    sched_model3 = worker_model(source, 22)
    strip_randomize!(sched_model3.graph, Random.MersenneTwister(22))
    install_strip_sample_bias!(sched_model3, source, view(xtrain, :, 1))
    sched_inline = warmed_inline_process(sched_algo, StatefulAlgorithms.Init(:dynamics; model = sched_model3.graph))
    print_row("scheduled_metropolis", "inline_process", nsteps * nrepeats, time_process_variant!(run_inline_process_once!, sched_inline, nrepeats), "steps")

    free_algo = free_phase_once_algorithm(config, nstates)
    free_model = worker_model(source, 30)
    free_proc = normal_process(
        free_algo,
        StatefulAlgorithms.Init(:_state;
            mnist_model = free_model,
            x = copy(view(xtrain, :, 1)),
            base_bias = base_magfield(source.graph).b,
            sample_buffer = zeros(PMNIST_FT, length(base_magfield(source.graph).b)),
            free_state = similar(II.state(free_model.graph)),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            rng = free_model.rng,
        ),
        StatefulAlgorithms.Init(:dynamics; model = free_model.graph),
    )
    print_row("free_phase_once", "normal_process_run_wait", config.free_sweeps * nstates * nrepeats, time_process_variant!(run_async_once!, free_proc, nrepeats), "steps")

    full_proc = full_worker_process(init_model(config, config.seed), 40)
    print_row("full_worker_sample", "normal_process_run_wait", nsamples, time_full_worker_variant!(run_async_once!, full_proc, xtrain, ytrain, nsamples), "samples")

    full_inline = full_worker_inline_process(init_model(config, config.seed), 42)
    print_row("full_worker_sample", "inline_process", nsamples, time_full_worker_variant!(run_inline_process_once!, full_inline, xtrain, ytrain, nsamples), "samples")
end

main()
