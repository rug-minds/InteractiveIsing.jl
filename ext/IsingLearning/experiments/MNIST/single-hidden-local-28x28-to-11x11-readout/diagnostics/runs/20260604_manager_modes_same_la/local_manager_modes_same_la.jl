using Dates
using Statistics

const MODE_BENCH_DIR = @__DIR__
const MODE_BENCH_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const MODE_BENCH_MANAGER_FILE = joinpath(MODE_BENCH_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"

include(MODE_BENCH_MANAGER_FILE)

"""Append one same-LA scheduler benchmark row."""
function append_mode_bench_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the local r8 benchmark config shared by all manager modes."""
function mode_bench_config(; workers::Int, batchsize::Int = 128)
    return LocalMNISTManagerConfig(;
        name = "local_manager_modes_same_la",
        workers,
        epochs = 1,
        batchsize,
        job_chunk_size = parse(Int, get(ENV, "ISING_MODE_BENCH_JOB_CHUNK_SIZE", "4")),
        train_per_class = parse(Int, get(ENV, "ISING_MODE_BENCH_TRAIN_PER_CLASS", "80")),
        test_per_class = 1,
        local_radius = parse(Int, get(ENV, "ISING_MODE_BENCH_RADIUS", "8")),
        free_reads = parse(Int, get(ENV, "ISING_MODE_BENCH_FREE_READS", "3")),
        nudge_reads = parse(Int, get(ENV, "ISING_MODE_BENCH_NUDGE_READS", "3")),
        free_sweeps = parse(Int, get(ENV, "ISING_MODE_BENCH_FREE_SWEEPS", "50")),
        nudge_sweeps = parse(Int, get(ENV, "ISING_MODE_BENCH_NUDGE_SWEEPS", "50")),
        β = parse(PMNIST_FT, get(ENV, "ISING_MODE_BENCH_BETA", "5.0")),
        outdir = String(MODE_BENCH_DIR),
    )
end

"""Resolve the same one-sample contrastive LoopAlgorithm used by local MNIST training."""
function mode_bench_algorithm(source::M) where {M<:LocalMNISTModel}
    dynamics_algorithm = mnist_dynamics_algorithm()
    return Processes.resolve(
        contrastive_worker_algorithm(deepcopy(dynamics_algorithm), source.config, length(II.state(source.graph))),
    )
end

"""Build one normal Process worker for same-LA serial measurements."""
function mode_bench_worker(source::M, algorithm::A) where {M<:LocalMNISTModel,A}
    return local_worker(source, 1, algorithm)
end

"""Load and run one sample through a normal Process using the inline path."""
function mode_bench_run_inline_sample!(
    worker::W,
    x::X,
    y::Y,
    sample_idx::I,
) where {W<:Processes.Process,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    context = worker_context(worker)
    load_sample_into_worker!(context, x, y, sample_idx)
    Processes.reset!(worker)
    return Processes.runprocessinline!(worker)
end

"""Load and run one sample through a normal Process using `run`/`wait`."""
function mode_bench_run_process_sample!(
    worker::W,
    x::X,
    y::Y,
    sample_idx::I,
) where {W<:Processes.Process,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    context = worker_context(worker)
    load_sample_into_worker!(context, x, y, sample_idx)
    Processes.reset!(worker)
    run(worker)
    wait(worker)
    return worker
end

"""Measure serial same-LA Process calls without manager dispatch."""
function measure_serial_process_modes!(path::P, source::M, xtrain::X, ytrain::Y) where {
    P<:AbstractString,
    M<:LocalMNISTModel,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    algorithm = mode_bench_algorithm(source)
    warmups = parse(Int, get(ENV, "ISING_MODE_BENCH_SERIAL_WARMUPS", "2"))
    repeats = parse(Int, get(ENV, "ISING_MODE_BENCH_SERIAL_REPEATS", "4"))
    modes = (;
        serial_process_inline = mode_bench_run_inline_sample!,
        serial_process_run_wait = mode_bench_run_process_sample!,
    )
    rows = NamedTuple[]
    for (mode, runner!) in pairs(modes)
        worker = mode_bench_worker(source, algorithm)
        try
            @inbounds for sample_idx in 1:warmups
                runner!(worker, xtrain, ytrain, sample_idx)
            end
            @inbounds for repeat_idx in 1:repeats
                sample_idx = warmups + repeat_idx
                seconds = @elapsed runner!(worker, xtrain, ytrain, sample_idx)
                row = mode_bench_row(source.config, mode, 1, 1, 1, seconds; repeat_idx, jobs = 1)
                push!(rows, row)
                append_mode_bench_row!(path, row)
                println(row)
                flush(stdout)
            end
        finally
            close(worker)
        end
    end
    return rows
end

"""Construct a per-sample-job manager so threaded/channel modes avoid per-job custom spawning."""
function per_sample_local_manager(source::M) where {M<:LocalMNISTModel}
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
    algorithm = mode_bench_algorithm(source)
    recipe = (;
        makeworker = (idx, manager) -> local_worker(manager.state.model, idx, algorithm),
        prepare! = (slot, job, manager) -> begin
            context = worker_context(slot.worker)
            x = manager.state.current_x[]
            y = manager.state.current_y[]
            load_sample_into_worker!(context, x, y, job)
            Processes.resetworker!(slot)
            return nothing
        end,
        flush! = manager -> flush_manager_buffers!(manager),
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = source.config.workers,
        config = source.config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = Int,
    )
end

"""Return the concrete manager scheduler object for one benchmark mode."""
function mode_bench_schedule(mode::S) where {S<:Symbol}
    mode === :spawn && return nothing
    mode === :greedy && return Processes.Greedy()
    mode === :channelworkers && return Processes.ChannelWorkers()
    throw(ArgumentError("unknown mode $(mode)"))
end

"""Run one manager batch in the requested mode and return the timed row."""
function measure_manager_mode!(
    path::P,
    source::M,
    xtrain::X,
    ytrain::Y,
    mode::S,
    repeat_idx::I,
) where {
    P<:AbstractString,
    M<:LocalMNISTModel,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    S<:Symbol,
    I<:Integer,
}
    manager = per_sample_local_manager(source)
    try
        jobs = collect(1:source.config.batchsize)
        manager.state.current_x[] = xtrain
        manager.state.current_y[] = ytrain
        schedule = mode_bench_schedule(mode)
        clear_manager_buffers!(manager)
        if isnothing(schedule)
            Processes.run!(manager, jobs)
        else
            Processes.run!(manager, jobs, schedule)
        end

        clear_manager_buffers!(manager)
        seconds = @elapsed begin
            if isnothing(schedule)
                Processes.run!(manager, jobs)
            else
                Processes.run!(manager, jobs, schedule)
            end
        end
        row = mode_bench_row(
            source.config,
            Symbol("manager_", mode),
            source.config.workers,
            source.config.batchsize,
            source.config.batchsize,
            seconds;
            repeat_idx,
            jobs = length(jobs),
        )
        append_mode_bench_row!(path, row)
        println(row)
        flush(stdout)
        return row
    finally
        close(manager)
    end
end

"""Create a normalized timing row for local r8 scheduler comparisons."""
function mode_bench_row(
    config::C,
    mode::S,
    workers::I,
    examples::J,
    batchsize::K,
    seconds::T;
    repeat_idx::L,
    jobs::M,
) where {
    C<:LocalMNISTManagerConfig,
    S<:Symbol,
    I<:Integer,
    J<:Integer,
    K<:Integer,
    T<:Real,
    L<:Integer,
    M<:Integer,
}
    return (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        mode = String(mode),
        repeat = repeat_idx,
        threads = Threads.nthreads(),
        workers,
        physical_core_ceiling = 16,
        batchsize,
        jobs,
        examples,
        radius = config.local_radius,
        free_sweeps = config.free_sweeps,
        nudge_sweeps = config.nudge_sweeps,
        free_reads = config.free_reads,
        nudge_reads = config.nudge_reads,
        beta = config.β,
        seconds,
        seconds_per_example = seconds / examples,
        examples_per_second = examples / seconds,
        projected_serial_seconds_per_example_at_16x = workers > 1 ? (seconds / examples) * 16 : seconds / examples,
    )
end

"""Run all current same-LA process and manager scheduling benchmarks."""
function main()
    mkpath(MODE_BENCH_DIR)
    csv_path = joinpath(MODE_BENCH_DIR, "local_manager_modes_same_la.csv")
    summary_path = joinpath(MODE_BENCH_DIR, "local_manager_modes_same_la_summary.csv")
    rm(csv_path; force = true)
    rm(summary_path; force = true)

    batchsize = parse(Int, get(ENV, "ISING_MODE_BENCH_BATCHSIZE", "128"))
    manager_repeats = parse(Int, get(ENV, "ISING_MODE_BENCH_MANAGER_REPEATS", "2"))
    worker_counts = Tuple(parse(Int, strip(part)) for part in split(get(ENV, "ISING_MODE_BENCH_WORKERS", "1,16,32"), ",") if !isempty(strip(part)))
    manager_modes = Tuple(Symbol(strip(part)) for part in split(get(ENV, "ISING_MODE_BENCH_MODES", "spawn,greedy,channelworkers"), ",") if !isempty(strip(part)))

    base_config = mode_bench_config(; workers = 1, batchsize)
    println("[", Dates.format(now(), "HH:MM:SS"), "] building source")
    flush(stdout)
    source = init_model(base_config)
    xtrain, ytrain = balanced_mnist(:train, base_config.train_per_class, base_config)

    rows = measure_serial_process_modes!(csv_path, source, xtrain, ytrain)

    for workers in worker_counts
        config = mode_bench_config(; workers, batchsize)
        manager_source = init_model(config)
        for mode in manager_modes
            for repeat_idx in 1:manager_repeats
                push!(rows, measure_manager_mode!(csv_path, manager_source, xtrain, ytrain, mode, repeat_idx))
            end
        end
    end

    for mode in unique(getproperty.(rows, :mode))
        selected = [row for row in rows if row.mode == mode]
        spe = getproperty.(selected, :seconds_per_example)
        eps = getproperty.(selected, :examples_per_second)
        summary = (;
            mode,
            rows = length(selected),
            median_seconds_per_example = median(spe),
            mean_seconds_per_example = mean(spe),
            median_examples_per_second = median(eps),
            mean_examples_per_second = mean(eps),
        )
        append_mode_bench_row!(summary_path, summary)
        println(summary)
    end
    println("csv=", csv_path)
    println("summary_csv=", summary_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
