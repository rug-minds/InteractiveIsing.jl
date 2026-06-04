using Dates

const LOCAL_SINGLE_DIR = @__DIR__
const LOCAL_SINGLE_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const LOCAL_SINGLE_MANAGER_FILE = joinpath(LOCAL_SINGLE_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"

include(LOCAL_SINGLE_MANAGER_FILE)

"""Append one local-MNIST serial Process timing row."""
function append_local_single_row!(row::R) where {R<:NamedTuple}
    path = joinpath(LOCAL_SINGLE_DIR, "local_single_process_same_la.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the local r8 config used by the old manager timing path."""
function local_single_config()
    return LocalMNISTManagerConfig(;
        name = "local_single_process_same_la",
        workers = 1,
        epochs = 1,
        batchsize = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_BATCHSIZE", "32")),
        job_chunk_size = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_JOB_CHUNK_SIZE", "1")),
        train_per_class = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_TRAIN_PER_CLASS", "100")),
        test_per_class = 1,
        local_radius = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_RADIUS", "8")),
        free_reads = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_FREE_READS", "3")),
        nudge_reads = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_NUDGE_READS", "3")),
        free_sweeps = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_FREE_SWEEPS", "50")),
        nudge_sweeps = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_NUDGE_SWEEPS", "50")),
        β = parse(PMNIST_FT, get(ENV, "ISING_LOCAL_SINGLE_BETA", "5.0")),
        outdir = String(LOCAL_SINGLE_DIR),
    )
end

"""Build one normal Process with the same local-MNIST contrastive LoopAlgorithm as the manager."""
function local_single_worker(source::M) where {M<:LocalMNISTModel}
    dynamics_algorithm = mnist_dynamics_algorithm()
    algorithm = Processes.resolve(
        contrastive_worker_algorithm(deepcopy(dynamics_algorithm), source.config, length(II.state(source.graph))),
    )
    return local_worker(source, 1, algorithm)
end

"""Run one local-MNIST sample through the normal Process inline path."""
function run_one_local_single_sample!(
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

"""Measure warmed single-process local-MNIST samples with the manager LoopAlgorithm."""
function measure_local_single_process!(source::M, xtrain::X, ytrain::Y) where {
    M<:LocalMNISTModel,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    worker = local_single_worker(source)
    try
        warmups = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_WARMUPS", "1"))
        measured = parse(Int, get(ENV, "ISING_LOCAL_SINGLE_EXAMPLES", "3"))
        println("[", Dates.format(now(), "HH:MM:SS"), "] warming local single Process")
        flush(stdout)
        @inbounds for sample_idx in 1:warmups
            run_one_local_single_sample!(worker, xtrain, ytrain, sample_idx)
        end

        rows = NamedTuple[]
        @inbounds for sample_idx in (warmups + 1):(warmups + measured)
            seconds = @elapsed run_one_local_single_sample!(worker, xtrain, ytrain, sample_idx)
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                mode = "local_serial_process_inline",
                threads = Threads.nthreads(),
                workers = 1,
                sample_idx,
                radius = source.config.local_radius,
                free_sweeps = source.config.free_sweeps,
                nudge_sweeps = source.config.nudge_sweeps,
                free_reads = source.config.free_reads,
                nudge_reads = source.config.nudge_reads,
                beta = source.config.β,
                examples = 1,
                seconds,
                seconds_per_example = seconds,
                examples_per_second = inv(seconds),
            )
            push!(rows, row)
            append_local_single_row!(row)
            println(row)
            flush(stdout)
        end
        return rows
    finally
        close(worker)
    end
end

"""Measure one warmed one-worker local ProcessManager batch for the same LoopAlgorithm."""
function measure_local_one_worker_manager!(source::M, xtrain::X, ytrain::Y, config::C) where {
    M<:LocalMNISTModel,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:LocalMNISTManagerConfig,
}
    manager = local_manager(source)
    try
        indices = collect(1:config.batchsize)
        jobs = batch_jobs(indices, config.batchsize)
        println("[", Dates.format(now(), "HH:MM:SS"), "] warming local one-worker manager")
        flush(stdout)
        run_minibatch!(manager, xtrain, ytrain, jobs; log_progress = false)

        seconds = @elapsed run_minibatch!(manager, xtrain, ytrain, jobs; log_progress = false)
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            mode = "local_one_worker_manager_chunk",
            threads = Threads.nthreads(),
            workers = 1,
            sample_idx = 0,
            radius = config.local_radius,
            free_sweeps = config.free_sweeps,
            nudge_sweeps = config.nudge_sweeps,
            free_reads = config.free_reads,
            nudge_reads = config.nudge_reads,
            beta = config.β,
            examples = length(indices),
            seconds,
            seconds_per_example = seconds / length(indices),
            examples_per_second = length(indices) / seconds,
        )
        append_local_single_row!(row)
        println(row)
        flush(stdout)
        return row
    finally
        close(manager)
    end
end

"""Run the local single-process same-LA diagnostic."""
function main()
    mkpath(LOCAL_SINGLE_DIR)
    rm(joinpath(LOCAL_SINGLE_DIR, "local_single_process_same_la.csv"); force = true)
    config = local_single_config()
    println("[", Dates.format(now(), "HH:MM:SS"), "] building local r$(config.local_radius) source")
    flush(stdout)
    source = init_model(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    rows = measure_local_single_process!(source, xtrain, ytrain)
    manager_row = measure_local_one_worker_manager!(source, xtrain, ytrain, config)
    return (; rows, manager_row)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
