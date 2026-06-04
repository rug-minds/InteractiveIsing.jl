using Dates
using Optimisers

const CHUNK_SIZE_GRID_DIR = @__DIR__
const CHUNK_SIZE_GRID_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(CHUNK_SIZE_GRID_BASELINE)

"""Append one timing row for the fixed-chunk-size manager scaling diagnostic."""
function append_chunk_size_grid_row!(row::R) where {R<:NamedTuple}
    path = joinpath(CHUNK_SIZE_GRID_DIR, "chunk_size_grid_16_32.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the real-run timing config with a fixed chunk size per worker."""
function chunk_size_grid_config(workers::I, chunk_size::J) where {I<:Integer,J<:Integer}
    workers_int = Int(workers)
    chunk_size_int = Int(chunk_size)
    return InputFieldMNISTConfig(;
        workers = workers_int,
        epochs = 1,
        batchsize = workers_int * chunk_size_int,
        scheduler = "spawn",
        chunk_size = chunk_size_int,
        train_per_class = 220,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        stepsize = 0.5f0,
        seed = 20260526,
        outdir = String(CHUNK_SIZE_GRID_DIR),
    )
end

"""Run one warmed chunked-spawn minibatch and return manager-only timings."""
function timed_fixed_chunk_minibatch!(
    manager::M,
    jobs_buffer::B,
    xtrain::X,
    ytrain::Y,
    indices::V,
    config::C,
) where {
    M<:StatefulAlgorithms.ProcessManager,
    B<:InputFieldMNISTChunkBuffer,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    V<:AbstractVector{Int},
    C<:InputFieldMNISTConfig,
}
    set_seconds = @elapsed set_manager_inputs!(manager, xtrain, ytrain)
    chunk_seconds = @elapsed jobs = fill_chunk_jobs!(jobs_buffer, indices, config.chunk_size)
    clear_seconds = @elapsed clear_manager_buffers!(manager)
    run_seconds = @elapsed begin
        manager.state.nsamples[] = sum(length, jobs)
        StatefulAlgorithms.run!(manager, jobs)
    end
    update_seconds = @elapsed begin
        manager.state.opt_state, params = Optimisers.update(
            manager.state.opt_state,
            manager.state.params[],
            manager.state.batch_gradient,
        )
        manager.state.params[] = params
    end
    sync_seconds = @elapsed sync_after_update!(manager, manager.state.params[])
    total_seconds = set_seconds + chunk_seconds + clear_seconds + run_seconds + update_seconds + sync_seconds
    return (;
        examples = length(indices),
        chunks = length(jobs),
        chunk_size = config.chunk_size,
        set_seconds,
        chunk_seconds,
        clear_seconds,
        run_seconds,
        update_seconds,
        sync_seconds,
        total_seconds,
        run_seconds_per_example = run_seconds / length(indices),
        total_seconds_per_example = total_seconds / length(indices),
        examples_per_second = length(indices) / total_seconds,
    )
end

"""Measure a worker/chunk-size point with the real training work per example."""
function measure_chunk_size_point!(
    setup,
    xtrain::X,
    ytrain::Y,
    workers::I,
    chunk_size::J,
) where {X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer,J<:Integer}
    config = chunk_size_grid_config(workers, chunk_size)
    manager = input_field_manager(setup.layer, setup.graph, config, Ref(setup.input_hidden_w))
    jobs_buffer = InputFieldMNISTChunkBuffer(config.workers, config.chunk_size)
    measured_batches = parse(Int, get(ENV, "ISING_MNIST_CHUNK_SIZE_GRID_BATCHES", "2"))
    try
        warm_indices = collect(1:config.batchsize)
        println("[", Dates.format(now(), "HH:MM:SS"), "] warmup workers=", workers, " chunk_size=", chunk_size, " batchsize=", config.batchsize)
        flush(stdout)
        timed_fixed_chunk_minibatch!(manager, jobs_buffer, xtrain, ytrain, warm_indices, config)

        rows = NamedTuple[]
        for batch_idx in 1:measured_batches
            first_idx = config.batchsize * batch_idx + 1
            last_idx = first_idx + config.batchsize - 1
            indices = collect(first_idx:last_idx)
            timing = timed_fixed_chunk_minibatch!(manager, jobs_buffer, xtrain, ytrain, indices, config)
            row = merge(
                (;
                    timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                    threads = Threads.nthreads(),
                    workers = config.workers,
                    batch = batch_idx,
                    sweeps = config.sweeps,
                    beta = config.β,
                    temp = config.temp,
                    stepsize = config.stepsize,
                    relaxation_steps = setup.relaxation_steps,
                ),
                timing,
            )
            push!(rows, row)
            append_chunk_size_grid_row!(row)
            println(row)
            flush(stdout)
        end
        return rows
    finally
        close(manager)
    end
end

"""Run a fixed-chunk-size 16-vs-32 worker scaling grid."""
function main()
    mkpath(CHUNK_SIZE_GRID_DIR)
    rm(joinpath(CHUNK_SIZE_GRID_DIR, "chunk_size_grid_16_32.csv"); force = true)

    max_workers = min(32, Threads.nthreads())
    worker_counts = max_workers >= 32 ? (16, 32) : (min(16, max_workers), max_workers)
    chunk_sizes = (4, 8, 16)

    config = chunk_size_grid_config(maximum(worker_counts), maximum(chunk_sizes))
    println("[", Dates.format(now(), "HH:MM:SS"), "] building setup threads=", Threads.nthreads())
    flush(stdout)
    setup = build_layer(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    all_rows = NamedTuple[]
    for chunk_size in chunk_sizes
        for workers in worker_counts
            append!(all_rows, measure_chunk_size_point!(setup, xtrain, ytrain, workers, chunk_size))
        end
    end
    println("[", Dates.format(now(), "HH:MM:SS"), "] done rows=", length(all_rows))
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
