using Dates
using Optimisers

const CHUNKED_SCALING_DIR = @__DIR__
const CHUNKED_SCALING_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(CHUNKED_SCALING_BASELINE)

"""Write one row to the chunked-spawn scaling CSV."""
function append_chunked_scaling_row!(row::R) where {R<:NamedTuple}
    path = joinpath(CHUNKED_SCALING_DIR, "chunked_spawn_16_vs_32.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the compact baseline config used for one worker-count timing run."""
function chunked_scaling_config(workers::I) where {I<:Integer}
    return InputFieldMNISTConfig(;
        workers = Int(workers),
        epochs = 1,
        batchsize = 128,
        scheduler = "spawn",
        chunk_size = 0,
        train_per_class = 80,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        seed = 20260526,
        outdir = String(CHUNKED_SCALING_DIR),
    )
end

"""Run one warmed chunked-spawn minibatch and return manager-only timings."""
function timed_chunked_minibatch!(
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
    chunk_seconds = @elapsed jobs = fill_chunk_jobs!(jobs_buffer, indices, manager_chunk_size(config, length(indices)))
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
        chunk_size = manager_chunk_size(config, length(indices)),
        set_seconds,
        chunk_seconds,
        clear_seconds,
        run_seconds,
        update_seconds,
        sync_seconds,
        total_seconds,
        run_seconds_per_example = run_seconds / length(indices),
        total_seconds_per_example = total_seconds / length(indices),
    )
end

"""Measure warmed chunked-spawn training batches for one worker count."""
function measure_worker_count!(setup, xtrain::X, ytrain::Y, workers::I) where {X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    config = chunked_scaling_config(workers)
    manager = input_field_manager(setup.layer, setup.graph, config)
    jobs_buffer = InputFieldMNISTChunkBuffer(config.workers, manager_chunk_size(config, config.batchsize))
    measured_batches = parse(Int, get(ENV, "ISING_MNIST_CHUNKED_SCALING_BATCHES", "3"))
    try
        warm_indices = collect(1:config.batchsize)
        println("[", Dates.format(now(), "HH:MM:SS"), "] warmup workers=", workers)
        flush(stdout)
        timed_chunked_minibatch!(manager, jobs_buffer, xtrain, ytrain, warm_indices, config)

        rows = NamedTuple[]
        for batch_idx in 1:measured_batches
            first_idx = config.batchsize * batch_idx + 1
            last_idx = first_idx + config.batchsize - 1
            indices = collect(first_idx:last_idx)
            timing = timed_chunked_minibatch!(manager, jobs_buffer, xtrain, ytrain, indices, config)
            row = merge(
                (;
                    timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                    threads = Threads.nthreads(),
                    workers = Int(workers),
                    batch = batch_idx,
                    sweeps = config.sweeps,
                    relaxation_steps = setup.relaxation_steps,
                ),
                timing,
            )
            push!(rows, row)
            append_chunked_scaling_row!(row)
            println(row)
            flush(stdout)
        end
        return rows
    finally
        close(manager)
    end
end

"""Run the 16-vs-32 worker chunked-spawn scaling diagnostic."""
function main()
    mkpath(CHUNKED_SCALING_DIR)
    rm(joinpath(CHUNKED_SCALING_DIR, "chunked_spawn_16_vs_32.csv"); force = true)
    max_workers = min(32, Threads.nthreads())
    worker_counts = max_workers >= 32 ? (16, 32) : (min(16, max_workers), max_workers)

    config = chunked_scaling_config(maximum(worker_counts))
    println("[", Dates.format(now(), "HH:MM:SS"), "] building setup threads=", Threads.nthreads())
    flush(stdout)
    setup = build_layer(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    all_rows = NamedTuple[]
    for workers in worker_counts
        append!(all_rows, measure_worker_count!(setup, xtrain, ytrain, workers))
    end
    println("[", Dates.format(now(), "HH:MM:SS"), "] done rows=", length(all_rows))
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
