using Dates
using Statistics

const RUN_DIR = @__DIR__
const BASELINE_PATH = normpath(joinpath(RUN_DIR, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(BASELINE_PATH)

"""Append one current-manager throughput row to a run-local CSV file."""
function append_manager32_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the one-batch full-learning config used for the 32-worker manager timing."""
function manager32_pipeline_config()
    return InputFieldMNISTConfig(;
        workers = min(32, Threads.nthreads()),
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
        stepsize = 0.5f0,
        seed = 20260526,
        outdir = String(RUN_DIR),
    )
end

"""Run and time one current chunked-manager minibatch including update and sync."""
function time_manager32_minibatch!(
    manager::M,
    jobs_buffer::B,
    xtrain::X,
    ytrain::Y,
    indices::V,
    config::C,
) where {
    M<:Processes.ProcessManager,
    B<:InputFieldMNISTChunkBuffer,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    V<:AbstractVector{Int},
    C<:InputFieldMNISTConfig,
}
    set_inputs_seconds = @elapsed set_manager_inputs!(manager, xtrain, ytrain)
    chunk_size = manager_chunk_size(config, length(indices))
    fill_seconds = @elapsed jobs = fill_chunk_jobs!(jobs_buffer, indices, chunk_size)
    minibatch_seconds = @elapsed run_minibatch!(manager, jobs)
    total_seconds = set_inputs_seconds + fill_seconds + minibatch_seconds
    return (;
        examples = length(indices),
        chunks = length(jobs),
        chunk_size,
        set_inputs_seconds,
        fill_seconds,
        minibatch_seconds,
        total_seconds,
        seconds_per_example = total_seconds / length(indices),
        examples_per_second = length(indices) / total_seconds,
    )
end

"""Benchmark the current 32-worker chunked manager over warmed minibatches."""
function main()
    mkpath(RUN_DIR)
    csv_path = joinpath(RUN_DIR, "manager_32_chunked_pipeline_current.csv")
    summary_path = joinpath(RUN_DIR, "manager_32_chunked_pipeline_current_summary.csv")
    rm(csv_path; force = true)
    rm(summary_path; force = true)

    repeats = parse(Int, get(ENV, "ISING_MNIST_MANAGER32_REPEATS", "3"))
    config = manager32_pipeline_config()

    println(now(), " begin manager32 current chunked pipeline threads=$(Threads.nthreads()) workers=$(config.workers) batchsize=$(config.batchsize) sweeps=$(config.sweeps) repeats=$(repeats)")
    flush(stdout)

    setup_seconds = @elapsed setup = build_layer(config)
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    input_hidden_w = Ref(setup.input_hidden_w)
    manager_seconds = @elapsed manager = input_field_manager(setup.layer, setup.graph, config, input_hidden_w)
    jobs_buffer = InputFieldMNISTChunkBuffer(config.workers, manager_chunk_size(config, config.batchsize))

    rows = NamedTuple[]
    try
        warm_indices = collect(1:config.batchsize)
        println(now(), " warmup begin")
        flush(stdout)
        warmup = time_manager32_minibatch!(manager, jobs_buffer, xtrain, ytrain, warm_indices, config)
        println(now(), " warmup total=$(warmup.total_seconds) spe=$(warmup.seconds_per_example) eps=$(warmup.examples_per_second)")
        flush(stdout)

        for repeat_idx in 1:repeats
            first_idx = repeat_idx * config.batchsize + 1
            last_idx = first_idx + config.batchsize - 1
            indices = collect(first_idx:last_idx)
            timed = time_manager32_minibatch!(manager, jobs_buffer, xtrain, ytrain, indices, config)
            row = merge(
                (;
                    timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                    repeat = repeat_idx,
                    threads = Threads.nthreads(),
                    workers = config.workers,
                    batchsize = config.batchsize,
                    sweeps = config.sweeps,
                    setup_seconds,
                    data_seconds,
                    manager_seconds,
                ),
                timed,
            )
            append_manager32_row!(csv_path, row)
            push!(rows, row)
            println(now(), " measured repeat=$(repeat_idx) total=$(row.total_seconds) spe=$(row.seconds_per_example) eps=$(row.examples_per_second) chunks=$(row.chunks)")
            flush(stdout)
        end
    finally
        close(manager)
    end

    spe = map(row -> row.seconds_per_example, rows)
    eps = map(row -> row.examples_per_second, rows)
    summary = (;
        repeats,
        threads = Threads.nthreads(),
        workers = config.workers,
        batchsize = config.batchsize,
        sweeps = config.sweeps,
        median_seconds_per_example = median(spe),
        mean_seconds_per_example = mean(spe),
        median_examples_per_second = median(eps),
        mean_examples_per_second = mean(eps),
    )
    append_manager32_row!(summary_path, summary)
    println(now(), " summary=$(summary)")
    println(now(), " csv=$(csv_path)")
    println(now(), " summary_csv=$(summary_path)")
    flush(stdout)
    return (; rows, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
