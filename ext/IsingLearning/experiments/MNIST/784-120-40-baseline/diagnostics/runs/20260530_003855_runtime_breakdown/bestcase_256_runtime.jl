using Dates
using Random
using Optimisers

const DIAG_OUTDIR = @__DIR__
const DIAG_BASELINE_FILE = normpath(joinpath(DIAG_OUTDIR, "..", "..", "..", "mnist_784_120_40_adam.jl"))

include(DIAG_BASELINE_FILE)

"""Print one timestamped diagnostic line and flush it immediately."""
function diag_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Append one diagnostic result row to `summary.csv`."""
function diag_append_row!(row::R) where {R<:NamedTuple}
    path = joinpath(DIAG_OUTDIR, "summary.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Build the exact baseline config used for the measured 256-example timing."""
function diagnostic_config()
    workers = min(32, Threads.nthreads())
    return InputFieldMNISTConfig(;
        workers,
        epochs = 0,
        batchsize = 128,
        train_per_class = 30,
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
        outdir = String(DIAG_OUTDIR),
    )
end

"""Construct independent process workers that share parameter storage with the source graph."""
function bestcase_workers(layer::L, source::G, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    C<:InputFieldMNISTConfig,
}
    algorithm = Processes.resolve(input_field_contrastive_algorithm(layer))
    return [
        input_field_worker(deepcopy(algorithm), layer, shared_worker_graph(source))
        for _ in 1:config.workers
    ]
end

"""Clear every worker-local gradient buffer before one measured minibatch."""
function clear_worker_gradients!(workers::W) where {W<:AbstractVector}
    for worker in workers
        clear_buffer!(worker_context(worker).buffers)
    end
    return workers
end

"""Run one process worker synchronously, avoiding manager dispatch and task spawning."""
@inline function run_worker_sample_direct!(worker::W, x, y, β::T) where {W,T<:Real}
    ctx = worker_context(worker)
    ctx.x .= x
    ctx.y .= y
    Processes.reset!(worker)
    Processes.loop(
        worker,
        Processes.getalgo(worker),
        Processes.context(worker),
        Processes.lifetime(worker),
        (; phase_beta = β),
    )
    return worker
end

"""Run one measured minibatch over persistent workers with static thread partitioning."""
function run_bestcase_batch!(
    workers::W,
    xtrain::X,
    ytrain::Y,
    indices::V,
    config::C,
) where {
    W<:AbstractVector,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    V<:AbstractVector{Int},
    C<:InputFieldMNISTConfig,
}
    nworkers = length(workers)
    Threads.@threads :static for worker_idx in 1:nworkers
        worker = workers[worker_idx]
        for pos in worker_idx:nworkers:length(indices)
            sample_idx = indices[pos]
            run_worker_sample_direct!(
                worker,
                @view(xtrain[:, sample_idx]),
                @view(ytrain[:, sample_idx]),
                config.β,
            )
        end
    end
    return workers
end

"""Accumulate worker-local gradients into one optimizer-facing minibatch gradient."""
function flush_bestcase_gradients!(
    batch_gradient::B,
    workers::W,
    config::C,
    nsamples::I,
) where {B,W<:AbstractVector,C<:InputFieldMNISTConfig,I<:Integer}
    clear_buffer!(batch_gradient)
    for worker in workers
        add_buffer!(batch_gradient, worker_context(worker).buffers)
    end
    scale_buffer!(batch_gradient, inv(FT(config.β) * FT(nsamples)))
    return batch_gradient
end

"""Synchronize updated parameters from the source graph into every worker graph."""
function sync_bestcase_workers!(workers::W, source::G, params::P) where {W<:AbstractVector,G,P}
    IsingLearning.sync_graph_params!(source, params)
    for worker in workers
        IsingLearning._sync_worker_graph_params!(worker_graph(worker), source, params)
    end
    return workers
end

"""Run the warmed best-case 256-example baseline training timing."""
function main()
    config = diagnostic_config()
    mkpath(DIAG_OUTDIR)
    diag_log("building baseline layer"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    relaxation_steps = setup.relaxation_steps
    work_steps_per_example = 2 * relaxation_steps

    diag_log("loading balanced train split"; train_per_class = config.train_per_class)
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    diag_log("constructing best-case workers")
    worker_seconds = @elapsed workers = bestcase_workers(setup.layer, setup.graph, config)

    params = IsingLearning.read_graph_params(setup.graph)
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), params)
    batch_gradient = IsingLearning.gradient_buffer(setup.graph)

    # Warm one sample to compile the process path without charging it to the measured 256 examples.
    diag_log("warming one sample")
    warmup_seconds = @elapsed begin
        clear_worker_gradients!(workers)
        run_worker_sample_direct!(workers[1], @view(xtrain[:, 1]), @view(ytrain[:, 1]), config.β)
        clear_worker_gradients!(workers)
    end

    measured_indices = collect(2:257)
    length(measured_indices) == 256 || error("diagnostic expected exactly 256 measured examples")
    batches = [measured_indices[1:128], measured_indices[129:256]]

    batch_rows = NamedTuple[]
    total_run = 0.0
    total_flush = 0.0
    total_update = 0.0
    total_sync = 0.0

    for (batch_idx, indices) in enumerate(batches)
        diag_log("running measured batch"; batch = batch_idx, examples = length(indices))
        clear_worker_gradients!(workers)

        run_seconds = @elapsed run_bestcase_batch!(workers, xtrain, ytrain, indices, config)
        flush_seconds = @elapsed flush_bestcase_gradients!(batch_gradient, workers, config, length(indices))
        update_seconds = @elapsed begin
            opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
        end
        sync_seconds = @elapsed sync_bestcase_workers!(workers, setup.graph, params)

        row = (;
            batch = batch_idx,
            examples = length(indices),
            run_seconds,
            flush_seconds,
            update_seconds,
            sync_seconds,
            total_seconds = run_seconds + flush_seconds + update_seconds + sync_seconds,
        )
        push!(batch_rows, row)
        diag_append_row!(row)
        diag_log("batch done"; row...)

        total_run += run_seconds
        total_flush += flush_seconds
        total_update += update_seconds
        total_sync += sync_seconds
    end

    total_examples = 256
    measured_total = total_run + total_flush + total_update + total_sync
    summary = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        measured_examples = total_examples,
        workers = config.workers,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        relaxation_steps,
        work_steps_per_example,
        total_work_steps = total_examples * work_steps_per_example,
        setup_seconds,
        data_seconds,
        worker_seconds,
        warmup_seconds,
        measured_run_seconds = total_run,
        measured_flush_seconds = total_flush,
        measured_update_seconds = total_update,
        measured_sync_seconds = total_sync,
        measured_total_seconds = measured_total,
        seconds_per_example = measured_total / total_examples,
        examples_per_second = total_examples / measured_total,
    )
    diag_append_row!(summary)
    diag_log("summary"; summary...)
    return (; summary, batch_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
