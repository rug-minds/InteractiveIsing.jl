include(joinpath(@__DIR__, "bestcase_256_runtime.jl"))

"""Build a one-worker config for the 64-example single-process baseline diagnostic."""
function single_process_config()
    return InputFieldMNISTConfig(;
        workers = 1,
        epochs = 0,
        batchsize = 64,
        train_per_class = 8,
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

"""Append one row to the single-process diagnostic CSV."""
function single_process_append_row!(row::R) where {R<:NamedTuple}
    path = joinpath(DIAG_OUTDIR, "single_process_64_summary.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run one warmed single-process 64-example timing with the baseline work per sample."""
function main()
    config = single_process_config()
    mkpath(DIAG_OUTDIR)
    diag_log("building single-process baseline layer"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    relaxation_steps = setup.relaxation_steps
    work_steps_per_example = 2 * relaxation_steps

    diag_log("loading train split"; train_per_class = config.train_per_class)
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    diag_log("constructing one best-case worker")
    worker_seconds = @elapsed workers = bestcase_workers(setup.layer, setup.graph, config)

    params = IsingLearning.read_graph_params(setup.graph)
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), params)
    batch_gradient = IsingLearning.gradient_buffer(setup.graph)

    diag_log("warming one sample")
    warmup_seconds = @elapsed begin
        clear_worker_gradients!(workers)
        run_worker_sample_direct!(workers[1], @view(xtrain[:, 1]), @view(ytrain[:, 1]), config.β)
        clear_worker_gradients!(workers)
    end

    indices = collect(2:65)
    length(indices) == 64 || error("diagnostic expected exactly 64 measured examples")
    clear_worker_gradients!(workers)

    diag_log("running measured single-process batch"; examples = length(indices))
    run_seconds = @elapsed run_bestcase_batch!(workers, xtrain, ytrain, indices, config)
    flush_seconds = @elapsed flush_bestcase_gradients!(batch_gradient, workers, config, length(indices))
    update_seconds = @elapsed begin
        opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
    end
    sync_seconds = @elapsed sync_bestcase_workers!(workers, setup.graph, params)

    measured_total = run_seconds + flush_seconds + update_seconds + sync_seconds
    summary = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        measured_examples = 64,
        workers = config.workers,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        relaxation_steps,
        work_steps_per_example,
        total_work_steps = 64 * work_steps_per_example,
        setup_seconds,
        data_seconds,
        worker_seconds,
        warmup_seconds,
        measured_run_seconds = run_seconds,
        measured_flush_seconds = flush_seconds,
        measured_update_seconds = update_seconds,
        measured_sync_seconds = sync_seconds,
        measured_total_seconds = measured_total,
        seconds_per_example = measured_total / 64,
        examples_per_second = 64 / measured_total,
    )
    single_process_append_row!(summary)
    diag_log("single-process summary"; summary...)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
