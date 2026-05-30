include(joinpath(@__DIR__, "bestcase_256_runtime.jl"))

"""Return a one-worker config for measuring one warmed baseline sample."""
function single_example_config()
    return InputFieldMNISTConfig(;
        workers = 1,
        epochs = 0,
        batchsize = 1,
        train_per_class = 1,
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

"""Append the single-example timing row to its own CSV."""
function single_example_append_row!(row::R) where {R<:NamedTuple}
    path = joinpath(DIAG_OUTDIR, "single_example_summary.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Measure one warmed single-process baseline training sample."""
function main()
    config = single_example_config()
    mkpath(DIAG_OUTDIR)
    diag_log("building single-example baseline layer"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    relaxation_steps = setup.relaxation_steps
    work_steps_per_example = 2 * relaxation_steps

    diag_log("loading tiny train split")
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    diag_log("constructing one worker")
    worker_seconds = @elapsed workers = bestcase_workers(setup.layer, setup.graph, config)
    worker = only(workers)

    diag_log("warming one sample")
    warmup_seconds = @elapsed begin
        clear_worker_gradients!(workers)
        run_worker_sample_direct!(worker, @view(xtrain[:, 1]), @view(ytrain[:, 1]), config.β)
        clear_worker_gradients!(workers)
    end

    diag_log("running measured single sample")
    sample_idx = 2
    measured_seconds = @elapsed begin
        run_worker_sample_direct!(worker, @view(xtrain[:, sample_idx]), @view(ytrain[:, sample_idx]), config.β)
    end

    summary = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        measured_examples = 1,
        workers = config.workers,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        relaxation_steps,
        work_steps_per_example,
        total_work_steps = work_steps_per_example,
        setup_seconds,
        data_seconds,
        worker_seconds,
        warmup_seconds,
        measured_sample_seconds = measured_seconds,
        examples_per_second = inv(measured_seconds),
    )
    single_example_append_row!(summary)
    diag_log("single-example summary"; summary...)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
