include(joinpath(@__DIR__, "single_example_runtime.jl"))

"""Run one prepared sample through the normal `Process` async path."""
function run_worker_sample_process!(worker::W, x, y, β::T) where {W,T<:Real}
    ctx = worker_context(worker)
    ctx.x .= x
    ctx.y .= y
    StatefulAlgorithms.reset!(worker)
    run(worker; phase_beta = β)
    wait(worker)
    return worker
end

"""Append one loop-vs-process comparison row."""
function loop_vs_process_append_row!(row::R) where {R<:NamedTuple}
    path = joinpath(DIAG_OUTDIR, "single_example_loop_vs_process.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Compare direct loop execution to the normal `Process` run path for one sample."""
function main()
    config = single_example_config()
    mkpath(DIAG_OUTDIR)
    diag_log("building loop-vs-process layer"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    relaxation_steps = setup.relaxation_steps

    diag_log("loading tiny train split")
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    diag_log("constructing one worker")
    worker_seconds = @elapsed workers = bestcase_workers(setup.layer, setup.graph, config)
    worker = only(workers)

    # Warm both call paths. The measured samples reuse the same prepared process.
    diag_log("warming direct loop and normal process")
    warm_loop_seconds = @elapsed run_worker_sample_direct!(worker, @view(xtrain[:, 1]), @view(ytrain[:, 1]), config.β)
    warm_process_seconds = @elapsed run_worker_sample_process!(worker, @view(xtrain[:, 1]), @view(ytrain[:, 1]), config.β)

    diag_log("measuring direct loop sample")
    direct_loop_seconds = @elapsed run_worker_sample_direct!(worker, @view(xtrain[:, 2]), @view(ytrain[:, 2]), config.β)

    diag_log("measuring normal process sample")
    normal_process_seconds = @elapsed run_worker_sample_process!(worker, @view(xtrain[:, 2]), @view(ytrain[:, 2]), config.β)

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        workers = config.workers,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        relaxation_steps,
        work_steps_per_example = 2 * relaxation_steps,
        setup_seconds,
        data_seconds,
        worker_seconds,
        warm_loop_seconds,
        warm_process_seconds,
        direct_loop_seconds,
        normal_process_seconds,
        process_over_loop = normal_process_seconds / direct_loop_seconds,
    )
    loop_vs_process_append_row!(row)
    diag_log("loop-vs-process summary"; row...)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
