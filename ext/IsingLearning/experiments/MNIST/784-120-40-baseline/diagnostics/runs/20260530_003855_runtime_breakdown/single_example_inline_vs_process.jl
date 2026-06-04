include(joinpath(@__DIR__, "single_example_runtime.jl"))

"""Set one baseline worker context to a concrete MNIST sample."""
@inline function set_baseline_worker_sample!(ctx, xtrain::X, ytrain::Y, sample_idx::I) where {
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    state_ctx = ctx._state
    state_ctx.x .= view(xtrain, :, Int(sample_idx))
    state_ctx.y .= view(ytrain, :, Int(sample_idx))
    return ctx
end

"""Create one warmed normal `Process` worker for the baseline sample algorithm."""
function warmed_normal_process_worker(layer::L, source::G, xtrain::X, ytrain::Y) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(layer))
    worker = input_field_worker(algorithm, layer, shared_worker_graph(source))
    set_baseline_worker_sample!(StatefulAlgorithms.context(worker), xtrain, ytrain, 1)
    StatefulAlgorithms.reset!(worker)
    run(worker)
    wait(worker)
    return worker
end

"""Create one warmed synchronous `InlineProcess` worker for the baseline sample algorithm."""
function warmed_inline_process_worker(layer::L, source::G, xtrain::X, ytrain::Y) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(layer))
    warm_worker = input_field_worker(algorithm, layer, shared_worker_graph(source))
    set_baseline_worker_sample!(StatefulAlgorithms.context(warm_worker), xtrain, ytrain, 1)
    StatefulAlgorithms.reset!(warm_worker)
    run(warm_worker)
    wait(warm_worker)

    return StatefulAlgorithms.InlineProcess(
        algorithm;
        context = StatefulAlgorithms.context(warm_worker),
        repeats = 1,
        threaded = false,
    )
end

"""Return the internally recorded process runtime in seconds."""
@inline function process_internal_seconds(worker::W) where {W<:StatefulAlgorithms.Process}
    return StatefulAlgorithms.runtime(worker)
end

"""Return the internally recorded inline-process runtime in seconds."""
@inline function process_internal_seconds(worker::W) where {W<:StatefulAlgorithms.InlineProcess}
    isnothing(worker.starttime) && error("InlineProcess has not run yet")
    stop_time = isnothing(worker.endtime) ? time_ns() : worker.endtime
    return Int(stop_time - worker.starttime) / 1e9
end

"""Run one normal `Process` sample and return wall/internal timing."""
function time_one_normal_process!(worker::W, xtrain::X, ytrain::Y, sample_idx::I) where {
    W<:StatefulAlgorithms.Process,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    set_baseline_worker_sample!(StatefulAlgorithms.context(worker), xtrain, ytrain, sample_idx)
    StatefulAlgorithms.reset!(worker)
    wall = @elapsed begin
        run(worker)
        wait(worker)
    end
    return (; wall, internal = process_internal_seconds(worker))
end

"""Run one synchronous `InlineProcess` sample and return wall/internal timing."""
function time_one_inline_process!(worker::W, xtrain::X, ytrain::Y, sample_idx::I) where {
    W<:StatefulAlgorithms.InlineProcess,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    set_baseline_worker_sample!(StatefulAlgorithms.context(worker), xtrain, ytrain, sample_idx)
    wall = @elapsed run(worker; threaded = false)
    return (; wall, internal = process_internal_seconds(worker))
end

"""Append one inline-vs-normal timing row."""
function inline_vs_process_append_row!(row::R) where {R<:NamedTuple}
    path = joinpath(DIAG_OUTDIR, "single_example_inline_vs_process.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Compare warmed `InlineProcess` and normal `Process` run paths for one sample."""
function main()
    config = single_example_config()
    mkpath(DIAG_OUTDIR)
    diag_log("building inline-vs-process layer"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    relaxation_steps = setup.relaxation_steps

    diag_log("loading tiny train split")
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    diag_log("constructing and warming normal process")
    normal_build_warm_seconds = @elapsed normal_worker =
        warmed_normal_process_worker(setup.layer, setup.graph, xtrain, ytrain)

    diag_log("constructing and warming inline process")
    inline_build_warm_seconds = @elapsed inline_worker =
        warmed_inline_process_worker(setup.layer, setup.graph, xtrain, ytrain)

    diag_log("measuring normal process")
    normal = time_one_normal_process!(normal_worker, xtrain, ytrain, 2)

    diag_log("measuring inline process")
    inline = time_one_inline_process!(inline_worker, xtrain, ytrain, 2)

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        measured_examples = 1,
        workers = config.workers,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        relaxation_steps,
        work_steps_per_example = 2 * relaxation_steps,
        setup_seconds,
        data_seconds,
        normal_build_warm_seconds,
        inline_build_warm_seconds,
        normal_wall_seconds = normal.wall,
        normal_internal_seconds = normal.internal,
        inline_wall_seconds = inline.wall,
        inline_internal_seconds = inline.internal,
        normal_over_inline_wall = normal.wall / inline.wall,
        normal_over_inline_internal = normal.internal / inline.internal,
    )
    inline_vs_process_append_row!(row)
    diag_log("inline-vs-process summary"; row...)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
