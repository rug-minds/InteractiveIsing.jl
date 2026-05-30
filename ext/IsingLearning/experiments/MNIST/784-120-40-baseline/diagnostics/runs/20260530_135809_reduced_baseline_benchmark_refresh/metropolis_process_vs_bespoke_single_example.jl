include(joinpath(@__DIR__, "hidden_output_field_input_single_example.jl"))

"""Return internal runtime seconds for `Process` and `InlineProcess` diagnostics."""
@inline function diagnostic_internal_seconds(worker::W) where {W<:Processes.Process}
    return Processes.runtime(worker)
end

"""Return internal runtime seconds for a synchronous `InlineProcess` diagnostic."""
@inline function diagnostic_internal_seconds(worker::W) where {W<:Processes.InlineProcess}
    isnothing(worker.starttime) && error("InlineProcess has not run yet")
    stop_time = isnothing(worker.endtime) ? time_ns() : worker.endtime
    return Int(stop_time - worker.starttime) / 1e9
end

"""Copy one sample into a baseline worker context and reset the worker."""
function prepare_baseline_process_sample!(worker::W, xtrain::X, ytrain::Y, sample_idx::I) where {
    W,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    ctx = worker_context(worker)
    load_sample_into_worker!(ctx, xtrain, ytrain, sample_idx)
    Processes.reset!(worker)
    return worker
end

"""Create one warmed normal Process for the reduced Metropolis baseline algorithm."""
function warmed_metropolis_process_worker(layer::L, source::G, input_hidden_w::R, xtrain::X, ytrain::Y) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    R<:Base.RefValue,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    algorithm = Processes.resolve(input_field_contrastive_algorithm(layer))
    worker = input_field_worker(algorithm, layer, shared_worker_graph(source), input_hidden_w)
    prepare_baseline_process_sample!(worker, xtrain, ytrain, 1)
    run(worker)
    wait(worker)
    prepare_baseline_process_sample!(worker, xtrain, ytrain, 1)
    run(worker)
    wait(worker)
    return worker
end

"""Create one warmed InlineProcess sharing a warmed Metropolis worker context."""
function warmed_metropolis_inline_worker(layer::L, source::G, input_hidden_w::R, xtrain::X, ytrain::Y) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    R<:Base.RefValue,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    warm_worker = warmed_metropolis_process_worker(layer, source, input_hidden_w, xtrain, ytrain)
    algorithm = Processes.resolve(input_field_contrastive_algorithm(layer))
    worker = Processes.InlineProcess(
        algorithm;
        context = Processes.context(warm_worker),
        repeats = 1,
        threaded = false,
    )
    run(worker; threaded = false)
    return worker
end

"""Time one normal Process run path for a single Metropolis MNIST sample."""
function time_metropolis_process!(worker::W, xtrain::X, ytrain::Y, sample_idx::I) where {
    W<:Processes.Process,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    prepare_baseline_process_sample!(worker, xtrain, ytrain, sample_idx)
    wall = @elapsed begin
        run(worker)
        wait(worker)
    end
    return (; wall, internal = diagnostic_internal_seconds(worker))
end

"""Time one InlineProcess run path for a single Metropolis MNIST sample."""
function time_metropolis_inline!(worker::W, xtrain::X, ytrain::Y, sample_idx::I) where {
    W<:Processes.InlineProcess,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
}
    ctx = Processes.context(worker)._state
    load_sample_into_worker!(ctx, xtrain, ytrain, sample_idx)
    wall = @elapsed run(worker; threaded = false)
    return (; wall, internal = diagnostic_internal_seconds(worker))
end

"""Append one exact Metropolis comparison row to disk."""
function append_metropolis_comparison_row!(row::R) where {R<:NamedTuple}
    path = joinpath(BESPOKE_OUTDIR, "metropolis_process_vs_bespoke_single_example.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Build the reduced hidden/output Metropolis layer used for exact process-vs-bespoke timing."""
function reduced_metropolis_process_setup(config::C) where {C<:InputFieldMNISTConfig}
    reduced = build_reduced_field_input_layer(config)
    dynamics = II.Metropolis()
    layer = IsingLearning.LayeredIsingGraphLayer(
        reduced.graph;
        input_idxs = Base.OneTo(INPUT_DIM),
        output_idxs = II.layerrange(reduced.graph[end]),
        β = config.β,
        free_relaxation_steps = reduced.relaxation_steps,
        nudged_relaxation_steps = reduced.relaxation_steps,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return merge(reduced, (; layer))
end

"""Compare direct bespoke, normal Process, and InlineProcess with reduced Metropolis dynamics."""
function main()
    config = bespoke_single_example_config()
    setup = reduced_metropolis_process_setup(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    direct_graph = setup.graph
    buffers = input_field_gradient_buffer(direct_graph, setup.input_hidden_w)
    input_w_gradient = similar(setup.input_hidden_w)
    free_state = similar(II.state(direct_graph))
    nudged_state = similar(II.state(direct_graph))
    rng = Random.MersenneTwister(config.seed + 91_001)
    input_hidden_w_ref = Ref(setup.input_hidden_w)

    bespoke_log("warming reduced exact Metropolis direct path")
    reduced_field_input_contrastive_sample!(
        direct_graph,
        setup.input_hidden_w,
        view(xtrain, :, 1),
        view(ytrain, :, 1),
        buffers,
        input_w_gradient,
        free_state,
        nudged_state,
        setup.active_idxs,
        setup.ptr_bounds,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    bespoke_log("measuring reduced exact Metropolis direct path")
    direct_wall = @elapsed direct_stats = reduced_field_input_contrastive_sample!(
        direct_graph,
        setup.input_hidden_w,
        view(xtrain, :, 2),
        view(ytrain, :, 2),
        buffers,
        input_w_gradient,
        free_state,
        nudged_state,
        setup.active_idxs,
        setup.ptr_bounds,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    bespoke_log("warming reduced exact Metropolis normal Process")
    normal_worker = warmed_metropolis_process_worker(setup.layer, setup.graph, input_hidden_w_ref, xtrain, ytrain)
    bespoke_log("measuring reduced exact Metropolis normal Process")
    normal = time_metropolis_process!(normal_worker, xtrain, ytrain, 2)

    bespoke_log("warming reduced exact Metropolis InlineProcess")
    inline_worker = warmed_metropolis_inline_worker(setup.layer, setup.graph, input_hidden_w_ref, xtrain, ytrain)
    bespoke_log("measuring reduced exact Metropolis InlineProcess")
    inline = time_metropolis_inline!(inline_worker, xtrain, ytrain, 2)

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "120-40_field_input_from_784",
        dynamics = "reduced_continuous_metropolis",
        measured_examples = 1,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        beta = config.β,
        temp = config.temp,
        relaxation_steps = setup.relaxation_steps,
        work_steps_per_example = 2 * setup.relaxation_steps,
        direct_wall_seconds = direct_wall,
        direct_free_seconds = direct_stats.free_seconds,
        direct_nudged_seconds = direct_stats.nudged_seconds,
        direct_gradient_seconds = direct_stats.gradient_seconds,
        direct_normalize_seconds = direct_stats.normalize_seconds,
        normal_process_wall_seconds = normal.wall,
        normal_process_internal_seconds = normal.internal,
        inline_process_wall_seconds = inline.wall,
        inline_process_internal_seconds = inline.internal,
        normal_over_direct_wall = normal.wall / direct_wall,
        inline_over_direct_wall = inline.wall / direct_wall,
        normal_over_inline_wall = normal.wall / inline.wall,
    )
    csv_path = append_metropolis_comparison_row!(row)
    bespoke_log("exact Metropolis comparison summary"; row..., csv = csv_path)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
