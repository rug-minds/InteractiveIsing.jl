using Dates
using Optimisers

const LANGEVIN_LEARNING_DIR = @__DIR__
const LANGEVIN_LEARNING_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(LANGEVIN_LEARNING_BASELINE)

"""Append one row for the reduced-graph LocalLangevin learning comparison."""
function append_langevin_learning_row!(row::R) where {R<:NamedTuple}
    path = joinpath(LANGEVIN_LEARNING_DIR, "local_langevin_learning_vs_process.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the real training config used for the full learning-step benchmark."""
function langevin_learning_config()
    return InputFieldMNISTConfig(;
        workers = 1,
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
        outdir = String(LANGEVIN_LEARNING_DIR),
    )
end

"""Run one already-initialized dynamics context for the requested number of steps."""
function relax_learning_context!(algorithm::A, context::C, nsteps::I) where {A,C,I<:Integer}
    @inbounds for _ in 1:Int(nsteps)
        StatefulAlgorithms.step!(algorithm, context)
    end
    return context
end

"""Synchronize one direct reduced-graph setup after an optimizer update."""
function sync_direct_learning_params!(graph::G, input_hidden_w::W, params::P) where {
    G,
    W<:AbstractMatrix,
    P<:NamedTuple,
}
    IsingLearning.sync_graph_params!(graph, (; w = params.w, b = params.b))
    input_hidden_w .= params.w_input
    return graph
end

"""Run one direct reduced-graph LocalLangevin learning sample without `StatefulAlgorithms.Process` overhead."""
function direct_learning_sample!(
    graph::G,
    layer::L,
    input_hidden_w::W,
    x::X,
    y::Y,
    buffers::B,
    input_pattern::P,
    equilibrium_state::S,
    nudged_state::S,
    free_context::FC,
    nudged_context::NC,
) where {
    G,
    L<:IsingLearning.LayeredIsingGraphLayer,
    W<:AbstractMatrix,
    X<:AbstractVector,
    Y<:AbstractVector,
    B,
    P<:AbstractVector,
    S<:AbstractVector,
    FC,
    NC,
}
    # Match the current one-sided field-input training path exactly.
    II.resetstate!(graph)
    project_input_field_pattern!(input_pattern, input_hidden_w, x)
    install_input_field_pattern!(graph, input_pattern)
    relax_learning_context!(layer.dynamics_algorithm, free_context, layer.free_relaxation_steps)
    copyto!(equilibrium_state, II.state(graph))

    II.state(graph) .= equilibrium_state
    install_input_field_pattern!(graph, input_pattern)
    IsingLearning.apply_targets(graph, y)
    IsingLearning.set_clamping_beta!(graph, layer.β)
    relax_learning_context!(layer.nudged_dynamics_algorithm, nudged_context, layer.nudged_relaxation_steps)
    copyto!(nudged_state, II.state(graph))

    IsingLearning.set_clamping_beta!(graph, zero(FT))
    accumulate_input_field_gradient!(graph, nudged_state, equilibrium_state, x, buffers, layer.β)
    return nothing
end

"""Run one full serial bespoke minibatch plus Adam update on the reduced graph."""
function time_direct_learning_minibatch!(setup, xtrain::X, ytrain::Y, config::C) where {
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:InputFieldMNISTConfig,
}
    graph = setup.graph
    layer = setup.layer
    input_hidden_w = copy(setup.input_hidden_w)
    sync_direct_learning_params!(graph, input_hidden_w, input_field_params(graph, input_hidden_w))

    batch_gradient = input_field_gradient_buffer(graph, input_hidden_w)
    input_pattern = zeros(FT, II.nstates(graph))
    equilibrium_state = copy(II.state(graph))
    nudged_state = similar(equilibrium_state)
    free_context = StatefulAlgorithms.init(layer.dynamics_algorithm, (; model = graph))
    nudged_context = StatefulAlgorithms.init(layer.nudged_dynamics_algorithm, (; model = graph))
    initial_params = input_field_params(graph, input_hidden_w)
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)

    first_idx = 1
    last_idx = config.batchsize
    run_batch! = function ()
        clear_buffer!(batch_gradient)
        @inbounds for sample_idx in first_idx:last_idx
            direct_learning_sample!(
                graph,
                layer,
                input_hidden_w,
                view(xtrain, :, sample_idx),
                view(ytrain, :, sample_idx),
                batch_gradient,
                input_pattern,
                equilibrium_state,
                nudged_state,
                free_context,
                nudged_context,
            )
        end
        scale_buffer!(batch_gradient, inv(FT(config.β) * FT(config.batchsize)))
        opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
        sync_direct_learning_params!(graph, input_hidden_w, params)
        return nothing
    end

    run_batch!()
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
    sync_direct_learning_params!(graph, input_hidden_w, initial_params)

    wall = @elapsed begin
        run_batch!()
    end
    return (; wall, seconds_per_example = wall / config.batchsize)
end

"""Run one full serial `Process` minibatch plus Adam update on the reduced graph."""
function time_serial_process_learning_minibatch!(setup, xtrain::X, ytrain::Y, config::C) where {
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:InputFieldMNISTConfig,
}
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    initial_params = input_field_params(source_graph, input_hidden_w_ref[])
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
    batch_gradient = input_field_gradient_buffer(source_graph, input_hidden_w_ref[])
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)

    try
        run_batch! = function ()
            clear_buffer!(worker_context(worker).buffers)
            @inbounds for sample_idx in 1:config.batchsize
                load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
                StatefulAlgorithms.reset!(worker)
                StatefulAlgorithms.runprocessinline!(worker; phase_beta = config.β)
            end
            clear_buffer!(batch_gradient)
            add_buffer!(batch_gradient, worker_context(worker).buffers)
            scale_buffer!(batch_gradient, inv(FT(config.β) * FT(config.batchsize)))
            opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
            IsingLearning.sync_graph_params!(source_graph, (; w = params.w, b = params.b))
            input_hidden_w_ref[] = params.w_input
            return nothing
        end

        run_batch!()
        params = initial_params
        opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
        IsingLearning.sync_graph_params!(source_graph, (; w = initial_params.w, b = initial_params.b))
        input_hidden_w_ref[] = initial_params.w_input
        clear_buffer!(worker_context(worker).buffers)

        wall = @elapsed begin
            run_batch!()
        end
        return (; wall, seconds_per_example = wall / config.batchsize)
    finally
        close(worker)
    end
end

"""Run one full 1-worker manager minibatch plus Adam update on the reduced graph."""
function time_manager_learning_minibatch!(setup, xtrain::X, ytrain::Y, config::C) where {
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:InputFieldMNISTConfig,
}
    manager = input_field_manager(setup.layer, setup.graph, config, Ref(copy(setup.input_hidden_w)))
    jobs_buffer = InputFieldMNISTChunkBuffer(1, config.batchsize)
    indices = collect(1:config.batchsize)

    try
        set_manager_inputs!(manager, xtrain, ytrain)
        jobs = fill_chunk_jobs!(jobs_buffer, indices, config.batchsize)
        initial_params = manager.state.params[]
        run_batch! = function ()
            clear_manager_buffers!(manager)
            manager.state.nsamples[] = sum(length, jobs)
            StatefulAlgorithms.run!(manager, jobs)
            manager.state.opt_state, params = Optimisers.update(
                manager.state.opt_state,
                manager.state.params[],
                manager.state.batch_gradient,
            )
            manager.state.params[] = params
            sync_after_update!(manager, params)
            return nothing
        end

        run_batch!()
        manager.state.params[] = initial_params
        manager.state.opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
        IsingLearning.sync_graph_params!(manager.state.source_graph, (; w = initial_params.w, b = initial_params.b))
        manager.state.input_hidden_w[] = initial_params.w_input
        clear_manager_buffers!(manager)

        wall = @elapsed begin
            run_batch!()
        end
        return (; wall, seconds_per_example = wall / config.batchsize)
    finally
        close(manager)
    end
end

"""Benchmark direct bespoke, serial `Process`, and 1-worker manager learning minibatches."""
function main()
    mkpath(LANGEVIN_LEARNING_DIR)
    config = langevin_learning_config()
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    direct_setup = build_layer(config)
    process_setup = build_layer(config)
    manager_setup = build_layer(config)
    direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
    serial_process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config)
    manager = time_manager_learning_minibatch!(manager_setup, xtrain, ytrain, config)

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "120-40_field_input_from_784",
        dynamics = "reduced_local_langevin_full_learning_minibatch",
        threads = Threads.nthreads(),
        batchsize = config.batchsize,
        sweeps = config.sweeps,
        beta = config.β,
        temp = config.temp,
        stepsize = config.stepsize,
        relaxation_steps = direct_setup.relaxation_steps,
        direct_bespoke_seconds = direct.wall,
        direct_bespoke_seconds_per_example = direct.seconds_per_example,
        serial_process_seconds = serial_process.wall,
        serial_process_seconds_per_example = serial_process.seconds_per_example,
        manager_seconds = manager.wall,
        manager_seconds_per_example = manager.seconds_per_example,
        process_over_bespoke = serial_process.wall / direct.wall,
        manager_over_bespoke = manager.wall / direct.wall,
        manager_over_process = manager.wall / serial_process.wall,
    )
    csv_path = append_langevin_learning_row!(row)
    println(row)
    println("csv=" * csv_path)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
