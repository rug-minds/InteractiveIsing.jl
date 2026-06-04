include("edge_signal_split_input_search.jl")

"""Build the split-input plus/minus nudged phase with a temporary temperature bump."""
function split_input_nudged_temp_algorithm(layer, base_temp::FT, nudged_temp::FT)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    plus_dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    minus_dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)

    plus = StatefulAlgorithms.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = beta
        @alias dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, nudged_beta)
        II.temp!(dynamics.model, nudged_temp)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, base_temp)
        plus_capture(isinggraph = model)
    end

    minus = StatefulAlgorithms.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = -beta
        @alias dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, nudged_beta)
        II.temp!(dynamics.model, nudged_temp)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, base_temp)
        minus_capture(isinggraph = model)
    end

    final = StatefulAlgorithms.@CompositeAlgorithm begin
        @input clamping_beta = beta
        @alias plus = plus
        @alias minus = minus
        plus.nudged_beta = clamping_beta
        @context c1 = plus()
        minus.nudged_beta = @transform(x -> -x, clamping_beta)
        @context c2 = minus()
    end
    return (; algorithm = final, plus, minus, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""Build free dynamics plus split-input nudged dynamics with a small temp bump."""
function split_input_forward_and_nudged_temp(layer, base_temp::FT, nudged_temp::FT)
    forward = IsingLearning.ForwardDynamics(layer).algorithm
    nudged = split_input_nudged_temp_algorithm(layer, base_temp, nudged_temp)
    beta = layer.β
    final = StatefulAlgorithms.@CompositeAlgorithm begin
        @input clamping_beta = beta
        @state buffers
        @alias plus = nudged.plus
        @alias minus = nudged.minus
        @alias plus_capture = nudged.plus_capture
        @alias minus_capture = nudged.minus_capture
        @context c1 = forward()
        plus.nudged_beta = clamping_beta
        plus()
        plus_capture(isinggraph = plus.dynamics.model)
        minus.nudged_beta = @transform(x -> -x, clamping_beta)
        minus()
        minus_capture(isinggraph = minus.dynamics.model)
        IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.dynamics.model, plus_capture.captured, minus_capture.captured, clamping_beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

"""Create one worker process for the split-input nudged-temperature routine."""
function split_input_nudged_temp_worker_process(layer, worker_graph, base_temp::FT, nudged_temp::FT)
    algo = StatefulAlgorithms.resolve(split_input_forward_and_nudged_temp(layer, base_temp, nudged_temp).algorithm)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = IsingLearning.gradient_buffer(worker_graph)

    return StatefulAlgorithms.Process(
        algo,
        StatefulAlgorithms.Init(:_state;
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(II.state(worker_graph)),
        ),
        StatefulAlgorithms.Init(:dynamics, model = worker_graph),
        StatefulAlgorithms.Init(:plus_capture, state = worker_graph),
        StatefulAlgorithms.Init(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

"""Initialize a split-input trainer with a nudged-only temperature multiplier."""
function split_input_nudged_temp_trainer(config::EdgeSignalXORConfig, nudged_temp_factor::FT)
    graph = split_input_edge_graph(config)
    layer = split_input_edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    base_temp = II.temp(graph)
    nudged_temp = nudged_temp_factor * base_temp

    worker_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(worker_graph, base_temp)
    worker = split_input_nudged_temp_worker_process(layer, worker_graph, base_temp, nudged_temp)
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, base_temp)
    validation_worker = IsingLearning._validation_process(layer, validation_graph)

    return IsingLearning.MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        [worker_graph],
        [worker],
        validation_graph,
        validation_worker,
        optimiser,
    )
end

"""Train one split-input graph with optional weight decay and nudged temp bump."""
function train_split_input_regularized_xor(config::EdgeSignalXORConfig; outdir, nudged_temp_factor::FT = one(FT))
    trainer = nudged_temp_factor == one(FT) ? split_input_edge_trainer(config) :
        split_input_nudged_temp_trainer(config, nudged_temp_factor)
    x, y = xor_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_learning_row!(rows, 0, metrics, zero(FT), II.temp(trainer.prototype_graph))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    initial_params = deepcopy(trainer.params)
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
        " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if config.weight_decay > 0
            trainer.params.w .*= (one(FT) - config.lr * config.weight_decay)
            IsingLearning._broadcast_params!(trainer)
        end
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_learning_row!(rows, epoch, metrics, grad_norm, II.temp(trainer.prototype_graph))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4),
                " means=", round.(metrics.means, digits = 3))
        end
    end

    learning_csv = write_csv(joinpath(outdir, "learning_metrics.csv"), rows)
    learning_png = plot_learning(joinpath(outdir, "learning_progress.png"), rows)
    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    best_graph_path = II.save_isinggraph(joinpath(outdir, "best_graph.jld2"), strip_weight_generators!(deepcopy(trainer.prototype_graph)))
    trainer.params = initial_params
    IsingLearning._broadcast_params!(trainer)
    initial_graph_path = II.save_isinggraph(joinpath(outdir, "initial_graph.jld2"), strip_weight_generators!(deepcopy(trainer.prototype_graph)))
    close_trainer!(trainer)
    return (; best, rows, learning_csv, learning_png, best_graph_path, initial_graph_path, best_params, initial_params)
end

"""Return regularization and small nudged-temperature trials for split-input XOR."""
function split_input_regularized_trials()
    base = (;
        epochs = 3600,
        log_every = 600,
        minit = 3,
        eval_repeats = 16,
        free_relaxation = 1200,
        nudged_relaxation = 1200,
        validation_relaxation = 3500,
        stepsize = FT(0.8),
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = 5,
        β = FT(0.20),
        lr = FT(0.00050),
        temp_fraction = FT(0.018),
        input_hidden_scale = FT(0.45),
        hidden_local_scale = FT(0.005),
        hidden_output_scale = FT(0.40),
        bias_scale = FT(0.0),
        skip_response = true,
        weight_seed = 703,
        bias_seed = 723,
    )
    return [
        ("baseline_T0018", EdgeSignalXORConfig(; weight_decay = FT(0), base_seed = 1_090_703, base...), one(FT)),
        ("wd_1e_minus_2", EdgeSignalXORConfig(; weight_decay = FT(1e-2), base_seed = 1_091_703, base...), one(FT)),
        ("wd_5e_minus_2", EdgeSignalXORConfig(; weight_decay = FT(5e-2), base_seed = 1_092_703, base...), one(FT)),
        ("nudged_Tx1p25", EdgeSignalXORConfig(; weight_decay = FT(0), base_seed = 1_093_703, base...), FT(1.25)),
        ("wd_1e_minus_2_nudged_Tx1p25", EdgeSignalXORConfig(; weight_decay = FT(1e-2), base_seed = 1_094_703, base...), FT(1.25)),
    ]
end

"""Run the split-input regularization scout and save a summary CSV."""
function run_split_input_regularized_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_regularized_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, config, nudged_temp_factor) in split_input_regularized_trials()
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_regularized_xor(config; outdir, nudged_temp_factor)
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "weight_decay" => config.weight_decay,
            "nudged_temp_factor" => nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved regularized split-input search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_regularized_search()
end
