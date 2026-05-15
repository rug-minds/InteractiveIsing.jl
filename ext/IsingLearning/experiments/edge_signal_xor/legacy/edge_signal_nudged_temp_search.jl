include("edge_signal_xor.jl")

"""
    NudgedTempEdgeTrial

One 8x8 edge-signal trial with a nudged-only temperature multiplier. The free
phase learns from the normal temperature, while the plus/minus clamped phases
can start from the free state at a hotter temperature.
"""
Base.@kwdef struct NudgedTempEdgeTrial
    name::String
    hidden_nn::Int
    β::FT
    lr::FT
    stepsize::FT
    temp_fraction::FT
    nudged_temp_factor::FT
    input_hidden_scale::FT
    hidden_local_scale::FT
    hidden_output_scale::FT
    free_relaxation::Int
    nudged_relaxation::Int
    validation_relaxation::Int
    minit::Int
    eval_repeats::Int
    epochs::Int
    log_every::Int
    weight_seed::Int
end

"""Convert one nudged-temperature trial into the base edge config."""
function edge_config(trial::NudgedTempEdgeTrial)
    return EdgeSignalXORConfig(
        epochs = trial.epochs,
        log_every = trial.log_every,
        minit = trial.minit,
        eval_repeats = trial.eval_repeats,
        free_relaxation = trial.free_relaxation,
        nudged_relaxation = trial.nudged_relaxation,
        validation_relaxation = trial.validation_relaxation,
        β = trial.β,
        lr = trial.lr,
        stepsize = trial.stepsize,
        temp_fraction = trial.temp_fraction,
        input_hidden_scale = trial.input_hidden_scale,
        hidden_local_scale = trial.hidden_local_scale,
        hidden_output_scale = trial.hidden_output_scale,
        bias_scale = FT(0.02),
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = trial.hidden_nn,
        target_scale = one(FT),
        skip_response = true,
        weight_seed = trial.weight_seed,
        bias_seed = trial.weight_seed + 23,
        base_seed = 840_000 + 1000 * trial.weight_seed,
    )
end

"""Build the plus/minus nudged phase with a temporary temperature bump."""
function edge_nudged_temp_algorithm(layer, base_temp::FT, nudged_temp::FT)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    plus_dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    minus_dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)

    plus = Processes.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, beta)
        II.temp!(dynamics.model, nudged_temp)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, base_temp)
        plus_capture(isinggraph = model)
    end

    minus = Processes.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, -beta)
        II.temp!(dynamics.model, nudged_temp)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, base_temp)
        minus_capture(isinggraph = model)
    end

    final = Processes.@CompositeAlgorithm begin
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""Build the normal free phase plus the temperature-bumped nudged phase."""
function edge_forward_and_nudged_temp(layer, base_temp::FT, nudged_temp::FT)
    forward = IsingLearning.ForwardDynamics(layer).algorithm
    nudged = edge_nudged_temp_algorithm(layer, base_temp, nudged_temp)
    beta = layer.β
    final = Processes.@CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = nudged.algorithm()
        IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

"""Create one worker process using the experiment-local nudged-temperature routine."""
function edge_nudged_temp_worker_process(layer, worker_graph, base_temp::FT, nudged_temp::FT)
    algo = Processes.resolve(edge_forward_and_nudged_temp(layer, base_temp, nudged_temp).algorithm)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = IsingLearning.gradient_buffer(worker_graph)

    return Processes.Process(
        algo,
        Processes.Init(:_state;
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(II.state(worker_graph)),
        ),
        Processes.Init(:dynamics, model = worker_graph),
        Processes.Init(:plus_capture, state = worker_graph),
        Processes.Init(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

"""Initialize the normal threaded trainer, replacing only the worker process."""
function edge_nudged_temp_trainer(config::EdgeSignalXORConfig, nudged_temp_factor::FT)
    graph = edge_signal_graph(config)
    layer = edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    base_temp = II.temp(graph)
    nudged_temp = nudged_temp_factor * base_temp

    worker_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(worker_graph, base_temp)
    worker = edge_nudged_temp_worker_process(layer, worker_graph, base_temp, nudged_temp)
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

"""Train one 8x8 edge graph with nudged-temperature workers."""
function train_edge_xor_nudged_temp(config::EdgeSignalXORConfig, nudged_temp_factor::FT; outdir)
    trainer = edge_nudged_temp_trainer(config, nudged_temp_factor)
    x, y = xor_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_learning_row!(rows, 0, metrics, zero(FT), II.temp(trainer.prototype_graph))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    initial_params = deepcopy(trainer.params)
    println("NN=", config.hidden_nn, " epoch=0 mse=", round(metrics.mse, digits = 6),
        " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_learning_row!(rows, epoch, metrics, grad_norm, II.temp(trainer.prototype_graph))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("NN=", config.hidden_nn, " epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4),
                " means=", round.(metrics.means, digits = 3))
        end
    end

    learning_csv = write_csv(joinpath(outdir, "learning_metrics.csv"), rows)
    learning_png = plot_learning(joinpath(outdir, "learning_progress.png"), rows)

    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    best_graph = strip_weight_generators!(deepcopy(trainer.prototype_graph))
    best_graph_path = II.save_isinggraph(joinpath(outdir, "best_graph.jld2"), best_graph)

    trainer.params = initial_params
    IsingLearning._broadcast_params!(trainer)
    initial_graph = strip_weight_generators!(deepcopy(trainer.prototype_graph))
    initial_graph_path = II.save_isinggraph(joinpath(outdir, "initial_graph.jld2"), initial_graph)

    close_trainer!(trainer)
    return (; best, rows, learning_csv, learning_png, best_graph_path, initial_graph_path, best_params, initial_params)
end

"""Return 8x8 trials focused on clamp strength, temperature, and nudged heat."""
function nudged_temp_edge_trials()
    base = (;
        epochs = 3000,
        log_every = 300,
        minit = 2,
        eval_repeats = 12,
        free_relaxation = 900,
        nudged_relaxation = 900,
        validation_relaxation = 1800,
        stepsize = FT(0.8),
    )
    return NudgedTempEdgeTrial[
        NudgedTempEdgeTrial(; name = "NN2_beta0p20_T0p04_nT2_edge0p25_h0p012", hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.04), nudged_temp_factor = FT(2.0),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.25),
            weight_seed = 301, base...),
        NudgedTempEdgeTrial(; name = "NN2_beta0p20_T0p04_nT4_edge0p25_h0p012", hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.04), nudged_temp_factor = FT(4.0),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.25),
            weight_seed = 302, base...),
        NudgedTempEdgeTrial(; name = "NN3_beta0p20_T0p035_nT3_edge0p30_h0p008", hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035), nudged_temp_factor = FT(3.0),
            input_hidden_scale = FT(0.30), hidden_local_scale = FT(0.008), hidden_output_scale = FT(0.30),
            weight_seed = 303, base...),
        NudgedTempEdgeTrial(; name = "NN2_beta0p35_T0p035_nT2_edge0p25_h0p012", hidden_nn = 2,
            β = FT(0.35), lr = FT(0.0006), temp_fraction = FT(0.035), nudged_temp_factor = FT(2.0),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.25),
            weight_seed = 304, base...),
    ]
end

"""Run the 8x8 nudged-temperature search and summarize its results."""
function run_nudged_temp_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_nudged_temp_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for trial in nudged_temp_edge_trials()
        println("\n=== ", trial.name, " ===")
        config = edge_config(trial)
        outdir = joinpath(rootdir, trial.name)
        mkpath(outdir)
        trained = train_edge_xor_nudged_temp(config, trial.nudged_temp_factor; outdir)
        final = isempty(trained.rows) ? nothing : trained.rows[end]
        push!(summary, Dict{String,Any}(
            "name" => trial.name,
            "hidden_nn" => trial.hidden_nn,
            "beta" => trial.β,
            "lr" => trial.lr,
            "temp_fraction" => trial.temp_fraction,
            "nudged_temp_factor" => trial.nudged_temp_factor,
            "input_hidden_scale" => trial.input_hidden_scale,
            "hidden_local_scale" => trial.hidden_local_scale,
            "hidden_output_scale" => trial.hidden_output_scale,
            "best_epoch" => trained.best.epoch,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "final_mse" => final === nothing ? missing : final["mse"],
            "final_accuracy" => final === nothing ? missing : final["accuracy"],
            "outdir" => outdir,
        ))
        write_run_readme(joinpath(outdir, "README.md"), config, trained, nothing)
        push!(results, (; trial, config, trained, outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), summary)
    write_nudged_temp_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved nudged-temperature edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write a short result table for the nudged-temperature edge search."""
function write_nudged_temp_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Edge Signal XOR With Nudged Temperature Bump")
        println(io)
        println(io, "The free phase runs at the configured base temperature. The plus/minus nudged phases run at `nudged_temp_factor * base_temperature`, then restore the base temperature.")
        println(io)
        println(io, "| trial | NN | beta | T fraction | nudged T factor | edge scale | hidden scale | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            println(io, "| `$(row["name"])` | $(row["hidden_nn"]) | $(row["beta"]) | $(row["temp_fraction"]) | $(row["nudged_temp_factor"]) | $(row["input_hidden_scale"]) | $(row["hidden_local_scale"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_nudged_temp_edge_search()
end
