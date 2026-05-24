using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "local_checkerboard_xor", "checkerboard_4x8_no_seed_grid.jl"))

"""
    SplitSnapshotSearchConfig(; kwargs...)

Experiment-local wrapper for the split-snapshot Langevin idea.

The free phase stores the early snapshot in the shared `equilibrium_state`
field used by the existing EqProp composite. The free dynamics model then keeps
running to the late endpoint, and that late model is used as the free state in
the contrastive gradient.
"""
Base.@kwdef struct SplitSnapshotSearchConfig
    search::StabilizedSearchConfig
    early_relaxation::Int = 25
    max_drift_fraction::FT = FT(0.15)
    save_threshold::FT = 0.25
    notes::String = ""
end

"""
    split_checker_dynamics(config, split)

Create the Langevin sampler for this split-snapshot run. This is separate from
the shared helper so the experiment can tune `max_drift_fraction` without
changing toolbox code.
"""
function split_checker_dynamics(config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig)
    if config.dynamics_mode === :local_langevin
        return II.LocalLangevin(
            stepsize = config.stepsize,
            max_drift_fraction = split.max_drift_fraction,
            adjusted = false,
            order = :random,
            group_steps = 1,
        )
    elseif config.dynamics_mode === :langevin
        return II.BlockLangevin(
            stepsize = config.stepsize,
            max_drift_fraction = split.max_drift_fraction,
            adjusted = false,
            block_size = config.block_size,
            group_steps = 1,
        )
    elseif config.dynamics_mode === :global_langevin
        return II.GlobalLangevin(
            stepsize = config.stepsize,
            max_drift_fraction = split.max_drift_fraction,
            adjusted = false,
            group_steps = 1,
        )
    else
        throw(ArgumentError("split-snapshot search needs Langevin dynamics, got $(config.dynamics_mode)"))
    end
end

"""
    split_free_dynamics_algorithm(config, split, graph)

Create the Langevin sampler used for the split free phase. This experiment uses
the direct sampler instead of a scheduler wrapper so the Processes context shape
matches the working checkerboard EqProp route.
"""
function split_free_dynamics_algorithm(config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig, graph)
    return split_checker_dynamics(config, split)
end

"""
    split_nudged_dynamics_algorithm(config, split, graph, name)

Create a Langevin sampler for one nudged branch. The `name` parameter is kept in
the signature for the experiment API.
"""
function split_nudged_dynamics_algorithm(config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig, graph, name::Val{Name}) where {Name}
    return split_checker_dynamics(config, split)
end

"""
    SplitSnapshotForwardDynamics(layer, config, split)

Free relaxation routine for the split-snapshot experiment. It records
`equilibrium_state` early for compatibility with the existing EqProp composite
state merge, then continues the model to the late free endpoint.
"""
function SplitSnapshotForwardDynamics(layer, config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig)
    n_units = layer.nunits
    early_steps = max(1, split.early_relaxation)
    late_steps = max(1, layer.free_relaxation_steps - early_steps)
    dynamics_algorithm = split_free_dynamics_algorithm(config, split, layer.model_graph)

    forward = @Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        stable_prepare_free_state!(dynamics.model, config)
        stable_apply_input!(dynamics.model, x, config, split.search)
        @repeat early_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(m -> II.state(m), dynamics.model))
        @repeat late_steps dynamics()
    end
    return (; algorithm = forward, dynamics = forward.dynamics)
end

"""
    SplitSnapshotNudgedDynamics(layer, config, split)

Plus/minus nudged routines for the split-snapshot experiment. Both nudged
branches restore from the shared `equilibrium_state` field, which the forward
routine deliberately fills with the early free snapshot.
"""
function SplitSnapshotNudgedDynamics(layer, config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    plus_dynamics_algorithm = split_nudged_dynamics_algorithm(config, split, layer.model_graph, Val(:split_plus))
    minus_dynamics_algorithm = split_nudged_dynamics_algorithm(config, split, layer.model_graph, Val(:split_minus))

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = beta
        @alias dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        stable_apply_input!(dynamics.model, x, config, split.search)
        checker_apply_targets!(dynamics.model, y)
        checker_set_clamping_beta!(dynamics.model, nudged_beta)
        model = @repeat relaxation_steps dynamics()
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = -beta
        @alias dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        stable_apply_input!(dynamics.model, x, config, split.search)
        checker_apply_targets!(dynamics.model, y)
        checker_set_clamping_beta!(dynamics.model, nudged_beta)
        model = @repeat relaxation_steps dynamics()
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @input clamping_beta = beta
        @state buffers
        @alias plus = plus
        @alias minus = minus
        plus.nudged_beta = clamping_beta
        @context c1 = plus()
        minus.nudged_beta = @transform(x -> -x, clamping_beta)
        @context c2 = minus()
    end
    return (; algorithm = final, plus, minus, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""
    SplitSnapshotForwardAndNudged(layer, config, split)

Composite training step for split-snapshot Langevin. The gradient uses the free
endpoint graph, while the nudged states came from the earlier free snapshot.
"""
function SplitSnapshotForwardAndNudged(layer, config::LocalCheckerboardConfig, split::SplitSnapshotSearchConfig)
    forward = SplitSnapshotForwardDynamics(layer, config, split).algorithm
    nudged = SplitSnapshotNudgedDynamics(layer, config, split)
    beta = layer.β
    final = @CompositeAlgorithm begin
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
        checker_set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.dynamics.model, plus_capture.captured, minus_capture.captured, clamping_beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

"""
    split_snapshot_worker_process(layer, graph, split)

Create one training worker for the split-snapshot Langevin composite.
"""
function split_snapshot_worker_process(layer, graph, split::SplitSnapshotSearchConfig)
    config = split.search.config
    algo = Processes.resolve(SplitSnapshotForwardAndNudged(layer, config, split).algorithm)
    buffers = IsingLearning.gradient_buffer(graph)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, target_dim(config)),
            buffers = buffers,
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed),
        Init(:plus_capture, state = graph),
        Init(:minus_capture, state = graph);
        repeat = 1,
    )
end

"""
    init_split_snapshot_trainer(layer, split; graph, optimiser)

Initialize a `CheckerTrainer` whose training workers use the split-snapshot
composite and whose validation worker remains the ordinary free relaxation.
"""
function init_split_snapshot_trainer(
    layer,
    split::SplitSnapshotSearchConfig;
    graph = layer.model_graph,
    optimiser = Optimisers.Adam(split.search.config.lr),
)
    config = split.search.config
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    workers = Process[]
    worker_graphs = typeof(graph)[]
    for _ in 1:config.workers
        wg = IsingLearning._worker_graph(graph, params)
        II.temp!(wg, effective_temp(wg, config))
        push!(worker_graphs, wg)
        push!(workers, split_snapshot_worker_process(layer, wg, split))
    end
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, effective_temp(validation_graph, config))
    validation_worker = stable_checker_validation_process(layer, validation_graph, split.search)
    return CheckerTrainer(layer, graph, params, opt_state, worker_graphs, workers, validation_graph, validation_worker, optimiser)
end

"""
    run_split_snapshot_config(split, outdir)

Train one split-snapshot Langevin configuration and save its best graph if the
configured MSE threshold is reached.
"""
function run_split_snapshot_config(split::SplitSnapshotSearchConfig, outdir)
    search = split.search
    config = search.config
    mkpath(outdir)
    graph = checkerboard_graph(config)
    assert_no_extra_local_potentials!(graph)
    clip_graph_parameters!(graph, search)
    layer = checkerboard_layer(graph, config)
    trainer = init_split_snapshot_trainer(layer, split; graph, optimiser = Optimisers.Adam(config.lr))
    x, y = xor_inputs_targets(config)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    rows = Dict{String,Any}[]
    best_params = deepcopy(trainer.params)
    best_mse = Inf
    best_acc = -Inf
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    eval_seed = config.base_seed + 50_000_000
    metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
    push!(rows, metric_row(config.name, 0, metrics, zero_grad, trainer, initial_params))
    print_metrics(0, metrics, zero_grad)
    if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
        best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
    end

    for epoch in 1:config.epochs
        grad = train_epoch_stable!(trainer, x, y, batch_gradient, epoch, search)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
            push!(rows, metric_row(config.name, epoch, metrics, grad, trainer, initial_params))
            print_metrics(epoch, metrics, grad)
            if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
                best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
            end
        end
    end

    trainer.params = best_params
    _broadcast_params!(trainer)
    graph_path = nothing
    svg_path = nothing
    if best_mse <= split.save_threshold
        strip_weight_generators!(trainer.prototype_graph)
        graph_path = II.save_isinggraph(joinpath(outdir, "$(config.name)_best_graph.jld2"), trainer.prototype_graph)
        svg_path = write_parameter_svg(joinpath(outdir, "$(config.name)_parameters.svg"), trainer.prototype_graph, config)
    end
    close_checker_trainer!(trainer)
    return (; rows, graph_path, svg_path, best_mse, best_acc, notes = split.notes)
end

"""
    split_snapshot_searches()

Return the initial grid for the split-snapshot Langevin experiment. It starts
with a small `2x2 -> 4x4 -> 2x2` topology and then tries the larger corrected
checkerboard topology.
"""
function split_snapshot_searches()
    epochs = parse(Int, get(ENV, "ISING_SPLIT_EPOCHS", "900"))
    log_every = parse(Int, get(ENV, "ISING_SPLIT_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_SPLIT_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_SPLIT_MINIT", "6"))
    eval_repeats = parse(Int, get(ENV, "ISING_SPLIT_EVAL_REPEATS", "20"))

    common = (;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        weight_decay = FT(0),
        grad_clip = FT(100),
        temp_is_factor = true,
        internal_nn = 1,
        input_internal_scale = FT(0),
        hidden_internal_scale = FT(0.06),
        output_internal_scale = FT(0.06),
        bias_scale = FT(0.02),
        weight_seed = 53,
        internal_seed = 59,
        bias_seed = 61,
        base_seed = 125000,
        init_mode = :random,
        state_mode = :continuous,
        output_clamp_mode = :pattern,
        doublewell_barrier = FT(0),
        free_temp_start_factor = FT(1.0),
        free_temp_stop_factor = FT(1.0),
        nudged_temp_stop_factor = FT(1.0),
        temp_schedule_power = FT(1.0),
    )

    specs = (
        (; name = "simple_local_s020_d035_T020_b10", side = 2, hidden = 4, radius = FT(2.25), inter = FT(0.16), mode = :local_langevin, block = 8, step = FT(0.20), drift = FT(0.35), temp = FT(0.020), beta = FT(1.0), lr = FT(0.0025), free = 500, early = 40, nudged = 700),
        (; name = "simple_local_s040_d060_T020_b10", side = 2, hidden = 4, radius = FT(2.25), inter = FT(0.16), mode = :local_langevin, block = 8, step = FT(0.40), drift = FT(0.60), temp = FT(0.020), beta = FT(1.0), lr = FT(0.0020), free = 500, early = 30, nudged = 800),
        (; name = "simple_local_s080_d100_T015_b10", side = 2, hidden = 4, radius = FT(2.25), inter = FT(0.16), mode = :local_langevin, block = 8, step = FT(0.80), drift = FT(1.00), temp = FT(0.015), beta = FT(1.0), lr = FT(0.0015), free = 600, early = 25, nudged = 900),
        (; name = "simple_local_s120_d100_T010_b15", side = 2, hidden = 4, radius = FT(2.25), inter = FT(0.18), mode = :local_langevin, block = 8, step = FT(1.20), drift = FT(1.00), temp = FT(0.010), beta = FT(1.5), lr = FT(0.0012), free = 700, early = 25, nudged = 1000),
        (; name = "big_local_s040_d060_T020_b10", side = 4, hidden = 8, radius = FT(2.25), inter = FT(0.14), mode = :local_langevin, block = 16, step = FT(0.40), drift = FT(0.60), temp = FT(0.020), beta = FT(1.0), lr = FT(0.0020), free = 1200, early = 80, nudged = 1600),
        (; name = "big_local_s080_d100_T015_b10", side = 4, hidden = 8, radius = FT(2.25), inter = FT(0.14), mode = :local_langevin, block = 16, step = FT(0.80), drift = FT(1.00), temp = FT(0.015), beta = FT(1.0), lr = FT(0.0015), free = 1400, early = 80, nudged = 1800),
    )

    splits = [
        SplitSnapshotSearchConfig(
            search = StabilizedSearchConfig(
                config = LocalCheckerboardConfig(;
                    common...,
                    name = spec.name,
                    side = spec.side,
                    hidden_side = spec.hidden,
                    code_side = spec.side,
                    code_stride = 1,
                    code_offset = (1, 1),
                    inter_radius = spec.radius,
                    inter_weight_scale = spec.inter,
                    dynamics_mode = spec.mode,
                    block_size = spec.block,
                    stepsize = spec.step,
                    temp = spec.temp,
                    β = spec.beta,
                    lr = spec.lr,
                    free_relaxation = spec.free,
                    nudged_relaxation = spec.nudged,
                ),
                full_bipolar_input = true,
                save_threshold = FT(0.20),
                notes = "split-snapshot Langevin: mode=$(spec.mode), side=$(spec.side), hidden=$(spec.hidden), stepsize=$(spec.step), max_drift=$(spec.drift), constant Tfactor=$(spec.temp), β=$(spec.beta), free=$(spec.free), early=$(spec.early), nudged=$(spec.nudged)",
            ),
            early_relaxation = spec.early,
            max_drift_fraction = spec.drift,
            save_threshold = FT(0.20),
            notes = "nudged starts from early free snapshot; gradient graph is full free endpoint",
        )
        for spec in specs
    ]
    only = get(ENV, "ISING_SPLIT_ONLY", "")
    if !isempty(only)
        splits = [split for split in splits if occursin(only, split.search.config.name)]
    end
    limit = parse(Int, get(ENV, "ISING_SPLIT_LIMIT", string(length(splits))))
    return splits[1:min(limit, length(splits))]
end

"""
    write_split_snapshot_readme(path, splits, results, csv_path, png_path)

Write a compact run document explaining the split-snapshot intervention and
the result table.
"""
function write_split_snapshot_readme(path, splits, results, csv_path, png_path)
    open(path, "w") do io
        println(io, "# Langevin Split-Snapshot XOR Search")
        println(io)
        println(io, "This run tests a radical Langevin-only EqProp variant. The free phase captures an early state and a later free endpoint. Plus/minus nudged phases restart from the early state, but the contrastive gradient is computed with the free endpoint graph.")
        println(io)
        println(io, "All configs use explicit bipolar frozen input, no input internal weights, output pattern clamping, symmetric adjacency, and no polynomial/double-well local potential.")
        println(io)
        println(io, "| Config | Best MSE | Best Acc | Saved | Notes |")
        println(io, "|---|---:|---:|---|---|")
        for (split, result) in zip(splits, results)
            cfg = split.search.config
            saved = isnothing(result.graph_path) ? "no" : "yes"
            println(io, "| `$(cfg.name)` | $(round(result.best_mse, digits=6)) | $(round(result.best_acc, digits=3)) | $saved | $(split.search.notes) |")
        end
        println(io)
        println(io, "Metrics CSV: `$(basename(csv_path))`")
        println(io, "Progress PNG: `$(basename(png_path))`")
    end
    return path
end

"""
    main()

Run the split-snapshot Langevin grid and save CSV, PNG, README, and any graph
that crosses the configured MSE threshold.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_SPLIT_DIR", joinpath(@__DIR__, "runs", "langevin_split_snapshot_$timestamp"))
    splits = split_snapshot_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, split) in enumerate(splits)
        cfg = split.search.config
        println("Running split-snapshot Langevin $idx/$(length(splits)): $(cfg.name)")
        result = run_split_snapshot_config(split, joinpath(outdir, cfg.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(cfg.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "langevin_split_snapshot_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "langevin_split_snapshot_progress.png"), all_rows)
    md_path = write_split_snapshot_readme(joinpath(outdir, "README.md"), splits, results, csv_path, png_path)
    println("Saved metrics: $csv_path")
    println("Saved plot: $png_path")
    println("Saved docs: $md_path")
    return (; outdir, splits, results, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
