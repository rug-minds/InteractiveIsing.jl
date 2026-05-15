include("run_split_input_timeavg_search.jl")

const EDGE_FEATURE_INPUT_SCALE_REF = Ref{FT}(0.8)
const EDGE_FEATURE_LOCAL_SCALE_REF = Ref{FT}(0.12)

"""
    edge_feature_input_weight(; c1, c2)

Map the two scalar inputs into four row bands on the first hidden edge.
Rows `1:2` detect `(+x1, -x2)`, rows `3:4` detect `(-x1, +x2)`,
rows `5:6` detect `(+x1, +x2)`, and rows `7:8` detect `(-x1, -x2)`.
"""
function edge_feature_input_weight(; c1, c2)
    left_hidden_edge(c2) || return zero(FT)
    row = c2[1]
    input = c1[1]
    if row <= 2
        return input == 1 ? EDGE_FEATURE_INPUT_SCALE_REF[] : -EDGE_FEATURE_INPUT_SCALE_REF[]
    elseif row <= 4
        return input == 1 ? -EDGE_FEATURE_INPUT_SCALE_REF[] : EDGE_FEATURE_INPUT_SCALE_REF[]
    elseif row <= 6
        return EDGE_FEATURE_INPUT_SCALE_REF[]
    else
        return -EDGE_FEATURE_INPUT_SCALE_REF[]
    end
end

"""Create the deterministic two-input to four-feature-edge generator."""
function edge_feature_input_generator(scale::FT)
    EDGE_FEATURE_INPUT_SCALE_REF[] = scale
    return @WG edge_feature_input_weight NN = :all symmetric = true
end

"""Return a simple ferromagnetic local hidden coupling for signal propagation."""
function edge_feature_hidden_local(; dr)
    return dr == 1 ? EDGE_FEATURE_LOCAL_SCALE_REF[] : zero(FT)
end

"""Create the hidden local coupling generator for the feature-band experiment."""
function edge_feature_hidden_local_generator(scale::FT)
    EDGE_FEATURE_LOCAL_SCALE_REF[] = scale
    return @WG edge_feature_hidden_local NN = 1 symmetric = true
end

"""Build the `2 -> 8x8 -> 1` feature-band edge graph."""
function edge_feature_graph(config::EdgeSignalXORConfig)
    HIDDEN_HEIGHT_REF[] = config.hidden_height
    HIDDEN_WIDTH_REF[] = config.hidden_width
    rng_hidden_output = Random.MersenneTwister(config.weight_seed + 2)
    rng_bias = Random.MersenneTwister(config.bias_seed)

    layer_input = II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))
    layer_hidden = II.Layer(
        config.hidden_height,
        config.hidden_width,
        II.StateSet(-one(FT), one(FT)),
        edge_feature_hidden_local_generator(config.hidden_local_scale),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    layer_output = II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = edge_feature_input_generator(config.input_hidden_scale)
    hidden_output = right_edge_to_output_generator(config.hidden_output_scale, rng_hidden_output)

    b = g -> config.bias_scale .* randn(rng_bias, FT, II.statelen(g))
    y = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = b) +
        II.Clamping(β = II.UniformArray(zero(FT)), y = y, mask = mask)

    graph = II.IsingGraph(
        layer_input,
        input_hidden,
        layer_hidden,
        hidden_output,
        layer_output,
        hamiltonian;
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    set_relative_temperature!(graph, config)
    return graph
end

"""Wrap the feature-band graph in the existing learning-layer interface."""
function edge_feature_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> edge_feature_graph(config);
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_relaxation,
        free_relaxation_steps = config.free_relaxation,
        nudged_relaxation_steps = config.nudged_relaxation,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Initialize a feature-band trainer with optional nudged-temperature bump."""
function edge_feature_trainer(config::EdgeSignalXORConfig, nudged_temp_factor::FT)
    graph = edge_feature_graph(config)
    layer = edge_feature_layer(graph, config)
    if nudged_temp_factor == one(FT)
        return init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(config.lr))
    end
    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
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

"""Train the feature-band edge graph with time-averaged validation."""
function train_edge_feature_xor(config::EdgeSignalXORConfig; outdir, nudged_temp_factor::FT, eval_burnin_sweeps::Int, eval_average_sweeps::Int)
    trainer = edge_feature_trainer(config, nudged_temp_factor)
    x, y = xor_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_timeavg!(
        trainer,
        x,
        y,
        config;
        seed_offset = config.base_seed + 30_000_000,
        burnin_sweeps = eval_burnin_sweeps,
        average_sweeps = eval_average_sweeps,
    )
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
            metrics = evaluate_timeavg!(
                trainer,
                x,
                y,
                config;
                seed_offset = config.base_seed + 30_000_000,
                burnin_sweeps = eval_burnin_sweeps,
                average_sweeps = eval_average_sweeps,
            )
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_learning_row!(rows, epoch, metrics, grad_norm, II.temp(trainer.prototype_graph))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4),
                " means=", round.(metrics.means, digits = 3))
            if metrics.acc == 1.0 && metrics.mse < 0.12
                println("early success at epoch ", epoch)
                break
            end
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

"""Return feature-band edge candidate settings."""
function edge_feature_specs()
    common = (;
        epochs = 8000,
        log_every = 500,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 30,
        nudged_sweeps = 60,
        validation_sweeps = 120,
        hidden_nn = 1,
        input_hidden_scale = FT(0.85),
        hidden_local_scale = FT(0.12),
        hidden_output_scale = FT(0.35),
        bias_scale = FT(0.02),
    )
    raw = [
        (; name = "feature_b0p35_T0p012_eta1p0_lr4e4_wd1e3", β = FT(0.35), temp_fraction = FT(0.012), stepsize = FT(1.0), lr = FT(0.0004), wd = FT(0.001), bump = FT(1.15), seed = 1_230_001),
        (; name = "feature_b0p50_T0p010_eta1p0_lr3e4_wd1e3", β = FT(0.50), temp_fraction = FT(0.010), stepsize = FT(1.0), lr = FT(0.0003), wd = FT(0.001), bump = FT(1.20), seed = 1_230_101),
    ]
    specs = []
    for spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            temp_fraction = spec.temp_fraction,
            stepsize = spec.stepsize,
            lr = spec.lr,
            weight_decay = spec.wd,
            base_seed = spec.seed,
        )
        push!(specs, (; spec.name, config = cfg.config, active_count = cfg.active_count, nudged_temp_factor = spec.bump))
    end
    return specs
end

"""Run the feature-band edge XOR search."""
function run_edge_feature_search(; rootdir = joinpath(@__DIR__, "runs", "edge_feature_bands_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 80
    eval_average_sweeps = 80
    for spec in edge_feature_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_edge_feature_xor(
            spec.config;
            outdir,
            nudged_temp_factor = spec.nudged_temp_factor,
            eval_burnin_sweeps,
            eval_average_sweeps,
        )
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => spec.name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "beta" => spec.config.β,
            "temp_fraction" => spec.config.temp_fraction,
            "stepsize" => spec.config.stepsize,
            "lr" => spec.config.lr,
            "weight_decay" => spec.config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "summary.csv"), rows)
        if trained.best.acc == 1.0 && trained.best.mse < 0.12
            println("Stopping grid after successful setting: ", spec.name)
            break
        end
    end
    println("Saved edge feature-band run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_edge_feature_search()
end
