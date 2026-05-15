include("run_split_input_timeavg_search.jl")

const STRUCTURED_INPUT_SCALE_REF = Ref{FT}(0.7)

"""
    structured_split_input_weight(; c1, c2)

Return a same-sign input coupling into the left hidden edge.
Input spin 1 drives the upper half; input spin 2 drives the lower half.
"""
function structured_split_input_weight(; c1, c2)
    left_hidden_edge(c2) || return zero(FT)
    top_half = c2[1] <= HIDDEN_HEIGHT_REF[] ÷ 2
    input_row = c1[1]
    if (input_row == 1 && top_half) || (input_row == 2 && !top_half)
        return STRUCTURED_INPUT_SCALE_REF[]
    end
    return zero(FT)
end

"""Create a deterministic split-input generator with no random sign disorder."""
function structured_split_input_generator(scale::FT)
    STRUCTURED_INPUT_SCALE_REF[] = scale
    return @WG structured_split_input_weight NN = :all symmetric = true
end

"""Build the split-edge graph with deterministic input-to-edge couplings."""
function structured_split_input_graph(config::EdgeSignalXORConfig)
    HIDDEN_HEIGHT_REF[] = config.hidden_height
    HIDDEN_WIDTH_REF[] = config.hidden_width
    rng_hidden = Random.MersenneTwister(config.weight_seed + 1)
    rng_hidden_output = Random.MersenneTwister(config.weight_seed + 2)
    rng_bias = Random.MersenneTwister(config.bias_seed)

    layer_input = II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))
    layer_hidden = II.Layer(
        config.hidden_height,
        config.hidden_width,
        II.StateSet(-one(FT), one(FT)),
        hidden_local_generator(config.hidden_local_scale, config.hidden_nn, rng_hidden),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    layer_output = II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = structured_split_input_generator(config.input_hidden_scale)
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

"""Wrap the deterministic split-input graph in the learning-layer interface."""
function structured_split_input_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> structured_split_input_graph(config);
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

"""Initialize a deterministic split-input trainer with optional nudged T bump."""
function structured_split_input_trainer(config::EdgeSignalXORConfig, nudged_temp_factor::FT)
    graph = structured_split_input_graph(config)
    layer = structured_split_input_layer(graph, config)
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

"""Train the deterministic split-input edge graph with time-averaged validation."""
function train_structured_split_input_xor(
    config::EdgeSignalXORConfig;
    outdir,
    nudged_temp_factor::FT,
    eval_burnin_sweeps::Int,
    eval_average_sweeps::Int,
)
    trainer = structured_split_input_trainer(config, nudged_temp_factor)
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

"""Return deterministic split-input candidate settings."""
function structured_split_input_specs()
    common = (;
        epochs = 8000,
        log_every = 500,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 30,
        nudged_sweeps = 80,
        validation_sweeps = 140,
        hidden_nn = 5,
        input_hidden_scale = FT(0.75),
        hidden_local_scale = FT(0.012),
        hidden_output_scale = FT(0.50),
        bias_scale = FT(0.10),
    )
    raw = [
        (; name = "structured_bias_b0p35_T0p010_eta1p0_lr25e4_wd4e3", β = FT(0.35), temp_fraction = FT(0.010), stepsize = FT(1.0), lr = FT(0.00025), wd = FT(0.004), bump = FT(1.25), seed = 1_190_001),
        (; name = "structured_bias_b0p50_T0p010_eta1p0_lr20e4_wd5e3", β = FT(0.50), temp_fraction = FT(0.010), stepsize = FT(1.0), lr = FT(0.00020), wd = FT(0.005), bump = FT(1.30), seed = 1_190_101),
        (; name = "structured_bias_b0p35_T0p007_eta1p2_lr18e4_wd4e3", β = FT(0.35), temp_fraction = FT(0.007), stepsize = FT(1.2), lr = FT(0.00018), wd = FT(0.004), bump = FT(1.35), seed = 1_190_201),
        (; name = "structured_bias_b0p25_T0p007_eta1p2_lr25e4_wd3e3", β = FT(0.25), temp_fraction = FT(0.007), stepsize = FT(1.2), lr = FT(0.00025), wd = FT(0.003), bump = FT(1.25), seed = 1_190_301),
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

"""Run the deterministic split-input edge search."""
function run_structured_split_input_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_structured_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 100
    eval_average_sweeps = 100
    for spec in structured_split_input_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_structured_split_input_xor(
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
    println("Saved structured split-input run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_structured_split_input_search()
end
