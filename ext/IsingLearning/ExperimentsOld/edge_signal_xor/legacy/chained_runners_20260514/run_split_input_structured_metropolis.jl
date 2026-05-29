include("run_split_input_structured_search.jl")

"""Build the deterministic split-edge graph with discrete Ising spins."""
function structured_split_input_metropolis_graph(config::EdgeSignalXORConfig)
    HIDDEN_HEIGHT_REF[] = config.hidden_height
    HIDDEN_WIDTH_REF[] = config.hidden_width
    rng_hidden = Random.MersenneTwister(config.weight_seed + 1)
    rng_hidden_output = Random.MersenneTwister(config.weight_seed + 2)
    rng_bias = Random.MersenneTwister(config.bias_seed)

    layer_input = II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), II.Discrete(), II.Coords(0, 0, 0))
    layer_hidden = II.Layer(
        config.hidden_height,
        config.hidden_width,
        II.StateSet(-one(FT), one(FT)),
        hidden_local_generator(config.hidden_local_scale, config.hidden_nn, rng_hidden),
        II.Discrete(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    layer_output = II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Discrete(), II.Coords(0, 0, 0))

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

"""Wrap the discrete graph with Metropolis dynamics."""
function structured_metropolis_layer(graph, config::EdgeSignalXORConfig)
    dynamics = II.IsingMetropolis()
    return LayeredIsingGraphLayer(
        () -> structured_split_input_metropolis_graph(config);
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

"""Initialize a discrete Metropolis trainer for the structured edge task."""
function structured_metropolis_trainer(config::EdgeSignalXORConfig)
    graph = structured_split_input_metropolis_graph(config)
    layer = structured_metropolis_layer(graph, config)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(config.lr))
end

"""Run one Metropolis validation trajectory and average the scalar output."""
function metropolis_timeavg_scalar_output!(trainer, x, config::EdgeSignalXORConfig; seed::Integer, burnin_sweeps::Integer, average_sweeps::Integer)
    graph = trainer.validation_graph
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)
    dynamics = II.IsingMetropolis()
    output_idx = only(II.layerrange(graph[end]))
    averager = EdgeOutputAverager(output_idx, Int(burnin_sweeps), Int(average_sweeps))
    sweep_steps = length(II.sampling_indices(graph.index_set))
    total_sweeps = burnin_sweeps + average_sweeps
    routine = Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @alias averager = averager
        @every sweep_steps averager(model = dynamics.model)
    end
    wrapped = Processes.@Routine begin
        @repeat (total_sweeps * sweep_steps) routine()
    end
    inputs = (Processes.Init(dynamics, model = graph, rng = Random.MersenneTwister(seed)),)
    process = Processes.Process(Processes.resolve(wrapped), inputs...; repeats = 1)
    run(process)
    wait(process)
    ctx = Processes.context(process).averager
    μ = edge_average_mean(ctx)
    σ = edge_average_std(ctx)
    close(process)
    return μ, σ
end

"""Evaluate the discrete Metropolis edge task by repeated time-averaged outputs."""
function evaluate_metropolis_timeavg!(trainer, x, y, config::EdgeSignalXORConfig; seed_offset::Integer, burnin_sweeps::Integer, average_sweeps::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            μ, _ = metropolis_timeavg_scalar_output!(
                trainer,
                view(x, :, sample_idx),
                config;
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
                burnin_sweeps,
                average_sweeps,
            )
            samples[repeat_idx] = μ
        end
        means[sample_idx] = mean(samples)
        stds[sample_idx] = std(samples)
    end
    targets = vec(y)
    return (;
        mse = mean(abs2, means .- targets),
        acc = mean(sign.(means) .== sign.(targets)),
        margin = minimum(abs.(means)),
        means,
        stds,
    )
end

"""Train the deterministic edge task with discrete Metropolis dynamics."""
function train_structured_metropolis_xor(config::EdgeSignalXORConfig; outdir, eval_burnin_sweeps::Int, eval_average_sweeps::Int)
    trainer = structured_metropolis_trainer(config)
    x, y = xor_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]
    metrics = evaluate_metropolis_timeavg!(
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
            metrics = evaluate_metropolis_timeavg!(
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

"""Return discrete Metropolis candidate settings for the edge task."""
function structured_metropolis_specs()
    common = (;
        epochs = 12000,
        log_every = 1000,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 80,
        nudged_sweeps = 120,
        validation_sweeps = 200,
        hidden_nn = 5,
        input_hidden_scale = FT(0.8),
        hidden_local_scale = FT(0.015),
        hidden_output_scale = FT(0.5),
    )
    raw = [
        (; name = "metro_b0p5_T0p015_lr4e4_wd2e3", β = FT(0.50), temp_fraction = FT(0.015), lr = FT(0.00040), wd = FT(0.002), seed = 1_200_001),
        (; name = "metro_b0p8_T0p020_lr3e4_wd3e3", β = FT(0.80), temp_fraction = FT(0.020), lr = FT(0.00030), wd = FT(0.003), seed = 1_200_101),
        (; name = "metro_b0p35_T0p010_lr5e4_wd1e3", β = FT(0.35), temp_fraction = FT(0.010), lr = FT(0.00050), wd = FT(0.001), seed = 1_200_201),
    ]
    specs = []
    for spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            temp_fraction = spec.temp_fraction,
            stepsize = FT(1.0),
            lr = spec.lr,
            weight_decay = spec.wd,
            base_seed = spec.seed,
        )
        push!(specs, (; spec.name, config = cfg.config, active_count = cfg.active_count))
    end
    return specs
end

"""Run the structured split-edge Metropolis search."""
function run_structured_metropolis_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_structured_metropolis_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 150
    eval_average_sweeps = 150
    for spec in structured_metropolis_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_structured_metropolis_xor(spec.config; outdir, eval_burnin_sweeps, eval_average_sweeps)
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
            "lr" => spec.config.lr,
            "weight_decay" => spec.config.weight_decay,
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "summary.csv"), rows)
        if trained.best.acc == 1.0 && trained.best.mse < 0.12
            println("Stopping grid after successful setting: ", spec.name)
            break
        end
    end
    println("Saved structured Metropolis run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_structured_metropolis_search()
end
