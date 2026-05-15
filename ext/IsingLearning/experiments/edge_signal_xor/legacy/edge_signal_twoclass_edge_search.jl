include("edge_signal_xor.jl")

const TWOCLASS_OUTPUT_SCALE_REF = Ref{FT}(0.20)
const TWOCLASS_OUTPUT_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))

"""Signed random connection from hidden right edge into the full two-class output layer."""
function hidden_to_twoclass_output_weight(; c1)
    return right_hidden_edge(c1) ? TWOCLASS_OUTPUT_SCALE_REF[] * randn(TWOCLASS_OUTPUT_RNG_REF[], FT) : zero(FT)
end

"""Create hidden-right-edge to two-class output couplings."""
function hidden_to_twoclass_output_generator(scale::FT, rng)
    TWOCLASS_OUTPUT_SCALE_REF[] = scale
    TWOCLASS_OUTPUT_RNG_REF[] = rng
    return @WG hidden_to_twoclass_output_weight NN = :all symmetric = true
end

"""
    twoclass_edge_graph(config)

Build `2 input spins -> 8x8 hidden -> 8x2 output`. Inputs still only touch the
hidden left edge, and output units still only see the hidden right edge. The
larger output layer gives the classifier two output populations instead of one
scalar spin.
"""
function twoclass_edge_graph(config::EdgeSignalXORConfig)
    HIDDEN_WIDTH_REF[] = config.hidden_width
    rng_input_hidden = Random.MersenneTwister(config.weight_seed)
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
    layer_output = II.Layer(8, 2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = input_to_left_edge_generator(config.input_hidden_scale, rng_input_hidden)
    hidden_output = hidden_to_twoclass_output_generator(config.hidden_output_scale, rng_hidden_output)

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

"""Return two-spin XOR inputs and a bipolar two-class output population target."""
function twoclass_edge_dataset(config::EdgeSignalXORConfig)
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 16, 4)
    for (idx, (a, b)) in enumerate(XOR_CASES)
        x[:, idx] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        false_sign = xor(a, b) ? -one(FT) : one(FT)
        true_sign = xor(a, b) ? one(FT) : -one(FT)
        y[1:8, idx] .= config.target_scale * false_sign
        y[9:16, idx] .= config.target_scale * true_sign
    end
    return x, y
end

"""Wrap the two-class edge graph in the standard IsingLearning layer interface."""
function twoclass_edge_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> twoclass_edge_graph(config);
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

"""Initialize a trainer for the two-class output-edge experiment."""
function twoclass_edge_trainer(config::EdgeSignalXORConfig)
    graph = twoclass_edge_graph(config)
    layer = twoclass_edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Return false-class and true-class scores from an output state vector."""
function twoclass_scores(output_state)
    false_score = mean(@view output_state[1:8])
    true_score = mean(@view output_state[9:16])
    return false_score, true_score
end

"""Run validation for one input and return the class-score difference."""
function twoclass_edge_score!(trainer, x, config::EdgeSignalXORConfig; seed::Integer)
    graph = trainer.validation_graph
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)
    run_dynamics_steps!(graph, edge_sampler(config), config.validation_relaxation; seed)
    false_score, true_score = twoclass_scores(copy(II.state(graph[end])))
    return true_score - false_score
end

"""Evaluate repeated-start two-class output differences for all XOR inputs."""
function evaluate_twoclass_edge!(trainer, x, y, config::EdgeSignalXORConfig; seed_offset::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    targets = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        target_false, target_true = twoclass_scores(view(y, :, sample_idx))
        targets[sample_idx] = target_true - target_false
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = twoclass_edge_score!(
                trainer,
                view(x, :, sample_idx),
                config;
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
            )
        end
        means[sample_idx] = mean(samples)
        stds[sample_idx] = std(samples)
    end
    return (;
        mse = mean(abs2, means .- targets),
        acc = mean(sign.(means) .== sign.(targets)),
        margin = minimum(abs.(means)),
        means,
        stds,
    )
end

"""Train one two-class output-edge graph and save diagnostics."""
function train_twoclass_edge_xor(config::EdgeSignalXORConfig; outdir)
    trainer = twoclass_edge_trainer(config)
    x, y = twoclass_edge_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_twoclass_edge!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_learning_row!(rows, 0, metrics, zero(FT), II.temp(trainer.prototype_graph))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    initial_params = deepcopy(trainer.params)
    println("NN=", config.hidden_nn, " epoch=0 mse=", round(metrics.mse, digits = 6),
        " acc=", metrics.acc, " scores=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_twoclass_edge!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_learning_row!(rows, epoch, metrics, grad_norm, II.temp(trainer.prototype_graph))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("NN=", config.hidden_nn, " epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4),
                " scores=", round.(metrics.means, digits = 3))
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

"""Return the first focused two-class edge search grid."""
function twoclass_edge_trials()
    base = (;
        epochs = 5000,
        log_every = 500,
        minit = 2,
        eval_repeats = 12,
        free_relaxation = 1000,
        nudged_relaxation = 1000,
        validation_relaxation = 2500,
        stepsize = FT(0.8),
    )
    return [
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.014), hidden_output_scale = FT(0.30),
            bias_scale = FT(0.01), weight_seed = 601, bias_seed = 621, base_seed = 990_601,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.030),
            input_hidden_scale = FT(0.30), hidden_local_scale = FT(0.009), hidden_output_scale = FT(0.35),
            bias_scale = FT(0.01), weight_seed = 602, bias_seed = 622, base_seed = 990_602,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 5,
            β = FT(0.20), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.35), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.01), weight_seed = 603, bias_seed = 623, base_seed = 990_603,
            skip_response = true, base...),
    ]
end

"""Run the two-class output-edge search."""
function run_twoclass_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_twoclass_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for config in twoclass_edge_trials()
        name = "NN$(config.hidden_nn)_beta$(replace(string(config.β), "." => "p"))_T$(replace(string(config.temp_fraction), "." => "p"))"
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_twoclass_edge_xor(config; outdir)
        final = isempty(trained.rows) ? nothing : trained.rows[end]
        push!(summary, Dict{String,Any}(
            "name" => name,
            "hidden_nn" => config.hidden_nn,
            "beta" => config.β,
            "temp_fraction" => config.temp_fraction,
            "input_hidden_scale" => config.input_hidden_scale,
            "hidden_local_scale" => config.hidden_local_scale,
            "hidden_output_scale" => config.hidden_output_scale,
            "best_epoch" => trained.best.epoch,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "final_mse" => final === nothing ? missing : final["mse"],
            "final_accuracy" => final === nothing ? missing : final["accuracy"],
            "outdir" => outdir,
        ))
        push!(results, (; config, trained, outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), summary)
    write_twoclass_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved two-class edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write a result table for the two-class edge search."""
function write_twoclass_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Edge Signal XOR With Two-Class Output Edge")
        println(io)
        println(io, "The input remains two raw spins connected only to the hidden left edge. The output is an 8x2 layer. Classification uses `mean(true column) - mean(false column)`.")
        println(io)
        println(io, "| trial | NN | beta | T fraction | input scale | hidden scale | output scale | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            println(io, "| `$(row["name"])` | $(row["hidden_nn"]) | $(row["beta"]) | $(row["temp_fraction"]) | $(row["input_hidden_scale"]) | $(row["hidden_local_scale"]) | $(row["hidden_output_scale"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_twoclass_edge_search()
end
