include("edge_signal_xor.jl")

const SPLIT_INPUT_SCALE_REF = Ref{FT}(0.25)
const SPLIT_INPUT_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))

"""
    split_input_left_edge_weight(; c1, c2)

Weight rule for prestructured edge input.

Input spin 1 connects only to the upper half of the hidden left edge. Input
spin 2 connects only to the lower half. This keeps the input dimension at two,
but prevents both input spins from driving the exact same boundary pixels.
"""
function split_input_left_edge_weight(; c1, c2)
    left_hidden_edge(c2) || return zero(FT)
    hidden_height = HIDDEN_HEIGHT_REF[]
    top_half = c2[1] <= hidden_height ÷ 2
    input_row = c1[1]
    if (input_row == 1 && top_half) || (input_row == 2 && !top_half)
        return SPLIT_INPUT_SCALE_REF[] * randn(SPLIT_INPUT_RNG_REF[], FT)
    end
    return zero(FT)
end

"""Create the split input-to-left-edge generator for the two input spins."""
function split_input_to_left_edge_generator(scale::FT, rng)
    SPLIT_INPUT_SCALE_REF[] = scale
    SPLIT_INPUT_RNG_REF[] = rng
    return @WG split_input_left_edge_weight NN = :all symmetric = true
end

"""
    split_input_edge_graph(config)

Build `2 input spins -> split left edge of 8x8 hidden -> right edge -> 1 output`.

Only the input-to-hidden generator differs from `edge_signal_graph`: each input
spin has its own half of the first hidden edge. The output remains a single
scalar spin connected to the last hidden edge.
"""
function split_input_edge_graph(config::EdgeSignalXORConfig)
    HIDDEN_HEIGHT_REF[] = config.hidden_height
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
    layer_output = II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = split_input_to_left_edge_generator(config.input_hidden_scale, rng_input_hidden)
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

"""Wrap the split-input edge graph in the standard IsingLearning layer interface."""
function split_input_edge_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> split_input_edge_graph(config);
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

"""Initialize the trainer for the split-input edge experiment."""
function split_input_edge_trainer(config::EdgeSignalXORConfig)
    graph = split_input_edge_graph(config)
    layer = split_input_edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Train one split-input edge graph and save best/initial graphs."""
function train_split_input_edge_xor(config::EdgeSignalXORConfig; outdir)
    trainer = split_input_edge_trainer(config)
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

"""Return a focused grid for split input halves on the 8x8 hidden edge."""
function split_input_edge_trials()
    base = (;
        epochs = 6000,
        log_every = 600,
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
            input_hidden_scale = FT(0.35), hidden_local_scale = FT(0.014), hidden_output_scale = FT(0.30),
            bias_scale = FT(0.01), weight_seed = 701, bias_seed = 721, base_seed = 1_070_701,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.030),
            input_hidden_scale = FT(0.40), hidden_local_scale = FT(0.009), hidden_output_scale = FT(0.35),
            bias_scale = FT(0.01), weight_seed = 702, bias_seed = 722, base_seed = 1_070_702,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 5,
            β = FT(0.20), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.01), weight_seed = 703, bias_seed = 723, base_seed = 1_070_703,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            β = FT(0.35), lr = FT(0.0005), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.40), hidden_local_scale = FT(0.009), hidden_output_scale = FT(0.35),
            bias_scale = FT(0.01), weight_seed = 704, bias_seed = 724, base_seed = 1_070_704,
            skip_response = true, base...),
    ]
end

"""Run the split-input edge search and write a summary CSV and markdown file."""
function run_split_input_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for config in split_input_edge_trials()
        name = "NN$(config.hidden_nn)_beta$(replace(string(config.β), "." => "p"))_T$(replace(string(config.temp_fraction), "." => "p"))"
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_edge_xor(config; outdir)
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
        write_run_readme(joinpath(outdir, "README.md"), config, trained, nothing)
        push!(results, (; config, trained, outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), summary)
    write_split_input_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved split-input edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write the split-input search result table."""
function write_split_input_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Edge Signal XOR With Split Input Halves")
        println(io)
        println(io, "Architecture remains `2 input spins -> left edge of 8x8 hidden -> right edge -> 1 output spin`.")
        println(io, "Input spin 1 connects only to the upper half of the first hidden edge. Input spin 2 connects only to the lower half.")
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
    run_split_input_edge_search()
end
