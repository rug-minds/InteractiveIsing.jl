include("edge_signal_xor.jl")

const OUTPUT_EDGE_SCALE_REF = Ref{FT}(0.12)
const OUTPUT_EDGE_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))

"""Signed random connection from the hidden right edge to an output-edge layer."""
function hidden_to_output_edge_weight(; c1)
    return right_hidden_edge(c1) ? OUTPUT_EDGE_SCALE_REF[] * randn(OUTPUT_EDGE_RNG_REF[], FT) : zero(FT)
end

"""Create hidden-right-edge to output-edge signed random couplings."""
function hidden_to_output_edge_generator(scale::FT, rng)
    OUTPUT_EDGE_SCALE_REF[] = scale
    OUTPUT_EDGE_RNG_REF[] = rng
    return @WG hidden_to_output_edge_weight NN = :all symmetric = true
end

"""
    output_edge_graph(config)

Build `2 input spins -> 8x8 hidden -> 8 output spins`. The input spins connect
only to the hidden left edge. Output spins connect only from the hidden right
edge. XOR class is represented by all output spins targeting the same sign.
"""
function output_edge_graph(config::EdgeSignalXORConfig)
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
    layer_output = II.Layer(8, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = input_to_left_edge_generator(config.input_hidden_scale, rng_input_hidden)
    hidden_output = hidden_to_output_edge_generator(config.hidden_output_scale, rng_hidden_output)

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

"""Return two-spin XOR inputs and an eight-spin same-sign output target."""
function output_edge_dataset(config::EdgeSignalXORConfig)
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 8, 4)
    for (idx, (a, b)) in enumerate(XOR_CASES)
        x[:, idx] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        class_sign = config.target_scale * (xor(a, b) ? one(FT) : -one(FT))
        y[:, idx] .= class_sign
    end
    return x, y
end

"""Wrap the output-edge graph in the standard IsingLearning layer interface."""
function output_edge_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> output_edge_graph(config);
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

"""Initialize a trainer for the output-edge readout experiment."""
function output_edge_trainer(config::EdgeSignalXORConfig)
    graph = output_edge_graph(config)
    layer = output_edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Run validation for one input and return the mean of the output edge."""
function output_edge_scalar!(trainer, x, config::EdgeSignalXORConfig; seed::Integer)
    graph = trainer.validation_graph
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)
    run_dynamics_steps!(graph, edge_sampler(config), config.validation_relaxation; seed)
    return mean(copy(II.state(graph[end])))
end

"""Evaluate output-edge class means across repeated starts."""
function evaluate_output_edge!(trainer, x, y, config::EdgeSignalXORConfig; seed_offset::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    targets = vec(mean(y; dims = 1))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = output_edge_scalar!(
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

"""Train one output-edge graph and save best/initial graphs."""
function train_output_edge_xor(config::EdgeSignalXORConfig; outdir)
    trainer = output_edge_trainer(config)
    x, y = output_edge_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_output_edge!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_learning_row!(rows, 0, metrics, zero(FT), II.temp(trainer.prototype_graph))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    initial_params = deepcopy(trainer.params)
    println("NN=", config.hidden_nn, " epoch=0 mse=", round(metrics.mse, digits = 6),
        " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_output_edge!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
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

"""Return focused 8x8 output-edge trials."""
function output_edge_trials()
    base = (;
        epochs = 4000,
        log_every = 400,
        minit = 2,
        eval_repeats = 12,
        free_relaxation = 1000,
        nudged_relaxation = 1000,
        validation_relaxation = 2200,
        stepsize = FT(0.8),
    )
    return [
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.22), hidden_local_scale = FT(0.015), hidden_output_scale = FT(0.22),
            bias_scale = FT(0.01), weight_seed = 501, bias_seed = 521, base_seed = 980_501,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.010), hidden_output_scale = FT(0.25),
            bias_scale = FT(0.01), weight_seed = 502, bias_seed = 522, base_seed = 980_502,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 2,
            β = FT(0.35), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.25),
            bias_scale = FT(0.01), weight_seed = 503, bias_seed = 523, base_seed = 980_503,
            skip_response = true, base...),
    ]
end

"""Run the output-edge readout search."""
function run_output_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_output_edge_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for config in output_edge_trials()
        name = "NN$(config.hidden_nn)_beta$(replace(string(config.β), "." => "p"))_T$(replace(string(config.temp_fraction), "." => "p"))"
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_output_edge_xor(config; outdir)
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
    write_output_edge_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved output-edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write a result table for the output-edge readout search."""
function write_output_edge_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Edge Signal XOR With Output Edge Readout")
        println(io)
        println(io, "This keeps two physical input spins on the left edge path, but uses eight output spins and classifies by their mean sign.")
        println(io)
        println(io, "| trial | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            println(io, "| `$(row["name"])` | $(row["hidden_nn"]) | $(row["beta"]) | $(row["temp_fraction"]) | $(row["input_hidden_scale"]) | $(row["hidden_local_scale"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_output_edge_search()
end
