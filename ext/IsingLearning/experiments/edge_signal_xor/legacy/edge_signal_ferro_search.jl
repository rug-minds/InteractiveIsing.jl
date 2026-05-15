include("edge_signal_xor.jl")

const FERRO_HIDDEN_SCALE_REF = Ref{FT}(0.04)
const FERRO_HIDDEN_NN_REF = Ref{Int}(1)
const FERRO_HIDDEN_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))

"""Positive local hidden-layer weight with small multiplicative randomness."""
function ferro_hidden_local_weight(; dr)
    if dr <= FERRO_HIDDEN_NN_REF[]
        return FERRO_HIDDEN_SCALE_REF[] * (one(FT) + FT(0.1) * randn(FERRO_HIDDEN_RNG_REF[], FT))
    end
    return zero(FT)
end

"""Create a symmetric positive local coupling generator for the hidden layer."""
function ferro_hidden_local_generator(scale::FT, nn::Integer, rng)
    FERRO_HIDDEN_SCALE_REF[] = scale
    FERRO_HIDDEN_NN_REF[] = Int(nn)
    FERRO_HIDDEN_RNG_REF[] = rng
    if nn == 1
        return @WG ferro_hidden_local_weight NN = 1 symmetric = true
    elseif nn == 2
        return @WG ferro_hidden_local_weight NN = 2 symmetric = true
    elseif nn == 3
        return @WG ferro_hidden_local_weight NN = 3 symmetric = true
    elseif nn == 5
        return @WG ferro_hidden_local_weight NN = 5 symmetric = true
    else
        error("hidden_nn=$(nn) is not listed for this ferromagnetic hidden generator")
    end
end

"""
    ferro_edge_signal_graph(config)

Build the same edge-input architecture as `edge_signal_graph`, but initialize
hidden-local couplings as positive local couplings instead of signed random
couplings. Input and output edge couplings remain signed random.
"""
function ferro_edge_signal_graph(config::EdgeSignalXORConfig)
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
        ferro_hidden_local_generator(config.hidden_local_scale, config.hidden_nn, rng_hidden),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    layer_output = II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))

    input_hidden = input_to_left_edge_generator(config.input_hidden_scale, rng_input_hidden)
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

"""Wrap a ferromagnetic-hidden edge graph in the learning layer interface."""
function ferro_edge_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> ferro_edge_signal_graph(config);
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

"""Initialize a trainer for the ferromagnetic-hidden edge experiment."""
function ferro_edge_trainer(config::EdgeSignalXORConfig)
    graph = ferro_edge_signal_graph(config)
    layer = ferro_edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Train one ferromagnetic-hidden edge graph."""
function train_ferro_edge_xor(config::EdgeSignalXORConfig; outdir)
    trainer = ferro_edge_trainer(config)
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

"""Return ferromagnetic-medium edge trials for the 8x8 graph."""
function ferro_edge_trials()
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
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 1,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.22), hidden_local_scale = FT(0.035), hidden_output_scale = FT(0.22),
            bias_scale = FT(0.01), weight_seed = 401, bias_seed = 421, base_seed = 940_401,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.020), hidden_output_scale = FT(0.25),
            bias_scale = FT(0.01), weight_seed = 402, bias_seed = 422, base_seed = 940_402,
            skip_response = true, base...),
        EdgeSignalXORConfig(; hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.30), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.30),
            bias_scale = FT(0.01), weight_seed = 403, bias_seed = 423, base_seed = 940_403,
            skip_response = true, base...),
    ]
end

"""Run the ferromagnetic-hidden 8x8 edge search."""
function run_ferro_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_ferro_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for config in ferro_edge_trials()
        name = "NN$(config.hidden_nn)_beta$(replace(string(config.β), "." => "p"))_T$(replace(string(config.temp_fraction), "." => "p"))"
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_ferro_edge_xor(config; outdir)
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
    write_ferro_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved ferromagnetic-hidden edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write the ferromagnetic-hidden edge search result table."""
function write_ferro_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Edge Signal XOR With Positive Hidden Local Couplings")
        println(io)
        println(io, "Hidden-local couplings start positive so the 8x8 hidden layer is a signal-carrying medium instead of a signed random local network. Edge input/output couplings are still signed random and trainable.")
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
    run_ferro_edge_search()
end
