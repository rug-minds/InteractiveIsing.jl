using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using SparseArrays
using Statistics
using Dates
using CairoMakie

const FT = Float64
const II = IsingLearning.InteractiveIsing

const CASES = ((false, false), (false, true), (true, false), (true, true))
const DETECTOR_SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
const OUTPUT_SIGNS = FT[-1, 1, 1, -1]

"""
    Analytic241Config(; kwargs...)

Configuration for a focused `2 -> 4 -> 1` XOR experiment. The file uses the
standard `Forward_and_Nudged` worker path, so it avoids the stale active-index
issue in the earlier hand-rolled split-context experiment.
"""
Base.@kwdef struct Analytic241Config
    epochs::Int = parse(Int, get(ENV, "ISING_241_EPOCHS", "3000"))
    log_every::Int = parse(Int, get(ENV, "ISING_241_LOG_EVERY", "100"))
    minit::Int = parse(Int, get(ENV, "ISING_241_MINIT", "8"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_241_EVAL_REPEATS", "32"))
    free_relaxation::Int = parse(Int, get(ENV, "ISING_241_FREE", "300"))
    nudged_relaxation::Int = parse(Int, get(ENV, "ISING_241_NUDGED", "300"))
    β::FT = parse(FT, get(ENV, "ISING_241_BETA", "1.0"))
    target_scale::FT = parse(FT, get(ENV, "ISING_241_TARGET_SCALE", "1.0"))
    lr::FT = parse(FT, get(ENV, "ISING_241_LR", "0.002"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_241_WEIGHT_DECAY", "1e-4"))
    temp::FT = parse(FT, get(ENV, "ISING_241_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_241_STEPSIZE", "0.2"))
    input_strength::FT = parse(FT, get(ENV, "ISING_241_INPUT_STRENGTH", "2.0"))
    readout_strength::FT = parse(FT, get(ENV, "ISING_241_READOUT_STRENGTH", "0.75"))
    random_weight_scale::FT = parse(FT, get(ENV, "ISING_241_RANDOM_WEIGHT_SCALE", "0.05"))
    random_bias_scale::FT = parse(FT, get(ENV, "ISING_241_RANDOM_BIAS_SCALE", "0.02"))
    weight_seed::Int = parse(Int, get(ENV, "ISING_241_WEIGHT_SEED", "31"))
    bias_seed::Int = parse(Int, get(ENV, "ISING_241_BIAS_SEED", "37"))
    base_seed::Int = parse(Int, get(ENV, "ISING_241_BASE_SEED", "91000"))
    init::Symbol = Symbol(get(ENV, "ISING_241_INIT", "random"))
end

"""Return the physical two-bit XOR dataset with one scalar bipolar target."""
function xor_dataset_241()
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 1, 4)
    for (col, (a, b)) in enumerate(CASES)
        x[:, col] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        y[1, col] = xor(a, b) ? one(FT) : -one(FT)
    end
    return x, y
end

"""Scale scalar XOR targets for curriculum-style margin tests."""
function scaled_targets(y, config::Analytic241Config)
    out = copy(y)
    out .*= config.target_scale
    return out
end

"""
    graph_241(config)

Build a continuous bounded `2 -> 4 -> 1` graph with all-to-all adjacent-layer
topology, trainable `Bilinear`/`MagField`, and masked direct output clamping.
"""
function graph_241(config::Analytic241Config)
    layers = (
        II.Layer(2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 1, 0)),
        II.Layer(4, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 2, 0)),
        II.Layer(1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 3, 0)),
    )
    rng_w = Random.MersenneTwister(config.weight_seed)
    wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> config.random_weight_scale * randn(rng_w, FT))
    rng_b = Random.MersenneTwister(config.bias_seed)
    b = g -> config.random_bias_scale .* randn(rng_b, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = b) +
        II.Clamping(β = II.UniformArray(zero(FT)), y = target, mask = mask)
    graph = II.IsingGraph(
        layers[1],
        deepcopy(wg),
        layers[2],
        deepcopy(wg),
        layers[3],
        hamiltonian;
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""
    apply_corner_detector_solution!(graph, config)

Write an analytic XOR solution into the graph:

`h_ab = sign(a*x1 + b*x2 - 1)` for four hidden corner detectors, then
`y = sign([-1,+1,+1,-1]'h)`.
"""
function apply_corner_detector_solution!(graph, config::Analytic241Config)
    fill!(SparseArrays.getnzval(II.adj(graph)), zero(FT))
    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    fill!(b, zero(FT))

    input_idxs = collect(II.layerrange(graph[1]))
    hidden_idxs = collect(II.layerrange(graph[2]))
    output_idx = only(II.layerrange(graph[3]))

    for (hidden_pos, hidden_idx) in enumerate(hidden_idxs)
        sx, sy = DETECTOR_SIGNS[hidden_pos]
        II.adj(graph)[hidden_idx, input_idxs[1]] = config.input_strength * sx
        II.adj(graph)[hidden_idx, input_idxs[2]] = config.input_strength * sy
        b[hidden_idx] = -config.input_strength
        II.adj(graph)[output_idx, hidden_idx] = config.readout_strength * OUTPUT_SIGNS[hidden_pos]
    end
    return graph
end

"""Return a `LayeredIsingGraphLayer` using the standard process learning path."""
function layer_241(graph, config::Analytic241Config)
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    return LayeredIsingGraphLayer(
        () -> graph_241(config);
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

"""Build a trainer and optionally overwrite the initial parameters analytically."""
function trainer_241(config::Analytic241Config)
    graph = graph_241(config)
    config.init === :analytic && apply_corner_detector_solution!(graph, config)
    layer = layer_241(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    trainer = init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
    return trainer
end

"""Repeat the four XOR samples `minit` times to get averaged stochastic gradients."""
function repeated_batch(x, y, minit::Integer)
    xbatch = Matrix{FT}(undef, size(x, 1), size(x, 2) * minit)
    ybatch = Matrix{FT}(undef, size(y, 1), size(y, 2) * minit)
    col = 1
    for _ in 1:minit, sample_idx in axes(x, 2)
        xbatch[:, col] .= view(x, :, sample_idx)
        ybatch[:, col] .= view(y, :, sample_idx)
        col += 1
    end
    return xbatch, ybatch
end

"""Run one free validation relaxation and return the scalar output."""
function scalar_output!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    II.Processes.isdone(worker) && close(worker)
    Random.seed!(seed)
    hasproperty(worker.context.dynamics, :rng) && Random.seed!(worker.context.dynamics.rng, seed)
    IsingLearning._write_input!(worker, x)
    II.Processes.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    return only(copy(IsingLearning._validation_output(trainer)))
end

"""Evaluate mean scalar output statistics over repeated random starts."""
function evaluate_241!(trainer, x, y, config::Analytic241Config; seed_offset::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = scalar_output!(
                trainer,
                view(x, :, sample_idx);
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
            )
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

"""Write logged scalar metrics as a simple CSV file."""
function write_csv_241(path, rows)
    mkpath(dirname(path))
    headers = sort!(collect(keys(first(rows))))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((row[h] for h in headers), ","))
        end
    end
    return path
end

"""Save compact MSE/accuracy/margin plots for the run."""
function plot_241(path, rows)
    fig = Figure(size = (950, 700))
    ax1 = Axis(fig[1, 1], title = "2->4->1 scalar XOR MSE", xlabel = "epoch", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy")
    ax3 = Axis(fig[2, 1], title = "margin", xlabel = "epoch", ylabel = "min |mean output|")
    ax4 = Axis(fig[2, 2], title = "gradient norm", xlabel = "epoch", ylabel = "||grad||")
    epochs = [row["epoch"] for row in rows]
    lines!(ax1, epochs, [row["mse"] for row in rows])
    lines!(ax2, epochs, [row["accuracy"] for row in rows])
    lines!(ax3, epochs, [row["margin"] for row in rows])
    lines!(ax4, epochs, [row["grad_norm"] for row in rows])
    save(path, fig)
    return path
end

"""Append one metrics row to the run log."""
function push_row!(rows, epoch, metrics, grad_norm)
    row = Dict{String,Any}(
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.acc,
        "margin" => metrics.margin,
        "grad_norm" => grad_norm,
    )
    for i in eachindex(metrics.means)
        row["mean_$i"] = metrics.means[i]
        row["std_$i"] = metrics.stds[i]
    end
    push!(rows, row)
    return rows
end

"""Train/evaluate the `2 -> 4 -> 1` scalar XOR experiment."""
function main()
    config = Analytic241Config()
    outdir = get(
        ENV,
        "ISING_241_DIR",
        joinpath(@__DIR__, "runs", "analytic_2_4_1_" * string(config.init) * "_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)

    trainer = trainer_241(config)
    x, yraw = xor_dataset_241()
    y = scaled_targets(yraw, config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_241!(trainer, x, y, config; seed_offset = config.base_seed + 20_000_000)
    push_row!(rows, 0, metrics, zero(FT))
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        config.weight_decay > 0 && (trainer.params.w .*= (one(FT) - config.lr * config.weight_decay))
        IsingLearning._broadcast_params!(trainer)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_241!(trainer, x, y, config; seed_offset = config.base_seed + 20_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_row!(rows, epoch, metrics, grad_norm)
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad_norm, digits = 4), " means=", round.(metrics.means, digits = 3))
        end
    end

    csv_path = write_csv_241(joinpath(outdir, "metrics.csv"), rows)
    png_path = plot_241(joinpath(outdir, "progress.png"), rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Analytic 2->4->1 Scalar XOR")
        println(io)
        println(io, "Initialization: `$(config.init)`")
        println(io, "The analytic construction uses hidden corner detectors `sign(a*x1 + b*x2 - 1)` and output signs `[-,+,+,-]`.")
        println(io)
        println(io, "- `T = $(config.temp)`")
        println(io, "- `stepsize = $(config.stepsize)`")
        println(io, "- `β = $(config.β)`")
        println(io, "- target scale = `$(config.target_scale)`")
        println(io, "- free/nudged = `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- `Minit = $(config.minit)`, eval repeats `$(config.eval_repeats)`")
        println(io)
        println(io, "CSV: `metrics.csv`")
        println(io, "Plot: `progress.png`")
    end
    close_trainer!(trainer)
    println("Saved run: ", outdir)
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
