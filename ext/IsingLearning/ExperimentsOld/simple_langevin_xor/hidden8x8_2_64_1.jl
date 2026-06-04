using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using SparseArrays
using Statistics
using Dates
using CairoMakie
import IsingLearning.InteractiveIsing: WeightGenerator, @WG

const FT = Float64
const II = IsingLearning.InteractiveIsing

const CASES = ((false, false), (false, true), (true, false), (true, true))
const HIDDEN_LOCAL_SCALE_REF = Ref{FT}(0.04)
const HIDDEN_LOCAL_NN_REF = Ref{Int}(1)
const HIDDEN_LOCAL_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))

"""Named local-coupling generator used so `WeightGenerator` can inspect kwargs."""
function hidden_local_weight(; dr)
    return dr <= HIDDEN_LOCAL_NN_REF[] ? HIDDEN_LOCAL_SCALE_REF[] * randn(HIDDEN_LOCAL_RNG_REF[], FT) : zero(FT)
end

"""
    Hidden8x8Config(; kwargs...)

Configuration for the larger scalar XOR architecture:

```text
2 input spins -> all-to-all -> 8x8 hidden layer with local couplings -> all-to-one -> 1 output spin
```

The file intentionally uses the standard IsingLearning trainer path and no
local potential. The only hidden-layer structure is the learned/local bilinear
coupling topology.
"""
Base.@kwdef struct Hidden8x8Config
    epochs::Int = parse(Int, get(ENV, "ISING_8X8_EPOCHS", "2500"))
    log_every::Int = parse(Int, get(ENV, "ISING_8X8_LOG_EVERY", "500"))
    minit::Int = parse(Int, get(ENV, "ISING_8X8_MINIT", "8"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_8X8_EVAL_REPEATS", "16"))
    free_relaxation::Int = parse(Int, get(ENV, "ISING_8X8_FREE", "600"))
    nudged_relaxation::Int = parse(Int, get(ENV, "ISING_8X8_NUDGED", "600"))
    β::FT = parse(FT, get(ENV, "ISING_8X8_BETA", "2.0"))
    lr::FT = parse(FT, get(ENV, "ISING_8X8_LR", "0.002"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_8X8_WEIGHT_DECAY", "0"))
    temp::FT = parse(FT, get(ENV, "ISING_8X8_TEMP", "0.005"))
    stepsize::FT = parse(FT, get(ENV, "ISING_8X8_STEPSIZE", "0.4"))
    input_hidden_scale::FT = parse(FT, get(ENV, "ISING_8X8_INPUT_HIDDEN_SCALE", "0.06"))
    hidden_nn::Int = parse(Int, get(ENV, "ISING_8X8_HIDDEN_NN", "1"))
    hidden_local_scale::FT = parse(FT, get(ENV, "ISING_8X8_HIDDEN_LOCAL_SCALE", "0.04"))
    hidden_output_scale::FT = parse(FT, get(ENV, "ISING_8X8_HIDDEN_OUTPUT_SCALE", "0.06"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_8X8_BIAS_SCALE", "0.02"))
    target_scale::FT = parse(FT, get(ENV, "ISING_8X8_TARGET_SCALE", "1.0"))
    weight_seed::Int = parse(Int, get(ENV, "ISING_8X8_WEIGHT_SEED", "31"))
    bias_seed::Int = parse(Int, get(ENV, "ISING_8X8_BIAS_SEED", "37"))
    base_seed::Int = parse(Int, get(ENV, "ISING_8X8_BASE_SEED", "121000"))
end

"""Return physical bipolar XOR inputs and one scalar bipolar target."""
function xor_dataset_8x8(config::Hidden8x8Config)
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 1, 4)
    for (col, (a, b)) in enumerate(CASES)
        x[:, col] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        y[1, col] = config.target_scale * (xor(a, b) ? one(FT) : -one(FT))
    end
    return x, y
end

"""Create an all-to-all random generator with one RNG and a chosen scale."""
function all_to_all_rng(scale::FT, rng)
    return II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> scale * randn(rng, FT))
end

"""Create the hidden-layer random local coupling generator up to distance `nn`."""
function hidden_local_rng(scale::FT, nn::Integer, rng)
    HIDDEN_LOCAL_SCALE_REF[] = scale
    HIDDEN_LOCAL_NN_REF[] = Int(nn)
    HIDDEN_LOCAL_RNG_REF[] = rng
    return II.WeightGenerator(hidden_local_weight, Int(nn), rng; symmetric = true)
end

"""
    graph_8x8(config)

Build the `2 -> 8x8 -> 1` graph. Input-hidden and hidden-output are all-to-all;
the hidden layer additionally has local internal couplings up to `hidden_nn`.
"""
function graph_8x8(config::Hidden8x8Config)
    rng_input_hidden = Random.MersenneTwister(config.weight_seed)
    rng_hidden = Random.MersenneTwister(config.weight_seed + 1)
    rng_hidden_output = Random.MersenneTwister(config.weight_seed + 2)
    rng_bias = Random.MersenneTwister(config.bias_seed)

    input_hidden_wg = all_to_all_rng(config.input_hidden_scale, rng_input_hidden)
    hidden_local_wg = hidden_local_rng(config.hidden_local_scale, config.hidden_nn, rng_hidden)
    hidden_output_wg = all_to_all_rng(config.hidden_output_scale, rng_hidden_output)

    layers = (
        II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 1, 0)),
        II.Layer(8, 8, II.StateSet(-one(FT), one(FT)), hidden_local_wg, II.Continuous(), II.Coords(0, 2, 0)),
        II.Layer(1, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 3, 0)),
    )

    b = g -> config.bias_scale .* randn(rng_bias, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = b) +
        II.Clamping(β = II.UniformArray(zero(FT)), y = target, mask = mask)

    graph = II.IsingGraph(
        layers[1],
        input_hidden_wg,
        layers[2],
        hidden_output_wg,
        layers[3],
        hamiltonian;
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Create the learning-layer wrapper with tuned unadjusted LocalLangevin."""
function layer_8x8(graph, config::Hidden8x8Config)
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    return LayeredIsingGraphLayer(
        () -> graph_8x8(config);
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

"""Initialize the standard threaded trainer for this experiment."""
function trainer_8x8(config::Hidden8x8Config)
    graph = graph_8x8(config)
    layer = layer_8x8(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Repeat the four XOR samples for gradient averaging."""
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

"""Run one validation relaxation and return the scalar output."""
function scalar_output_8x8!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    II.StatefulAlgorithms.isdone(worker) && close(worker)
    Random.seed!(seed)
    hasproperty(StatefulAlgorithms.context(worker).dynamics, :rng) && Random.seed!(StatefulAlgorithms.context(worker).dynamics.rng, seed)
    IsingLearning._write_input!(worker, x)
    II.StatefulAlgorithms.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    return only(copy(IsingLearning._validation_output(trainer)))
end

"""Evaluate repeated-start scalar output statistics for all four XOR cases."""
function evaluate_8x8!(trainer, x, y, config::Hidden8x8Config; seed_offset::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = scalar_output_8x8!(
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

"""Write logged metrics to CSV."""
function write_csv_8x8(path, rows)
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

"""Plot MSE, accuracy, margin, and gradient norm for the run."""
function plot_8x8(path, rows)
    fig = Figure(size = (1000, 760))
    ax1 = Axis(fig[1, 1], title = "2 -> 8x8 -> 1 MSE", xlabel = "epoch", ylabel = "MSE")
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

"""Append one metrics row."""
function push_row_8x8!(rows, epoch, metrics, grad_norm)
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

"""
    strip_weight_generators_8x8!(graph)

Remove construction-time random generator objects before writing the trained
graph. The trained adjacency is already materialized, so the generators are not
needed to reload the learned model.
"""
function strip_weight_generators_8x8!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

"""Run the larger hidden-layer scalar XOR experiment."""
function main()
    config = Hidden8x8Config()
    outdir = get(
        ENV,
        "ISING_8X8_DIR",
        joinpath(@__DIR__, "runs", "hidden8x8_2_64_1_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)

    trainer = trainer_8x8(config)
    x, y = xor_dataset_8x8(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_row_8x8!(rows, 0, metrics, zero(FT))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if config.weight_decay > 0
            trainer.params.w .*= (one(FT) - config.lr * config.weight_decay)
            IsingLearning._broadcast_params!(trainer)
        end
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_row_8x8!(rows, epoch, metrics, grad_norm)
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad_norm, digits = 4), " means=", round.(metrics.means, digits = 3))
        end
    end

    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    csv_path = write_csv_8x8(joinpath(outdir, "metrics.csv"), rows)
    png_path = plot_8x8(joinpath(outdir, "progress.png"), rows)
    graph_path = II.save_isinggraph(
        joinpath(outdir, "hidden8x8_2_64_1_best_graph.jld2"),
        strip_weight_generators_8x8!(deepcopy(trainer.prototype_graph)),
    )
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# 2 -> 8x8 -> 1 Scalar XOR")
        println(io)
        println(io, "Architecture: all-to-all input-hidden, hidden NN=1 local couplings, all-to-one hidden-output.")
        println(io)
        println(io, "- epochs/log_every: `$(config.epochs)` / `$(config.log_every)`")
        println(io, "- free/nudged: `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- Minit/eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- β/lr/T/stepsize: `$(config.β)` / `$(config.lr)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- hidden local NN: `$(config.hidden_nn)`")
        println(io, "- scales input-hidden/local/hidden-output/bias: `$(config.input_hidden_scale)` / `$(config.hidden_local_scale)` / `$(config.hidden_output_scale)` / `$(config.bias_scale)`")
        println(io)
        println(io, "Best logged: epoch `$(best.epoch)`, MSE `$(round(best.mse, digits = 6))`, accuracy `$(best.acc)`.")
        println(io)
        println(io, "CSV: `metrics.csv`")
        println(io, "Plot: `progress.png`")
        println(io, "Best graph: `hidden8x8_2_64_1_best_graph.jld2`")
    end
    close_trainer!(trainer)
    println("Saved run: ", outdir)
    return (; outdir, best, csv_path, png_path, graph_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
