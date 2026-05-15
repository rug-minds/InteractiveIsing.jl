using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using Optimisers
using Random
using SparseArrays
using LinearAlgebra
using Dates
using CairoMakie

const FT = Float64
const II = IsingLearning.InteractiveIsing
const Processes = II.Processes

# Multiplexed XOR pattern experiment.
#
# Inputs are four mutually orthogonal 4x4 global patterns. Outputs are two
# orthogonal 4x4 global patterns, so the output layer has to learn a distributed
# code instead of the two-spin one-hot code from xor_statistical_ep.jl.
const EPOCHS = parse(Int, get(ENV, "ISING_XOR_MUX_EPOCHS", "1000"))
const LOG_EVERY = parse(Int, get(ENV, "ISING_XOR_MUX_LOG_EVERY", "100"))
const MINIT = parse(Int, get(ENV, "ISING_XOR_MUX_MINIT", "4"))
const EVAL_REPEATS = parse(Int, get(ENV, "ISING_XOR_MUX_EVAL_REPEATS", "16"))
const SIDE = parse(Int, get(ENV, "ISING_XOR_MUX_SIDE", "4"))
const HIDDEN_SIDE = parse(Int, get(ENV, "ISING_XOR_MUX_HIDDEN_SIDE", "4"))
const INPUT_UNITS = SIDE * SIDE
const OUTPUT_UNITS = SIDE * SIDE
const HIDDEN_UNITS = HIDDEN_SIDE * HIDDEN_SIDE
const BETA = parse(FT, get(ENV, "ISING_XOR_MUX_BETA", "0.05"))
const LEARNING_RATE = parse(FT, get(ENV, "ISING_XOR_MUX_LR", "0.005"))
const WEIGHT_DECAY = parse(FT, get(ENV, "ISING_XOR_MUX_WEIGHT_DECAY", "1e-4"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_MUX_WEIGHT_SCALE", "0.03"))
const BIAS_SCALE = parse(FT, get(ENV, "ISING_XOR_MUX_BIAS_SCALE", "0.1"))
const TEMP = parse(FT, get(ENV, "ISING_XOR_MUX_TEMP", "0.001"))
const STEPSIZE = parse(FT, get(ENV, "ISING_XOR_MUX_STEPSIZE", "0.05"))
const BLOCK_SIZE = parse(Int, get(ENV, "ISING_XOR_MUX_BLOCK_SIZE", "8"))
const FREE_RELAXATION = parse(Int, get(ENV, "ISING_XOR_MUX_FREE_RELAXATION", "50"))
const NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_XOR_MUX_NUDGED_RELAXATION", "50"))
const WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_MUX_WEIGHT_SEED", "2"))
const BIAS_SEED = parse(Int, get(ENV, "ISING_XOR_MUX_BIAS_SEED", "11"))
const BASE_SEED = parse(Int, get(ENV, "ISING_XOR_MUX_BASE_SEED", "72000"))
const EVAL_SEED_OFFSET = parse(Int, get(ENV, "ISING_XOR_MUX_EVAL_SEED_OFFSET", "30000000"))
const OUTDIR = get(
    ENV,
    "ISING_XOR_MUX_DIR",
    joinpath(@__DIR__, "..", "runs", "xor_multiplexed_patterns_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
)

const CASES = ((false, false), (false, true), (true, false), (true, true))
const CASE_NAMES = ("ff", "ft", "tf", "tt")

case_index(a::Bool, b::Bool) = (a ? 2 : 0) + (b ? 1 : 0) + 1
xor_label(a::Bool, b::Bool) = xor(a, b)

function require_even_side()
    iseven(SIDE) || error("SIDE must be even for the default orthogonal multiplexed patterns")
    return nothing
end

function vertical_pattern()
    pattern = Matrix{FT}(undef, SIDE, SIDE)
    split = SIDE ÷ 2
    for j in 1:SIDE, i in 1:SIDE
        pattern[i, j] = i <= split ? -one(FT) : one(FT)
    end
    return vec(pattern)
end

function horizontal_pattern()
    pattern = Matrix{FT}(undef, SIDE, SIDE)
    split = SIDE ÷ 2
    for j in 1:SIDE, i in 1:SIDE
        pattern[i, j] = j <= split ? -one(FT) : one(FT)
    end
    return vec(pattern)
end

const P_VERTICAL = vertical_pattern()
const P_HORIZONTAL = horizontal_pattern()
const P_CHECKERBOARD = P_VERTICAL .* P_HORIZONTAL
const INPUT_PATTERNS = (
    fill(one(FT), INPUT_UNITS),
    P_VERTICAL,
    P_HORIZONTAL,
    P_CHECKERBOARD,
)

function check_pairwise_orthogonal(name, patterns)
    for i in eachindex(patterns), j in eachindex(patterns)
        i == j && continue
        dotij = dot(patterns[i], patterns[j])
        abs(dotij) <= FT(100) * eps(FT) * length(patterns[i]) ||
            error("$name patterns $i and $j are not orthogonal; dot = $dotij")
    end
    return nothing
end

function validate_patterns!()
    require_even_side()
    check_pairwise_orthogonal("input", INPUT_PATTERNS)
    check_pairwise_orthogonal("output", (P_VERTICAL, P_HORIZONTAL))
    return nothing
end

function xor_input(a::Bool, b::Bool)
    return INPUT_PATTERNS[case_index(a, b)]
end

function xor_target(a::Bool, b::Bool)
    return xor_label(a, b) ? P_HORIZONTAL : P_VERTICAL
end

function output_score(output, target)
    return -sum(abs2, output .- target) / FT(length(target))
end

function output_margin(output)
    return output_score(output, P_HORIZONTAL) - output_score(output, P_VERTICAL)
end

function predict_label(output)
    return output_margin(output) > zero(FT)
end

function xor_dataset()
    x = Matrix{FT}(undef, INPUT_UNITS, length(CASES))
    y = Matrix{FT}(undef, OUTPUT_UNITS, length(CASES))
    for (col, case) in enumerate(CASES)
        x[:, col] .= xor_input(case...)
        y[:, col] .= xor_target(case...)
    end
    return x, y
end

function signed_weight_generator()
    rng = Random.MersenneTwister(WEIGHT_SEED)
    return II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, FT))
end

function bias_generator()
    rng = Random.MersenneTwister(BIAS_SEED)
    return g -> BIAS_SCALE .* randn(rng, FT, II.statelen(g))
end

function xor_graph()
    layers = [
        II.Layer(SIDE, SIDE, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 1, 0); periodic = false),
        II.Layer(HIDDEN_SIDE, HIDDEN_SIDE, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 2, 0); periodic = false),
        II.Layer(SIDE, SIDE, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 3, 0); periodic = false),
    ]

    wg = signed_weight_generator()
    layer_args = Any[layers[1]]
    for idx in 2:length(layers)
        push!(layer_args, deepcopy(wg), layers[idx])
    end

    clamping_target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    clamping_beta = II.UniformArray(zero(FT))
    hamiltonian = II.Bilinear() + II.MagField(b = bias_generator()) +
        II.Clamping(β = clamping_beta, y = clamping_target)

    graph = II.IsingGraph(layer_args..., hamiltonian; precision = FT, index_set = g -> II.ToggledIndexSet(g))
    II.temp!(graph, TEMP)
    return graph
end

function xor_layer(graph)
    base_dynamics = II.BlockLangevin(stepsize = STEPSIZE, adjusted = false, block_size = BLOCK_SIZE, group_steps = 1)
    free_dynamics = deepcopy(base_dynamics)
    return LayeredIsingGraphLayer(
        () -> xor_graph();
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = BETA,
        fullsweeps = 1,
        relaxation_steps = FREE_RELAXATION,
        free_relaxation_steps = FREE_RELAXATION,
        nudged_relaxation_steps = NUDGED_RELAXATION,
        dynamics_algorithm = free_dynamics,
        nudged_dynamics_algorithm = deepcopy(base_dynamics),
        validation_algorithm = deepcopy(base_dynamics),
    )
end

function set_trainer_temperature!(trainer, T::Real)
    II.temp!(trainer.prototype_graph, FT(T))
    II.temp!(trainer.validation_graph, FT(T))
    foreach(g -> II.temp!(g, FT(T)), trainer.worker_graphs)
    return trainer
end

function seed_worker!(worker, seed::Integer)
    Random.seed!(seed)
    hasproperty(Processes.context(worker).dynamics, :rng) && Random.seed!(Processes.context(worker).dynamics.rng, seed)
    hasproperty(Processes.context(worker), :nudged_dynamics) &&
        hasproperty(Processes.context(worker).nudged_dynamics, :rng) &&
        Random.seed!(Processes.context(worker).nudged_dynamics.rng, seed + 1)
    return worker
end

function run_training_trajectory!(worker, x, y; seed::Integer)
    Processes.isdone(worker) && close(worker)
    seed_worker!(worker, seed)
    IsingLearning._write_example!(worker, x, y)
    Processes.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)

    free_state = Processes.context(worker)._state.equilibrium_state
    plus_state = Processes.context(worker).plus_capture.captured
    minus_state = Processes.context(worker).minus_capture.captured
    response = (
        sqrt(sum(abs2, plus_state .- free_state) / FT(length(free_state))) +
        sqrt(sum(abs2, minus_state .- free_state) / FT(length(free_state)))
    ) / FT(2)
    return response
end

function add_weight_decay!(gradient, params)
    WEIGHT_DECAY > zero(FT) || return gradient
    gradient.w .+= WEIGHT_DECAY .* params.w
    return gradient
end

function train_epoch!(trainer, x, y, batch_gradient, epoch::Integer)
    IsingLearning._reset_batch_buffers!(trainer)
    worker = only(trainer.workers)

    total_response = zero(FT)
    ntraj = 0
    for sample_idx in axes(x, 2)
        for init_idx in 1:MINIT
            seed = BASE_SEED + 1_000_000 * epoch + 10_000 * sample_idx + init_idx
            response = run_training_trajectory!(
                worker,
                view(x, :, sample_idx),
                view(y, :, sample_idx);
                seed,
            )
            total_response += response
            ntraj += 1
        end
    end

    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, ntraj)
    add_weight_decay!(batch_gradient, trainer.params)
    all(isfinite, batch_gradient.w) || error("non-finite weight gradient")
    all(isfinite, batch_gradient.b) || error("non-finite bias gradient")

    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    all(isfinite, trainer.params.w) || error("non-finite weights")
    all(isfinite, trainer.params.b) || error("non-finite biases")
    return (; grad_norm, response_norm = total_response / FT(max(ntraj, 1)))
end

function run_validation_output!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    Processes.isdone(worker) && close(worker)
    seed_worker!(worker, seed)
    IsingLearning._write_input!(worker, x)
    Processes.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    output = copy(IsingLearning._validation_output(trainer))
    all(isfinite, output) || error("non-finite validation output")
    return output
end

function output_std(outputs, mean_output)
    length(outputs) <= 1 && return zero(FT)
    var = zero(FT)
    for output in outputs
        var += sum(abs2, output .- mean_output) / FT(length(mean_output))
    end
    return sqrt(var / FT(length(outputs) - 1))
end

function evaluate_xor!(trainer, x, y; seed_offset::Integer = 20_000_000)
    outputs = Vector{Vector{FT}}(undef, size(x, 2))
    stds = zeros(FT, size(x, 2))

    for sample_idx in axes(x, 2)
        samples = Vector{Vector{FT}}(undef, EVAL_REPEATS)
        mean_output = zeros(FT, OUTPUT_UNITS)
        for repeat_idx in 1:EVAL_REPEATS
            out = run_validation_output!(
                trainer,
                view(x, :, sample_idx);
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
            )
            samples[repeat_idx] = out
            mean_output .+= out
        end
        mean_output ./= FT(EVAL_REPEATS)
        outputs[sample_idx] = mean_output
        stds[sample_idx] = output_std(samples, mean_output)
    end

    mse = zero(FT)
    ncorrect = 0
    for sample_idx in axes(x, 2)
        target = view(y, :, sample_idx)
        mse += sum(abs2, outputs[sample_idx] .- target) / FT(OUTPUT_UNITS)
        ncorrect += predict_label(outputs[sample_idx]) == xor_label(CASES[sample_idx]...)
    end
    mse /= FT(size(x, 2))
    accuracy = FT(ncorrect) / FT(size(x, 2))
    return (; mse, accuracy, outputs, stds)
end

function metric_row(epoch, metrics, grad_metrics, trainer, initial_params)
    param_delta = sqrt(
        sum(abs2, trainer.params.w .- initial_params.w) +
        sum(abs2, trainer.params.b .- initial_params.b)
    )
    row = Dict{String, Any}(
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.accuracy,
        "grad_norm" => grad_metrics.grad_norm,
        "response_norm" => grad_metrics.response_norm,
        "param_delta" => param_delta,
    )
    for (idx, name) in enumerate(CASE_NAMES)
        row["$(name)_margin"] = output_margin(metrics.outputs[idx])
        row["$(name)_std"] = metrics.stds[idx]
    end
    return row
end

function write_csv(path, rows)
    isempty(rows) && return path
    keys_order = [
        "epoch", "mse", "accuracy", "grad_norm", "response_norm", "param_delta",
        "ff_margin", "ff_std",
        "ft_margin", "ft_std",
        "tf_margin", "tf_std",
        "tt_margin", "tt_std",
    ]
    open(path, "w") do io
        println(io, join(keys_order, ","))
        for row in rows
            println(io, join((row[key] for key in keys_order), ","))
        end
    end
    return path
end

function plot_metrics(path, rows)
    isempty(rows) && return path
    epochs = [row["epoch"] for row in rows]
    fig = Figure(size = (1100, 850))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", title = "Mean output MSE")
    lines!(ax_mse, epochs, [row["mse"] for row in rows], color = :dodgerblue, linewidth = 2)

    ax_acc = Axis(fig[1, 2], xlabel = "epoch", ylabel = "accuracy", title = "XOR accuracy", limits = (nothing, (-0.05, 1.05)))
    lines!(ax_acc, epochs, [row["accuracy"] for row in rows], color = :seagreen, linewidth = 2)

    ax_grad = Axis(fig[2, 1], xlabel = "epoch", ylabel = "norm", title = "Gradient and nudged response")
    lines!(ax_grad, epochs, [row["grad_norm"] for row in rows], color = :firebrick, linewidth = 2, label = "gradient")
    lines!(ax_grad, epochs, [row["response_norm"] for row in rows], color = :darkorange, linewidth = 2, label = "response")
    axislegend(ax_grad, position = :rt)

    ax_delta = Axis(fig[2, 2], xlabel = "epoch", ylabel = "||θ - θ0||", title = "Parameter movement")
    lines!(ax_delta, epochs, [row["param_delta"] for row in rows], color = :mediumpurple, linewidth = 2)

    ax_scores = Axis(fig[3, 1:2], xlabel = "epoch", ylabel = "margin", title = "Output pattern margin true - false")
    for (name, color) in zip(CASE_NAMES, Makie.wong_colors())
        scores = [row["$(name)_margin"] for row in rows]
        lines!(ax_scores, epochs, scores, color = color, linewidth = 2, label = name)
    end
    hlines!(ax_scores, [0], color = (:black, 0.35), linestyle = :dash)
    axislegend(ax_scores, position = :rt)

    save(path, fig)
    return path
end

function print_case_outputs(metrics)
    for (idx, case) in enumerate(CASES)
        println(
            "    case=$case target=$(xor_target(case...)) ",
            "margin=$(round(output_margin(metrics.outputs[idx]), digits = 5)) ",
            "mean_head=", round.(metrics.outputs[idx][1:min(6, end)]; digits = 4),
            " std=$(round(metrics.stds[idx], digits = 5))",
        )
    end
end

function main()
    MINIT > 0 || error("ISING_XOR_MUX_MINIT must be positive")
    EVAL_REPEATS > 0 || error("ISING_XOR_MUX_EVAL_REPEATS must be positive")
    validate_patterns!()
    mkpath(OUTDIR)

    Random.seed!(BASE_SEED)
    graph = xor_graph()
    layer = xor_layer(graph)
    x, y = xor_dataset()
    trainer = init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(LEARNING_RATE))
    set_trainer_temperature!(trainer, TEMP)

    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    rows = Dict{String, Any}[]
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    println(
        "Running multiplexed XOR pattern experiment: ",
        (epochs = EPOCHS, log_every = LOG_EVERY, minit = MINIT, eval_repeats = EVAL_REPEATS,
         input = (SIDE, SIDE), hidden = (HIDDEN_SIDE, HIDDEN_SIDE), output = (SIDE, SIDE),
         input_gram_offdiag = maximum(abs(dot(INPUT_PATTERNS[i], INPUT_PATTERNS[j])) for i in 1:4 for j in 1:4 if i != j),
         output_dot = dot(P_VERTICAL, P_HORIZONTAL),
         beta = BETA, lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY,
         temp = TEMP, stepsize = STEPSIZE, block_size = BLOCK_SIZE,
         free_relaxation = FREE_RELAXATION, nudged_relaxation = NUDGED_RELAXATION),
    )

    validation_seed_offset = BASE_SEED + EVAL_SEED_OFFSET
    before = evaluate_xor!(trainer, x, y; seed_offset = validation_seed_offset)
    push!(rows, metric_row(0, before, zero_grad, trainer, initial_params))
    println("epoch=0 mse=$(round(before.mse, digits=6)) acc=$(before.accuracy)")
    print_case_outputs(before)

    best_mse = before.mse
    best_params = deepcopy(trainer.params)

    for epoch in 1:EPOCHS
        grad_metrics = train_epoch!(trainer, x, y, batch_gradient, epoch)
        if epoch % LOG_EVERY == 0 || epoch == 1 || epoch == EPOCHS
            metrics = evaluate_xor!(trainer, x, y; seed_offset = validation_seed_offset)
            row = metric_row(epoch, metrics, grad_metrics, trainer, initial_params)
            push!(rows, row)
            if metrics.mse < best_mse
                best_mse = metrics.mse
                best_params = deepcopy(trainer.params)
            end
            println(
                "epoch=$epoch mse=$(round(metrics.mse, digits=6)) acc=$(metrics.accuracy) ",
                "grad=$(round(grad_metrics.grad_norm, digits=6)) ",
                "response=$(round(grad_metrics.response_norm, digits=6)) ",
                "delta=$(round(row["param_delta"], digits=6))",
            )
        end
    end

    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    restored = evaluate_xor!(trainer, x, y; seed_offset = validation_seed_offset)
    println("restored best mse=$(round(restored.mse, digits=6)) acc=$(restored.accuracy)")
    print_case_outputs(restored)

    csv_path = write_csv(joinpath(OUTDIR, "xor_multiplexed_patterns.csv"), rows)
    png_path = plot_metrics(joinpath(OUTDIR, "xor_multiplexed_patterns.png"), rows)
    graph_path = II.save_isinggraph(joinpath(OUTDIR, "xor_multiplexed_patterns_trained_graph.jld2"), trainer.prototype_graph)
    close_trainer!(trainer)
    println("Saved CSV: $csv_path")
    println("Saved plot: $png_path")
    println("Saved graph: $graph_path")
    return nothing
end

main()
