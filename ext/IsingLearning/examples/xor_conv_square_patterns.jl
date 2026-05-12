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

# Multiplexed XOR pattern experiment with a square-local, convnet-like graph:
# input 16x16 -> hidden 16x16 -> hidden 16x16 -> output 16x16 by default.
# The XOR code itself is still a 4x4 multiplexed code embedded on an evenly
# spaced grid inside the 16x16 input/output layers.
#
# Hidden layers have trainable internal square-neighborhood connections. Adjacent
# layer connections are also square-local in shared layer coordinates.
const EPOCHS = parse(Int, get(ENV, "ISING_XOR_CONV_EPOCHS", "1000"))
const LOG_EVERY = parse(Int, get(ENV, "ISING_XOR_CONV_LOG_EVERY", "100"))
const MINIT = parse(Int, get(ENV, "ISING_XOR_CONV_MINIT", "4"))
const EVAL_REPEATS = parse(Int, get(ENV, "ISING_XOR_CONV_EVAL_REPEATS", "16"))
const NWORKERS = parse(Int, get(ENV, "ISING_XOR_CONV_THREADS", string(max(1, min(8, Threads.nthreads())))))
const SIDE = parse(Int, get(ENV, "ISING_XOR_CONV_SIDE", "16"))
const CODE_SIDE = parse(Int, get(ENV, "ISING_XOR_CONV_CODE_SIDE", "4"))
const INPUT_CODE_MODE = Symbol(get(ENV, "ISING_XOR_CONV_INPUT_CODE", "orthogonal"))
const HIDDEN_SIDE = parse(Int, get(ENV, "ISING_XOR_CONV_HIDDEN_SIDE", string(SIDE)))
const HIDDEN_LAYERS = parse(Int, get(ENV, "ISING_XOR_CONV_HIDDEN_LAYERS", "2"))
const INPUT_UNITS = SIDE * SIDE
const OUTPUT_UNITS = SIDE * SIDE
const CODE_UNITS = CODE_SIDE * CODE_SIDE
const HIDDEN_UNITS = HIDDEN_SIDE * HIDDEN_SIDE
const BETA = parse(FT, get(ENV, "ISING_XOR_CONV_BETA", "0.05"))
const LEARNING_RATE = parse(FT, get(ENV, "ISING_XOR_CONV_LR", "0.005"))
const WEIGHT_DECAY = parse(FT, get(ENV, "ISING_XOR_CONV_WEIGHT_DECAY", "1e-4"))
const GRAD_CLIP = parse(FT, get(ENV, "ISING_XOR_CONV_GRAD_CLIP", "Inf"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_WEIGHT_SCALE", "0.02"))
const INTERNAL_WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_INTERNAL_WEIGHT_SCALE", string(WEIGHT_SCALE)))
const INPUT_INTERNAL_WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_INPUT_INTERNAL_WEIGHT_SCALE", string(INTERNAL_WEIGHT_SCALE)))
const HIDDEN_INTERNAL_WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_HIDDEN_INTERNAL_WEIGHT_SCALE", string(INTERNAL_WEIGHT_SCALE)))
const OUTPUT_INTERNAL_WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_OUTPUT_INTERNAL_WEIGHT_SCALE", string(INTERNAL_WEIGHT_SCALE)))
const KERNEL_RADIUS = parse(Int, get(ENV, "ISING_XOR_CONV_KERNEL_RADIUS", "5"))
const INTERNAL_NN = parse(Int, get(ENV, "ISING_XOR_CONV_INTERNAL_NN", "5"))
const BIAS_SCALE = parse(FT, get(ENV, "ISING_XOR_CONV_BIAS_SCALE", "0.1"))
const TEMP = parse(FT, get(ENV, "ISING_XOR_CONV_TEMP", "0.001"))
const STEPSIZE = parse(FT, get(ENV, "ISING_XOR_CONV_STEPSIZE", "0.05"))
const BLOCK_SIZE = parse(Int, get(ENV, "ISING_XOR_CONV_BLOCK_SIZE", "16"))
const FREE_RELAXATION = parse(Int, get(ENV, "ISING_XOR_CONV_FREE_RELAXATION", "50"))
const NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_XOR_CONV_NUDGED_RELAXATION", "50"))
const WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_CONV_WEIGHT_SEED", "2"))
const INTERNAL_WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_CONV_INTERNAL_WEIGHT_SEED", "3"))
const BIAS_SEED = parse(Int, get(ENV, "ISING_XOR_CONV_BIAS_SEED", "11"))
const BASE_SEED = parse(Int, get(ENV, "ISING_XOR_CONV_BASE_SEED", "82000"))
const EVAL_SEED_OFFSET = parse(Int, get(ENV, "ISING_XOR_CONV_EVAL_SEED_OFFSET", "30000000"))
const OUTDIR = get(
    ENV,
    "ISING_XOR_CONV_DIR",
    joinpath(@__DIR__, "..", "runs", "xor_conv_square_patterns_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
)

const CASES = ((false, false), (false, true), (true, false), (true, true))
const CASE_NAMES = ("ff", "ft", "tf", "tt")

case_index(a::Bool, b::Bool) = (a ? 2 : 0) + (b ? 1 : 0) + 1
xor_label(a::Bool, b::Bool) = xor(a, b)

function require_even_side()
    iseven(CODE_SIDE) || error("CODE_SIDE must be even for the default orthogonal multiplexed patterns")
    SIDE >= CODE_SIDE || error("SIDE must be at least CODE_SIDE; got $SIDE and $CODE_SIDE")
    SIDE == HIDDEN_SIDE || error("Default square-local geometry expects SIDE == HIDDEN_SIDE; got $SIDE and $HIDDEN_SIDE")
    HIDDEN_LAYERS in (1, 2) || error("ISING_XOR_CONV_HIDDEN_LAYERS must be 1 or 2 for this example")
    INPUT_CODE_MODE in (:orthogonal, :onehot) ||
        error("ISING_XOR_CONV_INPUT_CODE must be :orthogonal or :onehot")
    return nothing
end

function code_positions()
    return [floor(Int, (idx - FT(0.5)) * SIDE / CODE_SIDE) + 1 for idx in 1:CODE_SIDE]
end

const CODE_POSITIONS = code_positions()

function code_indices(positions = CODE_POSITIONS)
    linear = LinearIndices((SIDE, SIDE))
    return [linear[i, j] for j in positions for i in positions]
end

function vertical_pattern()
    pattern = zeros(FT, SIDE, SIDE)
    split = CODE_SIDE ÷ 2
    for (jj, j) in enumerate(CODE_POSITIONS), (ii, i) in enumerate(CODE_POSITIONS)
        pattern[i, j] = ii <= split ? -one(FT) : one(FT)
    end
    return vec(pattern)
end

function horizontal_pattern()
    pattern = zeros(FT, SIDE, SIDE)
    split = CODE_SIDE ÷ 2
    for (jj, j) in enumerate(CODE_POSITIONS), (ii, i) in enumerate(CODE_POSITIONS)
        pattern[i, j] = jj <= split ? -one(FT) : one(FT)
    end
    return vec(pattern)
end

const CODE_IDXS = code_indices()
const P_VERTICAL = vertical_pattern()
const P_HORIZONTAL = horizontal_pattern()
const P_CHECKERBOARD = zeros(FT, INPUT_UNITS)
P_CHECKERBOARD[CODE_IDXS] .= P_VERTICAL[CODE_IDXS] .* P_HORIZONTAL[CODE_IDXS]

function orthogonal_input_patterns()
    return (
        let p = zeros(FT, INPUT_UNITS)
            p[CODE_IDXS] .= one(FT)
            p
        end,
        P_VERTICAL,
        P_HORIZONTAL,
        P_CHECKERBOARD,
    )
end

function onehot_input_patterns()
    patterns = ntuple(_ -> fill(zero(FT), INPUT_UNITS), 4)
    for case_idx in 1:4
        patterns[case_idx][CODE_IDXS] .= -one(FT)
        patterns[case_idx][CODE_IDXS[case_idx]] = one(FT)
    end
    return patterns
end

const INPUT_PATTERNS = INPUT_CODE_MODE === :onehot ? onehot_input_patterns() : orthogonal_input_patterns()

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
    INPUT_CODE_MODE === :orthogonal && check_pairwise_orthogonal("input", INPUT_PATTERNS)
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
    return -sum(abs2, output[CODE_IDXS] .- target[CODE_IDXS]) / FT(CODE_UNITS)
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
    radius = sqrt(FT(2 * KERNEL_RADIUS^2)) + FT(1e-6)
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> dr <= radius ? WEIGHT_SCALE * randn(rng, FT) : zero(FT),
        rng,
    )
end

function internal_nn_weight_generator(seed::Integer, scale::Real)
    rng = Random.MersenneTwister(seed)
    scale_ft = FT(scale)
    return II.WeightGeneratorOld(dr -> scale_ft * randn(rng, FT), INTERNAL_NN, rng)
end

function bias_generator()
    rng = Random.MersenneTwister(BIAS_SEED)
    return g -> BIAS_SCALE .* randn(rng, FT, II.statelen(g))
end

function zero_embedded_inactive_sites!(g)
    output_idxs = collect(II.graphidxs(g[end]))
    output_code = embedded_output_code_global_idxs(g)
    output_code_lookup = Set(output_code)
    @inbounds for idx in output_idxs
        idx in output_code_lookup && continue
        II.state(g)[idx] = zero(eltype(g))
    end
    return g
end

struct EmbeddedInputCodeIndexSet{I} <: II.UniformIndexPicker
    unclamped::Vector{I}
    input_clamped::Vector{I}
    active::Base.RefValue{Vector{I}}
    changed::Base.RefValue{Bool}
end

function embedded_input_code_global_idxs(g)
    return collect(II.graphidxs(g[1]))[CODE_IDXS]
end

function embedded_output_code_global_idxs(g)
    return collect(II.graphidxs(g[end]))[CODE_IDXS]
end

function EmbeddedInputCodeIndexSet(g)
    all_idxs = collect(II.graphidxs(g))
    input_code = embedded_input_code_global_idxs(g)
    output_code = embedded_output_code_global_idxs(g)
    input_code_lookup = Set(input_code)
    output_code_lookup = Set(output_code)
    output_layer_lookup = Set(collect(II.graphidxs(g[end])))

    # `apply_input` calls `off!(index_set, 1)` before every free/nudged run.
    # For this embedded-code experiment, that should freeze only the 4x4 input
    # code sites, not the full 16x16 input layer. The rest of the input layer
    # remains active and relaxes with the graph.
    #
    # The supervised output is also embedded in a 4x4 code inside a 16x16 layer.
    # The current direct Clamping term targets the full output layer, so sampling
    # non-code output sites would spend most of the nudged dynamics forcing those
    # sites toward zero. Keep only the output code sites active.
    output_noncode(idx) = (idx in output_layer_lookup) && !(idx in output_code_lookup)
    unclamped = [idx for idx in all_idxs if !output_noncode(idx)]
    input_clamped = [idx for idx in unclamped if !(idx in input_code_lookup)]

    return EmbeddedInputCodeIndexSet(unclamped, input_clamped, Ref(input_clamped), Ref(false))
end

@inline II.sampling_indices(is::EmbeddedInputCodeIndexSet) = is.active[]
@inline II.consume_changed!(is::EmbeddedInputCodeIndexSet) = (changed = is.changed[]; is.changed[] = false; changed)
@inline II.pick_idx(rng::Random.AbstractRNG, is::EmbeddedInputCodeIndexSet) = rand(rng, is.active[])

@inline function _set_active!(is::EmbeddedInputCodeIndexSet, active)
    if is.active[] !== active
        is.active[] = active
        is.changed[] = true
    end
    return is
end

@inline function II.off!(is::EmbeddedInputCodeIndexSet, layer_idx::Int)
    layer_idx == 1 && return _set_active!(is, is.input_clamped)
    return is
end

@inline function II.on!(is::EmbeddedInputCodeIndexSet, layer_idx::Int)
    layer_idx == 1 && return _set_active!(is, is.unclamped)
    return is
end

function xor_graph()
    input_wg = internal_nn_weight_generator(INTERNAL_WEIGHT_SEED, INPUT_INTERNAL_WEIGHT_SCALE)
    hidden_wgs = [
        internal_nn_weight_generator(INTERNAL_WEIGHT_SEED + idx, HIDDEN_INTERNAL_WEIGHT_SCALE)
        for idx in 1:HIDDEN_LAYERS
    ]
    output_wg = internal_nn_weight_generator(INTERNAL_WEIGHT_SEED + HIDDEN_LAYERS + 1, OUTPUT_INTERNAL_WEIGHT_SCALE)
    layers = Any[
        II.Layer(SIDE, SIDE, II.StateSet(-one(FT), one(FT)), input_wg, II.Continuous(), II.Coords(0, 1, 0); periodic = false),
    ]
    for hidden_idx in 1:HIDDEN_LAYERS
        push!(
            layers,
            II.Layer(
                HIDDEN_SIDE,
                HIDDEN_SIDE,
                II.StateSet(-one(FT), one(FT)),
                hidden_wgs[hidden_idx],
                II.Continuous(),
                II.Coords(0, hidden_idx + 1, 0);
                periodic = false,
            ),
        )
    end
    push!(
        layers,
        II.Layer(SIDE, SIDE, II.StateSet(-one(FT), one(FT)), output_wg, II.Continuous(), II.Coords(0, HIDDEN_LAYERS + 2, 0); periodic = false),
    )

    wg = signed_weight_generator()
    layer_args = Any[layers[1]]
    for idx in 2:length(layers)
        push!(layer_args, deepcopy(wg), layers[idx])
    end

    clamping_target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    clamping_beta = II.UniformArray(zero(FT))
    hamiltonian = II.Bilinear() + II.MagField(b = bias_generator()) +
        II.Clamping(β = clamping_beta, y = clamping_target)

    graph = II.IsingGraph(layer_args..., hamiltonian; precision = FT, index_set = EmbeddedInputCodeIndexSet)
    graph.addons[:after_apply_input!] = zero_embedded_inactive_sites!
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
    hasproperty(worker.context.dynamics, :rng) && Random.seed!(worker.context.dynamics.rng, seed)
    hasproperty(worker.context, :nudged_dynamics) &&
        hasproperty(worker.context.nudged_dynamics, :rng) &&
        Random.seed!(worker.context.nudged_dynamics.rng, seed + 1)
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
    return completed_training_response(worker)
end

function completed_training_response(worker)
    free_state = worker.context._state.equilibrium_state
    plus_state = worker.context.plus_capture.captured
    minus_state = worker.context.minus_capture.captured
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

function clip_gradient!(gradient)
    grad_norm = sqrt(sum(abs2, gradient.w) + sum(abs2, gradient.b))
    if isfinite(GRAD_CLIP) && grad_norm > GRAD_CLIP
        scale = GRAD_CLIP / grad_norm
        gradient.w .*= scale
        gradient.b .*= scale
        return GRAD_CLIP
    end
    return grad_norm
end

function train_epoch!(trainer, x, y, batch_gradient, epoch::Integer)
    IsingLearning._reset_batch_buffers!(trainer)
    workers = trainer.workers

    total_response = zero(FT)
    trajectories = [(sample_idx, init_idx) for sample_idx in axes(x, 2) for init_idx in 1:MINIT]
    next_traj = 1
    completed = 0

    while completed < length(trajectories)
        made_progress = false

        for worker in workers
            if !isnothing(worker.task) && Processes.isdone(worker)
                total_response += completed_training_response(worker)
                close(worker)
                completed += 1
                made_progress = true
            end

            if isnothing(worker.task) && next_traj <= length(trajectories)
                sample_idx, init_idx = trajectories[next_traj]
                seed = BASE_SEED + 1_000_000 * epoch + 10_000 * sample_idx + init_idx
                seed_worker!(worker, seed)
                IsingLearning._write_example!(worker, view(x, :, sample_idx), view(y, :, sample_idx))
                Processes.reset!(worker)

                # This is where the existing EP composite is executed.
                # `init_mnist_trainer` builds each worker from
                # `_worker_process(layer, graph)`, which resolves
                # `Forward_and_Nudged(layer)` into a `Process`.
                run(worker)
                next_traj += 1
                made_progress = true
            end
        end

        made_progress || yield()
    end

    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, length(trajectories))
    add_weight_decay!(batch_gradient, trainer.params)
    all(isfinite, batch_gradient.w) || error("non-finite weight gradient")
    all(isfinite, batch_gradient.b) || error("non-finite bias gradient")

    grad_norm = clip_gradient!(batch_gradient)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    all(isfinite, trainer.params.w) || error("non-finite weights")
    all(isfinite, trainer.params.b) || error("non-finite biases")
    return (; grad_norm, response_norm = total_response / FT(max(length(trajectories), 1)))
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
        var += sum(abs2, output[CODE_IDXS] .- mean_output[CODE_IDXS]) / FT(CODE_UNITS)
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
        mse += sum(abs2, outputs[sample_idx][CODE_IDXS] .- target[CODE_IDXS]) / FT(CODE_UNITS)
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
        target_name = xor_label(case...) ? "horizontal" : "vertical"
        println(
            "    case=$case target=$target_name ",
            "margin=$(round(output_margin(metrics.outputs[idx]), digits = 5)) ",
            "code_mean_head=", round.(metrics.outputs[idx][CODE_IDXS[1:min(6, end)]]; digits = 4),
            " std=$(round(metrics.stds[idx], digits = 5))",
        )
    end
end

function main()
    MINIT > 0 || error("ISING_XOR_CONV_MINIT must be positive")
    EVAL_REPEATS > 0 || error("ISING_XOR_CONV_EVAL_REPEATS must be positive")
    validate_patterns!()
    mkpath(OUTDIR)

    Random.seed!(BASE_SEED)
    graph = xor_graph()
    layer = xor_layer(graph)
    x, y = xor_dataset()

    # This reuses the normal IsingLearning training path. The worker process
    # contains the resolved `Forward_and_Nudged` composite; this file only
    # schedules repeated XOR trajectories across those workers.
    trainer = init_mnist_trainer(layer; graph, numthreads = NWORKERS, optimiser = Optimisers.Adam(LEARNING_RATE))
    set_trainer_temperature!(trainer, TEMP)

    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    rows = Dict{String, Any}[]
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    println(
        "Running square-local multiplexed XOR pattern experiment: ",
        (epochs = EPOCHS, log_every = LOG_EVERY, minit = MINIT, eval_repeats = EVAL_REPEATS,
         workers = NWORKERS,
         input_code = INPUT_CODE_MODE,
         input = (SIDE, SIDE), hidden_layers = HIDDEN_LAYERS, hidden = (HIDDEN_SIDE, HIDDEN_SIDE),
         output = (SIDE, SIDE),
         kernel_radius = KERNEL_RADIUS,
         internal_nn = INTERNAL_NN,
         input_code_positions = code_positions(),
         input_gram_offdiag = maximum(abs(dot(INPUT_PATTERNS[i], INPUT_PATTERNS[j])) for i in 1:4 for j in 1:4 if i != j),
         output_dot = dot(P_VERTICAL, P_HORIZONTAL),
         beta = BETA, lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY,
         internal_scales = (input = INPUT_INTERNAL_WEIGHT_SCALE, hidden = HIDDEN_INTERNAL_WEIGHT_SCALE, output = OUTPUT_INTERNAL_WEIGHT_SCALE),
         temp = TEMP, stepsize = STEPSIZE, block_size = BLOCK_SIZE,
         free_relaxation = FREE_RELAXATION, nudged_relaxation = NUDGED_RELAXATION),
    )

    validation_seed_offset = BASE_SEED + EVAL_SEED_OFFSET
    before = evaluate_xor!(trainer, x, y; seed_offset = validation_seed_offset)
    push!(rows, metric_row(0, before, zero_grad, trainer, initial_params))
    println("epoch=0 mse=$(round(before.mse, digits=6)) acc=$(before.accuracy)")
    print_case_outputs(before)

    best_accuracy = before.accuracy
    best_mse = before.mse
    best_params = deepcopy(trainer.params)

    for epoch in 1:EPOCHS
        grad_metrics = train_epoch!(trainer, x, y, batch_gradient, epoch)
        if epoch % LOG_EVERY == 0 || epoch == 1 || epoch == EPOCHS
            metrics = evaluate_xor!(trainer, x, y; seed_offset = validation_seed_offset)
            row = metric_row(epoch, metrics, grad_metrics, trainer, initial_params)
            push!(rows, row)
            if metrics.accuracy > best_accuracy || (metrics.accuracy == best_accuracy && metrics.mse < best_mse)
                best_accuracy = metrics.accuracy
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

    csv_path = write_csv(joinpath(OUTDIR, "xor_conv_square_patterns.csv"), rows)
    png_path = plot_metrics(joinpath(OUTDIR, "xor_conv_square_patterns.png"), rows)
    graph_path = II.save_isinggraph(joinpath(OUTDIR, "xor_conv_square_patterns_trained_graph.jld2"), trainer.prototype_graph)
    close_trainer!(trainer)
    println("Saved CSV: $csv_path")
    println("Saved plot: $png_path")
    println("Saved graph: $graph_path")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
