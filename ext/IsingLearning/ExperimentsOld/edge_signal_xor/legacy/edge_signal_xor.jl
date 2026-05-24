using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using Statistics
using LinearAlgebra
using Dates
using CairoMakie
import IsingLearning.InteractiveIsing: @WG, WeightGenerator

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float64

const XOR_CASES = ((false, false), (false, true), (true, false), (true, true))
const INPUT_HIDDEN_SCALE_REF = Ref{FT}(0.08)
const INPUT_HIDDEN_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))
const HIDDEN_LOCAL_SCALE_REF = Ref{FT}(0.04)
const HIDDEN_LOCAL_NN_REF = Ref{Int}(1)
const HIDDEN_LOCAL_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))
const HIDDEN_OUTPUT_SCALE_REF = Ref{FT}(0.08)
const HIDDEN_OUTPUT_RNG_REF = Ref{Random.MersenneTwister}(Random.MersenneTwister(0))
const HIDDEN_HEIGHT_REF = Ref{Int}(8)
const HIDDEN_WIDTH_REF = Ref{Int}(8)

"""Named input-hidden weight function used by `@WG` keyword dispatch."""
function input_hidden_edge_weight(; c2)
    return left_hidden_edge(c2) ? INPUT_HIDDEN_SCALE_REF[] * randn(INPUT_HIDDEN_RNG_REF[], FT) : zero(FT)
end

"""Named hidden local weight function used by `WeightGenerator` keyword dispatch."""
function hidden_local_weight(; dr)
    return dr <= HIDDEN_LOCAL_NN_REF[] ? HIDDEN_LOCAL_SCALE_REF[] * randn(HIDDEN_LOCAL_RNG_REF[], FT) : zero(FT)
end

"""Named hidden-output weight function used by `@WG` keyword dispatch."""
function hidden_output_edge_weight(; c1)
    return right_hidden_edge(c1) ? HIDDEN_OUTPUT_SCALE_REF[] * randn(HIDDEN_OUTPUT_RNG_REF[], FT) : zero(FT)
end

"""
    EdgeSignalXORConfig(; kwargs...)

Configuration for the edge-driven scalar XOR experiment.

The graph is `2 input spins -> 8x8 hidden spins -> 1 output spin`. The two
input spins connect only to the left edge of the hidden layer. The scalar
output spin connects only to the right edge of the hidden layer. The hidden
layer has learned local interactions with distance cutoff `hidden_nn`.
"""
Base.@kwdef struct EdgeSignalXORConfig
    epochs::Int = parse(Int, get(ENV, "EDGE_XOR_EPOCHS", "2500"))
    log_every::Int = parse(Int, get(ENV, "EDGE_XOR_LOG_EVERY", "250"))
    minit::Int = parse(Int, get(ENV, "EDGE_XOR_MINIT", "8"))
    eval_repeats::Int = parse(Int, get(ENV, "EDGE_XOR_EVAL_REPEATS", "16"))
    free_relaxation::Int = parse(Int, get(ENV, "EDGE_XOR_FREE", "600"))
    nudged_relaxation::Int = parse(Int, get(ENV, "EDGE_XOR_NUDGED", "600"))
    validation_relaxation::Int = parse(Int, get(ENV, "EDGE_XOR_VALIDATION", "1200"))
    β::FT = parse(FT, get(ENV, "EDGE_XOR_BETA", "2.0"))
    lr::FT = parse(FT, get(ENV, "EDGE_XOR_LR", "0.003"))
    weight_decay::FT = parse(FT, get(ENV, "EDGE_XOR_WEIGHT_DECAY", "0.0"))
    stepsize::FT = parse(FT, get(ENV, "EDGE_XOR_STEPSIZE", "0.4"))
    temp_fraction::FT = parse(FT, get(ENV, "EDGE_XOR_TEMP_FRACTION", "0.025"))
    input_hidden_scale::FT = parse(FT, get(ENV, "EDGE_XOR_INPUT_HIDDEN_SCALE", "0.08"))
    hidden_local_scale::FT = parse(FT, get(ENV, "EDGE_XOR_HIDDEN_LOCAL_SCALE", "0.04"))
    hidden_output_scale::FT = parse(FT, get(ENV, "EDGE_XOR_HIDDEN_OUTPUT_SCALE", "0.08"))
    bias_scale::FT = parse(FT, get(ENV, "EDGE_XOR_BIAS_SCALE", "0.02"))
    hidden_height::Int = parse(Int, get(ENV, "EDGE_XOR_HIDDEN_HEIGHT", "8"))
    hidden_width::Int = parse(Int, get(ENV, "EDGE_XOR_HIDDEN_WIDTH", "8"))
    hidden_nn::Int = parse(Int, get(ENV, "EDGE_XOR_HIDDEN_NN", "1"))
    target_scale::FT = parse(FT, get(ENV, "EDGE_XOR_TARGET_SCALE", "1.0"))
    response_pre_sweeps::Int = parse(Int, get(ENV, "EDGE_XOR_RESPONSE_PRE_SWEEPS", "60"))
    response_sweeps::Int = parse(Int, get(ENV, "EDGE_XOR_RESPONSE_SWEEPS", "80"))
    response_repeats::Int = parse(Int, get(ENV, "EDGE_XOR_RESPONSE_REPEATS", "8"))
    skip_response::Bool = parse(Bool, get(ENV, "EDGE_XOR_SKIP_RESPONSE", "false"))
    weight_seed::Int = parse(Int, get(ENV, "EDGE_XOR_WEIGHT_SEED", "41"))
    bias_seed::Int = parse(Int, get(ENV, "EDGE_XOR_BIAS_SEED", "43"))
    base_seed::Int = parse(Int, get(ENV, "EDGE_XOR_BASE_SEED", "405000"))
end

"""Return the four physical XOR input vectors and scalar XOR targets."""
function xor_dataset(config::EdgeSignalXORConfig)
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 1, 4)
    for (idx, (a, b)) in enumerate(XOR_CASES)
        x[:, idx] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        y[1, idx] = config.target_scale * (xor(a, b) ? one(FT) : -one(FT))
    end
    return x, y
end

"""Return true when `c` lies on the left edge of the hidden layer."""
left_hidden_edge(c) = c[2] == 1

"""Return true when `c` lies on the right edge of the hidden layer."""
right_hidden_edge(c, hidden_width::Integer = HIDDEN_WIDTH_REF[]) = c[2] == hidden_width

"""Create a random generator from the input layer to the hidden left edge."""
function input_to_left_edge_generator(scale::FT, rng)
    INPUT_HIDDEN_SCALE_REF[] = scale
    INPUT_HIDDEN_RNG_REF[] = rng
    return @WG input_hidden_edge_weight NN = :all symmetric = true
end

"""Create a random generator from the hidden right edge to the scalar output."""
function right_edge_to_output_generator(scale::FT, rng)
    HIDDEN_OUTPUT_SCALE_REF[] = scale
    HIDDEN_OUTPUT_RNG_REF[] = rng
    return @WG hidden_output_edge_weight NN = :all symmetric = true
end

"""Create a random symmetric hidden-layer local coupling generator."""
function hidden_local_generator(scale::FT, nn::Integer, rng)
    HIDDEN_LOCAL_SCALE_REF[] = scale
    HIDDEN_LOCAL_NN_REF[] = Int(nn)
    HIDDEN_LOCAL_RNG_REF[] = rng
    if nn == 0
        return @WG hidden_local_weight NN = 0 symmetric = true
    elseif nn == 1
        return @WG hidden_local_weight NN = 1 symmetric = true
    elseif nn == 2
        return @WG hidden_local_weight NN = 2 symmetric = true
    elseif nn == 3
        return @WG hidden_local_weight NN = 3 symmetric = true
    elseif nn == 4
        return @WG hidden_local_weight NN = 4 symmetric = true
    elseif nn == 5
        return @WG hidden_local_weight NN = 5 symmetric = true
    else
        error("hidden_nn=$(nn) is not listed in this experiment-local @WG dispatch table")
    end
end

"""Return the maximum absolute column sum of the graph adjacency."""
function max_interaction_scale(graph)
    A = II.adj(graph).sp
    isempty(A) && return zero(FT)
    return FT(maximum(vec(sum(abs, A; dims = 1))))
end

"""Set the simulation temperature from the current maximum interaction scale."""
function set_relative_temperature!(graph, config::EdgeSignalXORConfig)
    scale = max(max_interaction_scale(graph), eps(FT))
    temp = config.temp_fraction * scale
    II.temp!(graph, temp)
    return temp
end

"""Build the edge-driven `2 -> 8x8 -> 1` Ising graph."""
function edge_signal_graph(config::EdgeSignalXORConfig)
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

"""Return the LocalLangevin sampler used by both learning and response runs."""
function edge_sampler(config::EdgeSignalXORConfig)
    return II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
end

"""Wrap the edge graph in the existing IsingLearning layer interface."""
function edge_layer(graph, config::EdgeSignalXORConfig)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> edge_signal_graph(config);
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

"""Initialize the standard IsingLearning trainer on a single worker thread."""
function edge_trainer(config::EdgeSignalXORConfig)
    graph = edge_signal_graph(config)
    layer = edge_layer(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    return init_mnist_trainer(layer; graph, numthreads = 1, optimiser)
end

"""Repeat the four XOR samples to average stochastic equilibrium-prop gradients."""
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

"""Run one validation relaxation for one input and return the scalar output."""
function scalar_output!(trainer, x, config::EdgeSignalXORConfig; seed::Integer)
    graph = trainer.validation_graph
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)
    run_dynamics_steps!(graph, edge_sampler(config), config.validation_relaxation; seed)
    return only(copy(II.state(graph[end])))
end

"""Evaluate repeated-start scalar outputs for all XOR inputs."""
function evaluate!(trainer, x, y, config::EdgeSignalXORConfig; seed_offset::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = scalar_output!(
                trainer,
                view(x, :, sample_idx),
                config;
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

"""Record one row of learning diagnostics."""
function push_learning_row!(rows, epoch, metrics, grad_norm, temp)
    row = Dict{String,Any}(
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.acc,
        "margin" => metrics.margin,
        "grad_norm" => grad_norm,
        "temperature" => temp,
    )
    for i in eachindex(metrics.means)
        row["mean_$i"] = metrics.means[i]
        row["std_$i"] = metrics.stds[i]
    end
    push!(rows, row)
    return rows
end

"""Write a vector of dictionary rows as a CSV file."""
function write_csv(path, rows)
    mkpath(dirname(path))
    isempty(rows) && return path
    headers = sort!(collect(keys(first(rows))))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((row[h] for h in headers), ","))
        end
    end
    return path
end

"""Plot the learning metrics for one run."""
function plot_learning(path, rows)
    fig = Figure(size = (1100, 760))
    ax1 = Axis(fig[1, 1], title = "edge XOR MSE", xlabel = "epoch", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy")
    ax3 = Axis(fig[2, 1], title = "output margin", xlabel = "epoch", ylabel = "min |mean output|")
    ax4 = Axis(fig[2, 2], title = "gradient norm", xlabel = "epoch", ylabel = "||grad||")
    epochs = [row["epoch"] for row in rows]
    lines!(ax1, epochs, [row["mse"] for row in rows])
    lines!(ax2, epochs, [row["accuracy"] for row in rows])
    lines!(ax3, epochs, [row["margin"] for row in rows])
    lines!(ax4, epochs, [row["grad_norm"] for row in rows])
    save(path, fig)
    return path
end

"""Remove construction-time random generators before saving a learned graph."""
function strip_weight_generators!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

"""Train one edge-signal graph and return the best parameters found."""
function train_edge_xor(config::EdgeSignalXORConfig; outdir)
    trainer = edge_trainer(config)
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

"""Mutable container used by the response logger process algorithm."""
mutable struct ResponseTrace
    rows::Vector{Dict{String,Any}}
    profiles::Vector{Dict{String,Any}}
end

"""Create an empty response trace."""
ResponseTrace() = ResponseTrace(Dict{String,Any}[], Dict{String,Any}[])

"""ProcessAlgorithm that records response metrics after scheduled dynamics steps."""
struct SweepResponseLogger{Trace,State,Kind} <: Processes.ProcessAlgorithm
    trace::Trace
    pre_state::State
    graph_kind::Kind
    nn::Int
    source_case::Int
    target_case::Int
    repeat_idx::Int
    sweep::Base.RefValue{Int}
end

Processes.init(::SweepResponseLogger, context) = (;)

function Processes.step!(logger::SweepResponseLogger, context)
    log_response!(
        logger.trace,
        context.model,
        logger.pre_state;
        graph_kind = logger.graph_kind,
        nn = logger.nn,
        source_case = logger.source_case,
        target_case = logger.target_case,
        repeat_idx = logger.repeat_idx,
        sweep = logger.sweep[],
    )
    logger.sweep[] += 1
    return (;)
end

"""Return hidden, left-edge, right-edge, and output index ranges for a graph."""
function response_indices(graph)
    hidden_range = collect(II.layerrange(graph[2]))
    output_idx = only(II.layerrange(graph[end]))
    hidden_state = II.state(graph[2])
    hidden_matrix = reshape(hidden_range, size(hidden_state))
    left_edge = collect(hidden_matrix[:, 1])
    right_edge = collect(hidden_matrix[:, end])
    return (; hidden_range, output_idx, left_edge, right_edge)
end

"""Append scalar and column-profile response diagnostics to `trace`."""
function log_response!(trace::ResponseTrace, graph, pre_state; graph_kind, nn, source_case, target_case, repeat_idx, sweep)
    idxs = response_indices(graph)
    s = II.state(graph)
    Δhidden = reshape(s[idxs.hidden_range] .- pre_state[idxs.hidden_range], size(II.state(graph[2])))
    col_abs = vec(mean(abs.(Δhidden); dims = 1))
    total_abs = sum(col_abs)
    front = total_abs == 0 ? zero(FT) : sum((1:8) .* col_abs) / total_abs
    output_response = FT(s[idxs.output_idx] - pre_state[idxs.output_idx])
    right_response = FT(mean(s[idxs.right_edge] .- pre_state[idxs.right_edge]))
    left_response = FT(mean(s[idxs.left_edge] .- pre_state[idxs.left_edge]))

    push!(trace.rows, Dict{String,Any}(
        "graph_kind" => graph_kind,
        "nn" => nn,
        "source_case" => source_case,
        "target_case" => target_case,
        "repeat" => repeat_idx,
        "sweep" => sweep,
        "output_response" => output_response,
        "right_edge_response" => right_response,
        "left_edge_response" => left_response,
        "total_abs_hidden_response" => FT(total_abs),
        "front_position" => FT(front),
    ))

    for col in axes(Δhidden, 2)
        push!(trace.profiles, Dict{String,Any}(
            "graph_kind" => graph_kind,
            "nn" => nn,
            "source_case" => source_case,
            "target_case" => target_case,
            "repeat" => repeat_idx,
            "sweep" => sweep,
            "column" => col,
            "mean_response" => FT(mean(@view Δhidden[:, col])),
            "mean_abs_response" => FT(mean(abs, @view Δhidden[:, col])),
        ))
    end
    return trace
end

"""Freeze the input layer and write a two-spin XOR input vector."""
function apply_edge_input!(graph, x)
    II.off!(graph.index_set, 1)
    II.state(graph[1]) .= reshape(x, size(II.state(graph[1])))
    return graph
end

"""Randomize every layer state using the graph's existing layer initializers."""
function randomize_graph_state!(graph)
    for layer_idx in 1:length(getfield(graph, :layers))
        II.initstate!(graph[layer_idx])
    end
    return graph
end

"""Run a fixed number of single-spin LocalLangevin steps through Processes."""
function run_dynamics_steps!(graph, sampler, nsteps::Integer; seed::Integer)
    dynamics = deepcopy(sampler)
    routine = II.Processes.@Routine begin
        @repeat nsteps dynamics()
    end
    inputs = (Processes.Init(dynamics, model = graph, rng = Random.MersenneTwister(seed)),)
    process = Processes.Process(Processes.resolve(routine), inputs...; repeats = 1)
    run(process)
    wait(process)
    close(process)
    return graph
end

"""Run a scheduled response trace from one input case to another."""
function run_transition_response!(trace, graph, sampler, x; graph_kind, nn, source_case, target_case, repeat_idx, config)
    seed_base = config.base_seed + 100_000_000 + 1_000_000 * nn + 10_000 * source_case + 100 * target_case + repeat_idx
    Random.seed!(seed_base)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, view(x, :, source_case))
    sweep_steps = length(II.sampling_indices(graph.index_set))
    run_dynamics_steps!(graph, sampler, config.response_pre_sweeps * sweep_steps; seed = seed_base + 1)
    pre_state = copy(II.state(graph))

    apply_edge_input!(graph, view(x, :, target_case))
    log_response!(trace, graph, pre_state;
        graph_kind, nn, source_case, target_case, repeat_idx, sweep = 0)

    sweep = Ref(1)
    logger = SweepResponseLogger(trace, pre_state, graph_kind, nn, source_case, target_case, repeat_idx, sweep)
    dynamics = deepcopy(sampler)
    routine = II.Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @every sweep_steps logger(model = dynamics.model)
    end
    total_steps = config.response_sweeps * sweep_steps
    wrapped = II.Processes.@Routine begin
        @repeat total_steps routine()
    end
    inputs = (Processes.Init(dynamics, model = graph, rng = Random.MersenneTwister(seed_base + 2)),)
    process = Processes.Process(Processes.resolve(wrapped), inputs...; repeats = 1)
    run(process)
    wait(process)
    close(process)
    return trace
end

"""Apply a saved parameter tuple to a fresh graph and return that graph."""
function graph_from_params(config::EdgeSignalXORConfig, params)
    graph = edge_signal_graph(config)
    layer = edge_layer(graph, config)
    trainer = init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(config.lr))
    trainer.params = deepcopy(params)
    IsingLearning._broadcast_params!(trainer)
    result = deepcopy(trainer.prototype_graph)
    close_trainer!(trainer)
    return result
end

"""Measure response traces for the random-initialized and learned graph."""
function measure_responses(config::EdgeSignalXORConfig, trained; outdir)
    x, _ = xor_dataset(config)
    sampler = edge_sampler(config)
    trace = ResponseTrace()
    graph_specs = (
        ("initial", graph_from_params(config, trained.initial_params)),
        ("learned", graph_from_params(config, trained.best_params)),
    )
    for (graph_kind, graph) in graph_specs
        for source_case in axes(x, 2), target_case in axes(x, 2)
            source_case == target_case && continue
            for repeat_idx in 1:config.response_repeats
                run_transition_response!(
                    trace,
                    graph,
                    sampler,
                    x;
                    graph_kind,
                    nn = config.hidden_nn,
                    source_case,
                    target_case,
                    repeat_idx,
                    config,
                )
            end
        end
    end
    rows_csv = write_csv(joinpath(outdir, "response_rows.csv"), trace.rows)
    profiles_csv = write_csv(joinpath(outdir, "response_profiles.csv"), trace.profiles)
    response_png = plot_response(joinpath(outdir, "response_summary.png"), trace.rows, trace.profiles)
    return (; rows_csv, profiles_csv, response_png)
end

"""Return mean y-values grouped by graph kind and sweep for plotting."""
function grouped_mean(rows, field::String)
    groups = Dict{Tuple{String,Int},Vector{FT}}()
    for row in rows
        key = (String(row["graph_kind"]), Int(row["sweep"]))
        push!(get!(groups, key, FT[]), FT(row[field]))
    end
    kinds = sort!(unique(first.(keys(groups))))
    return kinds, groups
end

"""Plot average response curves and hidden-column response profiles."""
function plot_response(path, rows, profiles)
    fig = Figure(size = (1100, 760))
    ax1 = Axis(fig[1, 1], title = "output response after input switch", xlabel = "sweep", ylabel = "mean |Δ output|")
    ax2 = Axis(fig[1, 2], title = "response front", xlabel = "sweep", ylabel = "mean front column")
    ax3 = Axis(fig[2, 1], title = "right edge response", xlabel = "sweep", ylabel = "mean |Δ right edge|")
    ax4 = Axis(fig[2, 2], title = "hidden column response", xlabel = "column", ylabel = "mean |Δ hidden|")

    for field_ax in (("output_response", ax1), ("front_position", ax2), ("right_edge_response", ax3))
        field, ax = field_ax
        kinds, groups = grouped_mean(rows, field)
        for kind in kinds
            sweeps = sort([s for (k, s) in keys(groups) if k == kind])
            vals = [mean(abs, groups[(kind, sweep)]) for sweep in sweeps]
            lines!(ax, sweeps, vals, label = kind)
        end
        axislegend(ax, position = :rb)
    end

    profile_groups = Dict{Tuple{String,Int},Vector{FT}}()
    for row in profiles
        key = (String(row["graph_kind"]), Int(row["column"]))
        push!(get!(profile_groups, key, FT[]), FT(row["mean_abs_response"]))
    end
    for kind in sort!(unique(first.(keys(profile_groups))))
        cols = sort([c for (k, c) in keys(profile_groups) if k == kind])
        vals = [mean(profile_groups[(kind, col)]) for col in cols]
        lines!(ax4, cols, vals, label = kind)
    end
    axislegend(ax4, position = :rb)
    save(path, fig)
    return path
end

"""Write a concise README for one NN run."""
function write_run_readme(path, config::EdgeSignalXORConfig, trained, response)
    open(path, "w") do io
        println(io, "# Edge Signal XOR")
        println(io)
        println(io, "Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.")
        println(io, "Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.")
        println(io)
        println(io, "- hidden local NN: `$(config.hidden_nn)`")
        println(io, "- epochs/log_every: `$(config.epochs)` / `$(config.log_every)`")
        println(io, "- Minit/eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- free/nudged steps: `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- validation steps: `$(config.validation_relaxation)`")
        println(io, "- beta/lr/stepsize: `$(config.β)` / `$(config.lr)` / `$(config.stepsize)`")
        println(io, "- temperature fraction of max column interaction: `$(config.temp_fraction)`")
        println(io, "- response skipped: `$(config.skip_response)`")
        println(io)
        println(io, "Best logged learning result: epoch `$(trained.best.epoch)`, MSE `$(round(trained.best.mse, digits = 6))`, accuracy `$(trained.best.acc)`.")
        println(io)
        println(io, "Files:")
        println(io, "- `learning_metrics.csv`")
        println(io, "- `learning_progress.png`")
        if !config.skip_response
            println(io, "- `response_rows.csv`")
            println(io, "- `response_profiles.csv`")
            println(io, "- `response_summary.png`")
        end
        println(io, "- `initial_graph.jld2`")
        println(io, "- `best_graph.jld2`")
    end
    return path
end

"""Run one NN setting: train first, then compare initial vs learned responses."""
function run_one_nn(config::EdgeSignalXORConfig, rootdir)
    outdir = joinpath(rootdir, "NN$(config.hidden_nn)")
    mkpath(outdir)
    trained = train_edge_xor(config; outdir)
    response = config.skip_response ? nothing : measure_responses(config, trained; outdir)
    write_run_readme(joinpath(outdir, "README.md"), config, trained, response)
    return (; config, trained, response, outdir)
end

"""Parse comma-separated NN values from `EDGE_XOR_NNS`."""
function configured_nns(default_nn::Int)
    text = get(ENV, "EDGE_XOR_NNS", string(default_nn))
    return parse.(Int, split(text, ","))
end

"""Run the full edge-signal XOR experiment folder."""
function main()
    base_config = EdgeSignalXORConfig()
    rootdir = get(
        ENV,
        "EDGE_XOR_DIR",
        joinpath(@__DIR__, "runs", "edge_signal_xor_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(rootdir)
    results = []
    for nn in configured_nns(base_config.hidden_nn)
        config = EdgeSignalXORConfig(; hidden_nn = nn)
        push!(results, run_one_nn(config, rootdir))
    end
    summary_rows = [
        Dict{String,Any}(
            "nn" => result.config.hidden_nn,
            "best_epoch" => result.trained.best.epoch,
            "best_mse" => result.trained.best.mse,
            "best_accuracy" => result.trained.best.acc,
            "outdir" => result.outdir,
        ) for result in results
    ]
    write_csv(joinpath(rootdir, "summary.csv"), summary_rows)
    println("Saved edge signal XOR experiment: ", rootdir)
    return (; rootdir, results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
