using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using SparseArrays
using Statistics
using Dates
using CairoMakie
using IsingLearning.InteractiveIsing.Processes

const FT = Float64
const II = IsingLearning.InteractiveIsing
const Processes = II.Processes

const CASES = ((false, false), (false, true), (true, false), (true, true))

"""
    SimpleXorConfig(; kwargs...)

Configuration for the controlled `2 -> 16 -> 2` XOR test. The graph has two
physical bipolar input spins, sixteen hidden spins, and two output spins.
"""
Base.@kwdef struct SimpleXorConfig
    name::String = "simple_2_16_2"
    epochs::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_EPOCHS", "2000"))
    log_every::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_LOG_EVERY", "250"))
    minit::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_MINIT", "4"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_EVAL_REPEATS", "32"))
    workers::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_THREADS", "1"))
    free_relaxation::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_FREE", "1000"))
    nudged_relaxation::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_NUDGED", "1000"))
    early_relaxation::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_EARLY", "20"))
    β::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_BETA", "1.0"))
    lr::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_LR", "0.01"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_WEIGHT_DECAY", "0.0"))
    grad_clip::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_GRAD_CLIP", "50"))
    temp::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_STEPSIZE", "0.10"))
    max_drift_fraction::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_MAX_DRIFT", "1.00"))
    dynamics_mode::Symbol = Symbol(get(ENV, "ISING_SIMPLE_XOR_DYNAMICS", "block"))
    block_size::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_BLOCK_SIZE", "8"))
    weight_scale::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_WEIGHT_SCALE", "0.20"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_SIMPLE_XOR_BIAS_SCALE", "0.05"))
    init_mode::Symbol = Symbol(get(ENV, "ISING_SIMPLE_XOR_INIT_MODE", "zero"))
    weight_seed::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_WEIGHT_SEED", "13"))
    bias_seed::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_BIAS_SEED", "23"))
    base_seed::Int = parse(Int, get(ENV, "ISING_SIMPLE_XOR_BASE_SEED", "87300"))
end

"""
    simple_initstate!(model, config)

Initialize one free relaxation for this experiment through the normal graph
reset path. This file intentionally uses random initial states only.
"""
function simple_initstate!(model, config::SimpleXorConfig)
    if config.init_mode === :random
        II.resetstate!(model)
    elseif config.init_mode === :zero
        fill!(II.state(model), zero(eltype(model)))
    else
        throw(ArgumentError("ISING_SIMPLE_XOR_INIT_MODE must be random or zero, got $(config.init_mode)"))
    end
    return model
end

"""
    SimpleLayer(graph, config)

Construct the `LayeredIsingGraphLayer` wrapper used by the trainer. The sampler
is always unadjusted `LocalLangevin`, because this test is about energy
minimization behavior rather than exact Boltzmann sampling.
"""
function SimpleLayer(graph, config::SimpleXorConfig)
    dynamics = simple_dynamics(config)
    return LayeredIsingGraphLayer(
        () -> simple_graph(config);
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

"""
    simple_dataset()

Return bipolar XOR inputs and two-spin bipolar class targets.
"""
function simple_dataset()
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 2, 4)
    for (col, (a, b)) in enumerate(CASES)
        x[:, col] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        y[:, col] .= xor(a, b) ? (-one(FT), one(FT)) : (one(FT), -one(FT))
    end
    return x, y
end

"""
    simple_graph(config)

Build a `2 -> 16 -> 2` continuous graph with all-to-all inter-layer weights,
trainable magnetic field, and masked direct output clamping. There is no
polynomial local potential.
"""
function simple_graph(config::SimpleXorConfig)
    layers = (
        II.Layer(2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 1, 0)),
        II.Layer(16, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 2, 0)),
        II.Layer(2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 3, 0)),
    )
    rng_w = Random.MersenneTwister(config.weight_seed)
    wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> config.weight_scale * randn(rng_w, FT))
    rng_b = Random.MersenneTwister(config.bias_seed)
    b = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    clamping_target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    clamping_mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = b) +
        II.Clamping(β = II.UniformArray(zero(FT)), y = clamping_target, mask = clamping_mask)
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
    simple_dynamics(config)

Return one `LocalLangevin` algorithm for a process branch.
"""
function simple_dynamics(config::SimpleXorConfig)
    config.dynamics_mode === :local && return II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    config.dynamics_mode === :block && return II.BlockLangevin(
        stepsize = config.stepsize,
        adjusted = false,
        block_size = config.block_size,
        group_steps = 1,
    )
    throw(ArgumentError("ISING_SIMPLE_XOR_DYNAMICS must be local or block, got $(config.dynamics_mode)"))
end

"""
    dynamics_input(name, graph, seed)

Create a branch context with its own graph and Langevin RNG.
"""
function dynamics_input(name::Symbol, graph, seed::Integer)
    return Init(name, model = graph)
end

"""
    simple_forward(layer, config; split = false)

Free-phase routine. In normal mode it records the final free endpoint as
`equilibrium_state`. In split mode it records an early free snapshot, then keeps
running the same free model to the late endpoint used by the gradient.
"""
function simple_forward(layer, config::SimpleXorConfig; split::Bool = false)
    n_units = layer.nunits
    early_steps = split ? clamp(config.early_relaxation, 1, layer.free_relaxation_steps) : layer.free_relaxation_steps
    late_steps = split ? max(0, layer.free_relaxation_steps - early_steps) : 0
    dynamics_algorithm = simple_dynamics(config)
    forward = @Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        simple_initstate!(dynamics.model, config)
        IsingLearning.apply_input(dynamics.model, x)
        @repeat early_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(m -> II.state(m), dynamics.model))
        @repeat late_steps dynamics()
    end
    return (; algorithm = forward, dynamics = forward.dynamics)
end

"""
    simple_nudged(layer, config)

Plus/minus nudged routines. Both restore from `equilibrium_state`, re-apply the
input and two-output target, then capture the final nudged states.
"""
function simple_nudged(layer, config::SimpleXorConfig)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    dynamics_algorithm = simple_dynamics(config)

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, beta)
        model = @repeat relaxation_steps dynamics()
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, -beta)
        model = @repeat relaxation_steps dynamics()
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""
    simple_forward_and_nudged(layer, config; split = false)

Full EqProp composite. The normal route nudges from the free endpoint; the
split route nudges from the early snapshot while still using the late free
model in `contrastive_gradient`.
"""
function simple_forward_and_nudged(layer, config::SimpleXorConfig; split::Bool = false)
    forward = simple_forward(layer, config; split).algorithm
    nudged = simple_nudged(layer, config)
    beta = layer.β
    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = nudged.algorithm()
        IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

"""
    simple_worker_process(layer, graph, config; split)

Create one training worker using either the normal or split-snapshot composite.
"""
function simple_worker_process(layer, graph, config::SimpleXorConfig; split::Bool)
    algo = Processes.resolve(simple_forward_and_nudged(layer, config; split).algorithm)
    buffers = IsingLearning.gradient_buffer(graph)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, length(layer.output_layer)),
            buffers = buffers,
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed),
        Init(:plus_capture, state = graph),
        Init(:minus_capture, state = graph);
        repeat = 1,
    )
end

"""
    simple_validation_process(layer, graph, config)

Create a free-relaxation worker used for repeated output statistics.
"""
function simple_validation_process(layer, graph, config::SimpleXorConfig)
    algo = Processes.resolve(simple_forward(layer, config; split = false).algorithm)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed + 50_000);
        repeat = 1,
    )
end

"""
    init_simple_trainer(layer, config; graph, split)

Initialize the threaded trainer object used by the compact training loop.
"""
function init_simple_trainer(layer, config::SimpleXorConfig; graph = layer.model_graph, split::Bool)
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), params)
    workers = Process[]
    worker_graphs = typeof(graph)[]
    for worker_idx in 1:config.workers
        wg = IsingLearning._worker_graph(graph, params)
        II.temp!(wg, config.temp)
        push!(worker_graphs, wg)
        push!(workers, simple_worker_process(layer, wg, config; split))
    end
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, config.temp)
    validation_worker = simple_validation_process(layer, validation_graph, config)
    return IsingLearning.MNISTThreadedTrainer(
        layer, graph, params, opt_state, worker_graphs, workers,
        validation_graph, validation_worker, Optimisers.Adam(config.lr),
    )
end

"""
    seed_worker!(worker, seed)

Seed all Langevin branch RNGs visible in a worker context.
"""
function seed_worker!(worker, seed::Integer)
    Random.seed!(seed)
    for (offset, name) in enumerate((:dynamics,))
        hasproperty(Processes.context(worker), name) || continue
        context = getproperty(Processes.context(worker), name)
        hasproperty(context, :rng) && Random.seed!(context.rng, seed + 10_000 * offset)
    end
    return worker
end

"""
    start_training_worker!(worker, x, y; seed)

Start one training trajectory through the Process scheduler. The caller should
collect it with `finish_training_worker!`; this lets a batch of workers run in
parallel instead of starting and waiting for each trajectory serially.
"""
function start_training_worker!(worker, x, y; seed::Integer)
    Processes.isdone(worker) && close(worker)
    seed_worker!(worker, seed)
    IsingLearning.zero_buffer!(Processes.context(worker)._state.buffers)
    IsingLearning._write_example!(worker, x, y)
    Processes.reset!(worker)
    run(worker)
    return worker
end

"""
    finish_training_worker!(worker, batch_gradient, responses)

Wait for a started worker, merge its context, add its gradient buffer into the
epoch buffer, and record the plus/minus response norm.
"""
function finish_training_worker!(worker, batch_gradient, responses)
    wait(worker)
    close(worker)
    free_state = Processes.context(worker)._state.equilibrium_state
    plus_state = Processes.context(worker).plus_capture.captured
    minus_state = Processes.context(worker).minus_capture.captured
    response = (
        sqrt(sum(abs2, plus_state .- free_state) / FT(length(free_state))) +
        sqrt(sum(abs2, minus_state .- free_state) / FT(length(free_state)))
    ) / 2
    push!(responses, response)
    IsingLearning.add_buffer!(batch_gradient, Processes.context(worker)._state.buffers)
    return worker
end

"""
    train_epoch!(trainer, x, y, batch_gradient, epoch, config)

Average EqProp gradients over all samples and repeated random initial states,
then apply Adam with optional weight decay and clipping.
"""
function train_epoch!(trainer, x, y, batch_gradient, epoch::Integer, config::SimpleXorConfig)
    IsingLearning.zero_buffer!(batch_gradient)
    responses = FT[]
    running = Process[]
    ntraj = 0
    for sample_idx in axes(x, 2), init_idx in 1:config.minit
        worker_idx = mod1(ntraj + 1, length(trainer.workers))
        worker = trainer.workers[worker_idx]
        seed = config.base_seed + 1_000_000 * epoch + 10_000 * sample_idx + init_idx
        start_training_worker!(worker, view(x, :, sample_idx), view(y, :, sample_idx); seed)
        push!(running, worker)
        ntraj += 1
        if length(running) == length(trainer.workers)
            for task_worker in running
                finish_training_worker!(task_worker, batch_gradient, responses)
            end
            empty!(running)
        end
    end
    for task_worker in running
        finish_training_worker!(task_worker, batch_gradient, responses)
    end
    IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
    config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
    config.grad_clip > 0 && clamp!(batch_gradient.w, -config.grad_clip, config.grad_clip)
    config.grad_clip > 0 && clamp!(batch_gradient.b, -config.grad_clip, config.grad_clip)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    return (; grad_norm, response_norm = isempty(responses) ? zero(FT) : mean(responses))
end

"""
    validation_output!(trainer, x; seed)

Run one validation free relaxation and return the two-spin output vector.
"""
function validation_output!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    Processes.isdone(worker) && close(worker)
    seed_worker!(worker, seed)
    IsingLearning._write_input!(worker, x)
    Processes.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    return copy(IsingLearning._validation_output(trainer))
end

"""
    evaluate_simple!(trainer, x, y, config; seed_offset)

Evaluate mean output statistics over repeated random initial states.
"""
function evaluate_simple!(trainer, x, y, config::SimpleXorConfig; seed_offset::Integer)
    outputs = [zeros(FT, size(y, 1)) for _ in axes(x, 2)]
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = [zeros(FT, size(y, 1)) for _ in 1:config.eval_repeats]
        mean_output = zeros(FT, size(y, 1))
        for repeat_idx in 1:config.eval_repeats
            samples[repeat_idx] = validation_output!(
                trainer,
                view(x, :, sample_idx);
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
            )
            mean_output .+= samples[repeat_idx]
        end
        mean_output ./= FT(config.eval_repeats)
        outputs[sample_idx] .= mean_output
        stds[sample_idx] = config.eval_repeats <= 1 ? zero(FT) :
            sqrt(sum(sum(abs2, sample .- mean_output) / FT(length(mean_output)) for sample in samples) / FT(config.eval_repeats - 1))
    end
    mse = mean(sum(abs2, outputs[i] .- view(y, :, i)) / FT(size(y, 1)) for i in axes(y, 2))
    acc = mean(argmax(outputs[i]) == argmax(view(y, :, i)) for i in axes(y, 2))
    margin = minimum(abs(outputs[i][2] - outputs[i][1]) for i in eachindex(outputs))
    return (; mse, acc, margin, means = outputs, stds)
end

"""
    metric_row(route, epoch, metrics, grad)

Convert one logged point into a CSV/plot row.
"""
function metric_row(route, epoch, metrics, grad)
    row = Dict{String,Any}(
        "route" => route,
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.acc,
        "margin" => metrics.margin,
        "grad_norm" => grad.grad_norm,
        "response_norm" => grad.response_norm,
    )
    for i in eachindex(metrics.means)
        row["mean_$(i)_1"] = metrics.means[i][1]
        row["mean_$(i)_2"] = metrics.means[i][2]
        row["std_$i"] = metrics.stds[i]
    end
    return row
end

"""Return rounded output means for readable progress logging."""
format_means(metrics) = [round.(m; digits = 3) for m in metrics.means]

"""
    write_csv(path, rows)

Write logged rows without pulling in a CSV dependency.
"""
function write_csv(path, rows)
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

"""
    plot_rows(path, rows)

Save a compact comparison plot for normal and split-snapshot routes.
"""
function plot_rows(path, rows)
    fig = Figure(size = (1100, 850))
    axes = (
        Axis(fig[1, 1], title = "2->4->1 XOR MSE", xlabel = "epoch", ylabel = "MSE"),
        Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy"),
        Axis(fig[2, 1], title = "output margin", xlabel = "epoch", ylabel = "min |output|"),
        Axis(fig[2, 2], title = "gradient norm", xlabel = "epoch", ylabel = "||grad||"),
    )
    routes = unique(row["route"] for row in rows)
    for route in routes
        subset = [row for row in rows if row["route"] == route]
        epochs = [row["epoch"] for row in subset]
        lines!(axes[1], epochs, [row["mse"] for row in subset], label = route)
        lines!(axes[2], epochs, [row["accuracy"] for row in subset], label = route)
        lines!(axes[3], epochs, [row["margin"] for row in subset], label = route)
        lines!(axes[4], epochs, [row["grad_norm"] for row in subset], label = route)
    end
    axislegend(axes[1], position = :rt)
    save(path, fig)
    return path
end

"""
    close_trainer!(trainer)

Close any active worker tasks owned by a trainer.
"""
function close_trainer!(trainer)
    for worker in trainer.workers
        isnothing(worker.task) || close(worker)
    end
    isnothing(trainer.validation_worker.task) || close(trainer.validation_worker)
    return trainer
end

"""
    run_route(config, route; split)

Train one route and return logged metrics and best score.
"""
function run_route(config::SimpleXorConfig, route::String; split::Bool)
    graph = simple_graph(config)
    layer = SimpleLayer(graph, config)
    trainer = init_simple_trainer(layer, config; graph, split)
    x, y = simple_dataset()
    batch_gradient = IsingLearning.gradient_buffer(graph)
    rows = Dict{String,Any}[]
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))
    best_mse = Inf
    best_acc = zero(FT)

    metrics = evaluate_simple!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    push!(rows, metric_row(route, 0, metrics, zero_grad))
    println(route, " epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", format_means(metrics))

    for epoch in 1:config.epochs
        grad = train_epoch!(trainer, x, y, batch_gradient, epoch, config)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_simple!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
            push!(rows, metric_row(route, epoch, metrics, grad))
            if metrics.acc > best_acc || (metrics.acc == best_acc && metrics.mse < best_mse)
                best_acc = metrics.acc
                best_mse = metrics.mse
            end
            println(route, " epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad.grad_norm, digits = 4), " means=", format_means(metrics))
        end
    end
    close_trainer!(trainer)
    return (; rows, best_mse, best_acc)
end

"""
    write_readme(path, config, results, csv_path, png_path)

Document the exact controlled comparison.
"""
function write_readme(path, config::SimpleXorConfig, results, csv_path, png_path)
    open(path, "w") do io
        println(io, "# Simple 2->16->2 Langevin XOR")
        println(io)
        println(io, "This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one two-output spin.")
        println(io)
        println(io, "The default route uses unadjusted `BlockLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.")
        println(io)
        println(io, "- Temperature: `$(config.temp)`")
        println(io, "- Dynamics: `$(config.dynamics_mode)`")
        println(io, "- Stepsize: `$(config.stepsize)`")
        println(io, "- Block size: `$(config.block_size)`")
        println(io, "- Max drift fraction: `$(config.max_drift_fraction)`")
        println(io, "- Free / early / nudged: `$(config.free_relaxation)` / `$(config.early_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- Minit / eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- Init mode: `$(config.init_mode)`")
        println(io)
        println(io, "| Route | Best MSE | Best Acc |")
        println(io, "|---|---:|---:|")
        for result in results
            println(io, "| `$(result.route)` | $(round(result.best_mse, digits = 6)) | $(round(result.best_acc, digits = 3)) |")
        end
        println(io)
        println(io, "CSV: `$(basename(csv_path))`")
        println(io, "Plot: `$(basename(png_path))`")
    end
    return path
end

"""
    main()

Run the normal two-output Langevin XOR route. The split route remains available
through `ISING_SIMPLE_XOR_ROUTE=split` for diagnostics.
"""
function main()
    config = SimpleXorConfig()
    outdir = get(
        ENV,
        "ISING_SIMPLE_XOR_DIR",
        joinpath(@__DIR__, "runs", "simple_2_16_2_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    route_mode = Symbol(get(ENV, "ISING_SIMPLE_XOR_ROUTE", "normal"))
    results = []
    normal = nothing
    split = nothing
    if route_mode in (:both, :normal)
        normal = run_route(config, "normal"; split = false)
        append!(all_rows, normal.rows)
        push!(results, (route = "normal", best_mse = normal.best_mse, best_acc = normal.best_acc))
    end
    if route_mode in (:both, :split)
        split = run_route(config, "split"; split = true)
        append!(all_rows, split.rows)
        push!(results, (route = "split", best_mse = split.best_mse, best_acc = split.best_acc))
    end
    route_mode in (:both, :normal, :split) || throw(ArgumentError("ISING_SIMPLE_XOR_ROUTE must be both, normal, or split"))
    csv_path = write_csv(joinpath(outdir, "simple_2_16_2_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "simple_2_16_2_progress.png"), all_rows)
    md_path = write_readme(
        joinpath(outdir, "README.md"),
        config,
        results,
        csv_path,
        png_path,
    )
    println("Saved metrics: ", csv_path)
    println("Saved plot: ", png_path)
    println("Saved docs: ", md_path)
    return (; outdir, normal, split, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
