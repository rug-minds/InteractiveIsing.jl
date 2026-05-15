include("edge_signal_split_input_core.jl")

const LAYOUT_INPUT_SCALE_REF = Ref{FT}(0.75)

"""
    EdgeOutputAverager(output_idx, burnin_sweeps, average_sweeps)

Process algorithm used only for validation in this file. It samples one scalar
output spin once per full sweep after an initial burn-in.
"""
struct EdgeOutputAverager <: Processes.ProcessAlgorithm
    output_idx::Int
    burnin_sweeps::Int
    average_sweeps::Int
end

Processes.init(::EdgeOutputAverager, context) =
    (; sum = zero(FT), sumsq = zero(FT), count = 0, seen_sweeps = 0)

"""Update the scalar output average after each scheduled full sweep."""
function Processes.step!(averager::EdgeOutputAverager, context)
    seen = context.seen_sweeps + 1
    sum = context.sum
    sumsq = context.sumsq
    count = context.count
    if seen > averager.burnin_sweeps && count < averager.average_sweeps
        value = FT(II.state(context.model)[averager.output_idx])
        sum += value
        sumsq += value^2
        count += 1
    end
    return (; sum, sumsq, count, seen_sweeps = seen)
end

"""Return the mean stored in an `EdgeOutputAverager` subcontext."""
edge_average_mean(ctx) = ctx.count == 0 ? FT(NaN) : ctx.sum / ctx.count

"""Return the standard deviation stored in an `EdgeOutputAverager` subcontext."""
function edge_average_std(ctx)
    ctx.count <= 1 && return zero(FT)
    μ = edge_average_mean(ctx)
    return sqrt(max(zero(FT), ctx.sumsq / ctx.count - μ^2))
end

"""
    half_edge_input_weight(; c1, c2)

Weight function for the half-edge input layout. Input spin 1 drives rows
`1:N/2`; input spin 2 drives rows `N/2+1:N` on the hidden layer's left edge.
"""
function half_edge_input_weight(; c1, c2)
    left_hidden_edge(c2) || return zero(FT)
    top_half = c2[1] <= HIDDEN_HEIGHT_REF[] ÷ 2
    input_row = c1[1]
    if (input_row == 1 && top_half) || (input_row == 2 && !top_half)
        return LAYOUT_INPUT_SCALE_REF[]
    end
    return zero(FT)
end

"""
    interlaced_edge_input_weight(; c1, c2)

Weight function for the interlaced edge layout. Input spin 1 drives odd hidden
edge rows; input spin 2 drives even hidden edge rows.
"""
function interlaced_edge_input_weight(; c1, c2)
    left_hidden_edge(c2) || return zero(FT)
    row_is_odd = isodd(c2[1])
    input_row = c1[1]
    if (input_row == 1 && row_is_odd) || (input_row == 2 && !row_is_odd)
        return LAYOUT_INPUT_SCALE_REF[]
    end
    return zero(FT)
end

"""Create the deterministic input-to-edge generator for `layout`."""
function layout_input_generator(layout::Symbol, scale::FT)
    LAYOUT_INPUT_SCALE_REF[] = scale
    if layout === :half
        return @WG half_edge_input_weight NN = :all symmetric = true
    elseif layout === :interlaced
        return @WG interlaced_edge_input_weight NN = :all symmetric = true
    else
        error("unknown edge input layout $(layout)")
    end
end

"""Build a `2 -> 8x8 -> 1` edge graph with the requested input layout."""
function layout_edge_graph(config::EdgeSignalXORConfig, layout::Symbol)
    HIDDEN_HEIGHT_REF[] = config.hidden_height
    HIDDEN_WIDTH_REF[] = config.hidden_width
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

    input_hidden = layout_input_generator(layout, config.input_hidden_scale)
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

"""Wrap a layout-specific edge graph in the existing learning-layer interface."""
function layout_edge_layer(graph, config::EdgeSignalXORConfig, layout::Symbol)
    dynamics = edge_sampler(config)
    return LayeredIsingGraphLayer(
        () -> layout_edge_graph(config, layout);
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

"""Initialize a trainer for one edge input layout."""
function layout_edge_trainer(config::EdgeSignalXORConfig, layout::Symbol, nudged_temp_factor::FT)
    graph = layout_edge_graph(config, layout)
    layer = layout_edge_layer(graph, config, layout)
    if nudged_temp_factor == one(FT)
        return init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(config.lr))
    end

    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    base_temp = II.temp(graph)
    nudged_temp = nudged_temp_factor * base_temp
    worker_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(worker_graph, base_temp)
    worker = split_input_nudged_temp_worker_process(layer, worker_graph, base_temp, nudged_temp)
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, base_temp)
    validation_worker = IsingLearning._validation_process(layer, validation_graph)

    return IsingLearning.MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        [worker_graph],
        [worker],
        validation_graph,
        validation_worker,
        optimiser,
    )
end

"""Run one time-averaged validation trajectory for one XOR input."""
function timeavg_scalar_output!(trainer, x, config::EdgeSignalXORConfig; seed::Integer, burnin_sweeps::Integer, average_sweeps::Integer)
    graph = trainer.validation_graph
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)

    sampler = edge_sampler(config)
    dynamics = deepcopy(sampler)
    output_idx = only(II.layerrange(graph[end]))
    averager = EdgeOutputAverager(output_idx, Int(burnin_sweeps), Int(average_sweeps))
    sweep_steps = length(II.sampling_indices(graph.index_set))
    total_sweeps = burnin_sweeps + average_sweeps

    routine = Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @alias averager = averager
        @every sweep_steps averager(model = dynamics.model)
    end
    wrapped = Processes.@Routine begin
        @repeat (total_sweeps * sweep_steps) routine()
    end
    inputs = II._merge_graph_inputs(wrapped, graph, Processes.Init(dynamics, rng = Random.MersenneTwister(seed)))
    process = Processes.Process(Processes.resolve(wrapped), inputs...; repeats = 1)
    run(process)
    wait(process)
    ctx = Processes.context(process).averager
    μ = edge_average_mean(ctx)
    σ = edge_average_std(ctx)
    close(process)
    return μ, σ
end

"""Evaluate all XOR cases with repeated time-averaged validation runs."""
function evaluate_timeavg!(trainer, x, y, config::EdgeSignalXORConfig; seed_offset::Integer, burnin_sweeps::Integer, average_sweeps::Integer)
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = zeros(FT, config.eval_repeats)
        for repeat_idx in 1:config.eval_repeats
            μ, _ = timeavg_scalar_output!(
                trainer,
                view(x, :, sample_idx),
                config;
                seed = seed_offset + 10_000 * sample_idx + repeat_idx,
                burnin_sweeps,
                average_sweeps,
            )
            samples[repeat_idx] = μ
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

"""Train one layout-specific graph and select checkpoints by time-averaged validation."""
function train_edge_input_layout_xor(
    config::EdgeSignalXORConfig;
    layout::Symbol,
    outdir,
    nudged_temp_factor::FT,
    eval_burnin_sweeps::Int,
    eval_average_sweeps::Int,
)
    trainer = layout_edge_trainer(config, layout, nudged_temp_factor)
    x, y = xor_dataset(config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_timeavg!(
        trainer,
        x,
        y,
        config;
        seed_offset = config.base_seed + 30_000_000,
        burnin_sweeps = eval_burnin_sweeps,
        average_sweeps = eval_average_sweeps,
    )
    push_learning_row!(rows, 0, metrics, zero(FT), II.temp(trainer.prototype_graph))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    initial_params = deepcopy(trainer.params)
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
        " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        if config.weight_decay > 0
            trainer.params.w .*= (one(FT) - config.lr * config.weight_decay)
            IsingLearning._broadcast_params!(trainer)
        end
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_timeavg!(
                trainer,
                x,
                y,
                config;
                seed_offset = config.base_seed + 30_000_000,
                burnin_sweeps = eval_burnin_sweeps,
                average_sweeps = eval_average_sweeps,
            )
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_learning_row!(rows, epoch, metrics, grad_norm, II.temp(trainer.prototype_graph))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4),
                " means=", round.(metrics.means, digits = 3))
        end
    end

    learning_csv = write_csv(joinpath(outdir, "learning_metrics.csv"), rows)
    learning_png = plot_learning(joinpath(outdir, "learning_progress.png"), rows)
    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    best_graph_path = II.save_isinggraph(joinpath(outdir, "best_graph.jld2"), strip_weight_generators!(deepcopy(trainer.prototype_graph)))
    trainer.params = initial_params
    IsingLearning._broadcast_params!(trainer)
    initial_graph_path = II.save_isinggraph(joinpath(outdir, "initial_graph.jld2"), strip_weight_generators!(deepcopy(trainer.prototype_graph)))
    close_trainer!(trainer)
    return (; best, rows, learning_csv, learning_png, best_graph_path, initial_graph_path, best_params, initial_params)
end

"""
    edge_input_layout_grid_specs()

Return a compact grid that compares the half-edge and interlaced input layouts
under identical training settings.
"""
function edge_input_layout_grid_specs()
    common = (;
        epochs = 3500,
        log_every = 500,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 30,
        nudged_sweeps = 80,
        validation_sweeps = 140,
        hidden_nn = 5,
        bias_scale = FT(0.10),
    )
    raw = [
        (; tag = "b0p50_T0p010_eta1p0_lr20e4_wd5e3", β = FT(0.50), temp_fraction = FT(0.010), stepsize = FT(1.0), input_scale = FT(0.75), local_scale = FT(0.012), output_scale = FT(0.50), lr = FT(0.00020), wd = FT(0.005), bump = FT(1.30), seed = 1_330_001),
        (; tag = "b0p35_T0p010_eta1p0_lr25e4_wd4e3", β = FT(0.35), temp_fraction = FT(0.010), stepsize = FT(1.0), input_scale = FT(0.75), local_scale = FT(0.012), output_scale = FT(0.50), lr = FT(0.00025), wd = FT(0.004), bump = FT(1.25), seed = 1_330_101),
        (; tag = "b0p50_T0p007_eta1p2_lr15e4_wd5e3", β = FT(0.50), temp_fraction = FT(0.007), stepsize = FT(1.2), input_scale = FT(0.75), local_scale = FT(0.012), output_scale = FT(0.50), lr = FT(0.00015), wd = FT(0.005), bump = FT(1.35), seed = 1_330_201),
        (; tag = "b0p35_T0p012_eta0p8_lr25e4_wd3e3", β = FT(0.35), temp_fraction = FT(0.012), stepsize = FT(0.8), input_scale = FT(0.90), local_scale = FT(0.010), output_scale = FT(0.60), lr = FT(0.00025), wd = FT(0.003), bump = FT(1.20), seed = 1_330_301),
    ]
    specs = []
    for layout in (:half, :interlaced), spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            temp_fraction = spec.temp_fraction,
            stepsize = spec.stepsize,
            input_hidden_scale = spec.input_scale,
            hidden_local_scale = spec.local_scale,
            hidden_output_scale = spec.output_scale,
            lr = spec.lr,
            weight_decay = spec.wd,
            base_seed = spec.seed + (layout === :interlaced ? 20_000 : 0),
        )
        push!(specs, (;
            name = string(layout, "_", spec.tag),
            layout,
            config = cfg.config,
            active_count = cfg.active_count,
            nudged_temp_factor = spec.bump,
        ))
    end
    return specs
end

"""Run the half-edge versus interlaced-edge grid and write `summary.csv`."""
function run_edge_input_layout_grid(; rootdir = joinpath(@__DIR__, "runs", "edge_input_layout_grid_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 100
    eval_average_sweeps = 100
    for spec in edge_input_layout_grid_specs()
        println("\n=== ", spec.name, " ===")
        println("layout=", spec.layout,
            " active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_edge_input_layout_xor(
            spec.config;
            layout = spec.layout,
            outdir,
            nudged_temp_factor = spec.nudged_temp_factor,
            eval_burnin_sweeps,
            eval_average_sweeps,
        )
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => spec.name,
            "layout" => String(spec.layout),
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "beta" => spec.config.β,
            "temp_fraction" => spec.config.temp_fraction,
            "stepsize" => spec.config.stepsize,
            "input_hidden_scale" => spec.config.input_hidden_scale,
            "hidden_local_scale" => spec.config.hidden_local_scale,
            "hidden_output_scale" => spec.config.hidden_output_scale,
            "lr" => spec.config.lr,
            "weight_decay" => spec.config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "summary.csv"), rows)
    end
    println("Saved edge input layout grid: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_edge_input_layout_grid()
end
