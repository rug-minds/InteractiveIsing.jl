include("edge_signal_split_input_core.jl")

"""Accumulate a scalar output average while a validation trajectory runs."""
struct EdgeOutputAverager <: Processes.ProcessAlgorithm
    output_idx::Int
    burnin_sweeps::Int
    average_sweeps::Int
end

Processes.init(::EdgeOutputAverager, context) =
    (; sum = zero(FT), sumsq = zero(FT), count = 0, seen_sweeps = 0)

"""Sample the output spin once per scheduled full sweep after burn-in."""
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
    timeavg_scalar_output!(trainer, x, config; seed, burnin_sweeps, average_sweeps)

Run one validation trajectory and classify from the time-averaged output spin.
The input remains frozen while all non-input spins evolve.
"""
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

"""Evaluate all four XOR cases with repeated time-averaged validation runs."""
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

"""Train one split-edge config while selecting checkpoints by time-averaged validation."""
function train_split_input_timeavg_xor(
    config::EdgeSignalXORConfig;
    outdir,
    nudged_temp_factor::FT,
    eval_burnin_sweeps::Int,
    eval_average_sweeps::Int,
)
    trainer = nudged_temp_factor == one(FT) ? split_input_edge_trainer(config) :
        split_input_nudged_temp_trainer(config, nudged_temp_factor)
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
            if metrics.acc == 1.0 && metrics.mse < 0.12
                println("early success at epoch ", epoch)
                break
            end
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

"""Return the compact grid for the current split-edge time-average search."""
function split_input_timeavg_specs()
    common = (;
        epochs = 5000,
        log_every = 500,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 30,
        nudged_sweeps = 50,
        validation_sweeps = 100,
        hidden_nn = 5,
        input_hidden_scale = FT(0.55),
        hidden_local_scale = FT(0.003),
        hidden_output_scale = FT(0.55),
        weight_decay = FT(0.004),
    )
    raw = [
        (; name = "b0p35_T0p014_eta0p8_lr3e4", β = FT(0.35), temp_fraction = FT(0.014), stepsize = FT(0.8), lr = FT(0.00030), bump = FT(1.15), seed = 1_180_001),
        (; name = "b0p50_T0p014_eta0p8_lr25e4", β = FT(0.50), temp_fraction = FT(0.014), stepsize = FT(0.8), lr = FT(0.00025), bump = FT(1.20), seed = 1_180_101),
        (; name = "b0p35_T0p010_eta1p0_lr25e4", β = FT(0.35), temp_fraction = FT(0.010), stepsize = FT(1.0), lr = FT(0.00025), bump = FT(1.25), seed = 1_180_201),
        (; name = "b0p25_T0p010_eta1p0_lr35e4", β = FT(0.25), temp_fraction = FT(0.010), stepsize = FT(1.0), lr = FT(0.00035), bump = FT(1.15), seed = 1_180_301),
    ]
    specs = []
    for spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            temp_fraction = spec.temp_fraction,
            stepsize = spec.stepsize,
            lr = spec.lr,
            base_seed = spec.seed,
        )
        push!(specs, (; spec.name, config = cfg.config, active_count = cfg.active_count, nudged_temp_factor = spec.bump))
    end
    return specs
end

"""Run the split-edge time-averaged validation search."""
function run_split_input_timeavg_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_timeavg_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 60
    eval_average_sweeps = 60
    for spec in split_input_timeavg_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_split_input_timeavg_xor(
            spec.config;
            outdir,
            nudged_temp_factor = spec.nudged_temp_factor,
            eval_burnin_sweeps,
            eval_average_sweeps,
        )
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => spec.name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "beta" => spec.config.β,
            "temp_fraction" => spec.config.temp_fraction,
            "stepsize" => spec.config.stepsize,
            "lr" => spec.config.lr,
            "weight_decay" => spec.config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
        if trained.best.acc == 1.0 && trained.best.mse < 0.12
            println("Stopping grid after successful setting: ", spec.name)
            break
        end
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved split-input time-average search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_timeavg_search()
end
