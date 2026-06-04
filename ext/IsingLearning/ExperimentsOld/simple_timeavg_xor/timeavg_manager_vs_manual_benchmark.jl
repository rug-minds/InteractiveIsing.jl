using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "simple_2_4_1_timeavg_learning.jl"))

using Statistics

"""
    BenchmarkConfig(; kwargs...)

Small timing configuration for comparing the persistent `ProcessManager` path
against a direct manual worker loop on the same 2->4->1 time-averaged XOR task.
"""
Base.@kwdef mutable struct BenchmarkConfig
    epochs::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_EPOCHS", "10"))
    workers::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_WORKERS", string(max(1, min(Threads.nthreads(), 8)))))
    minit::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_MINIT", "1"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_EVAL_REPEATS", "1"))
    free_relaxation::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_FREE", "10"))
    nudged_relaxation::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_NUDGED", "10"))
    eval_burnin_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_EVAL_BURNIN", "2"))
    eval_average_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_BENCH_EVAL_AVG", "5"))
    temp::FT = parse(FT, get(ENV, "ISING_TIMEAVG_BENCH_TEMP", "0.01"))
    stepsize::FT = parse(FT, get(ENV, "ISING_TIMEAVG_BENCH_STEPSIZE", "0.2"))
    outdir::String = get(
        ENV,
        "ISING_TIMEAVG_BENCH_DIR",
        joinpath(@__DIR__, "runs", "timeavg_benchmark_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

"""
    BenchmarkCache()

Reusable benchmark state for REPL/daemon-driven timing. The cache owns the
manager trainer, manual trainer, reusable gradient buffers, and dataset so
`main()` can be called repeatedly without rebuilding all workers every time.
"""
mutable struct BenchmarkCache
    manager_trainer::Any
    manual_trainer::Any
    manager_gradient::Any
    manual_gradient::Any
    x::Any
    y::Any
    signature::Any
end

BenchmarkCache() = BenchmarkCache(nothing, nothing, nothing, nothing, nothing, nothing, nothing)

CONFIG = isdefined(@__MODULE__, :CONFIG) ? CONFIG : BenchmarkConfig()
CACHE = isdefined(@__MODULE__, :CACHE) ? CACHE : BenchmarkCache()

"""
    timeavg_config(config)

Translate benchmark settings to the local time-averaged XOR experiment config.
"""
function timeavg_config(config::BenchmarkConfig)
    return TimeAverageXorConfig(
        epochs = config.epochs,
        log_every = config.epochs,
        minit = config.minit,
        eval_repeats = config.eval_repeats,
        workers = config.workers,
        free_relaxation = config.free_relaxation,
        nudged_relaxation = config.nudged_relaxation,
        eval_burnin_sweeps = config.eval_burnin_sweeps,
        eval_average_sweeps = config.eval_average_sweeps,
        temp = config.temp,
        stepsize = config.stepsize,
        outdir = config.outdir,
    )
end

"""
    benchmark_signature(config)

Return the subset of config fields that are baked into worker processes and
therefore require rebuilding cached trainers when changed.
"""
function benchmark_signature(config::BenchmarkConfig)
    return (;
        workers = config.workers,
        minit = config.minit,
        eval_repeats = config.eval_repeats,
        free_relaxation = config.free_relaxation,
        nudged_relaxation = config.nudged_relaxation,
        eval_burnin_sweeps = config.eval_burnin_sweeps,
        eval_average_sweeps = config.eval_average_sweeps,
        temp = config.temp,
        stepsize = config.stepsize,
    )
end

"""
    ensure_cache!(cache, config; rebuild=false)

Build or reuse the cached manager/manual trainers. Changing fields in
`benchmark_signature(config)` rebuilds automatically; changing `epochs` or
`outdir` does not.
"""
function ensure_cache!(cache::BenchmarkCache, config::BenchmarkConfig; rebuild::Bool=false)
    signature = benchmark_signature(config)
    if rebuild || cache.signature != signature || cache.manager_trainer === nothing || cache.manual_trainer === nothing
        tconfig = timeavg_config(config)
        x, y = simple_dataset()
        cache.manager_trainer = init_timeavg_trainer(tconfig)
        cache.manual_trainer = init_manual_trainer(tconfig)
        cache.manager_gradient = IsingLearning.gradient_buffer(cache.manager_trainer.prototype_graph)
        cache.manual_gradient = IsingLearning.gradient_buffer(cache.manual_trainer.prototype_graph)
        cache.x = x
        cache.y = y
        cache.signature = signature
    end
    return timeavg_config(config)
end

"""
    ManualTimeAvgTrainer

Trainer state for the manual baseline. It owns reusable worker `Process` objects
directly instead of routing jobs through `ProcessManager`.
"""
mutable struct ManualTimeAvgTrainer{Layer,Graph,Params,OptState,Optimiser,TW,EW}
    layer::Layer
    prototype_graph::Graph
    params::Params
    opt_state::OptState
    optimiser::Optimiser
    train_workers::TW
    eval_workers::EW
end

"""
    init_manual_trainer(config)

Create the same graph, parameters, optimizer, and worker processes used by the
manager experiment, but keep the worker vectors directly on the manual trainer.
"""
function init_manual_trainer(config::TimeAverageXorConfig)
    sc = simple_config(config)
    graph = simple_graph(sc)
    layer = SimpleLayer(graph, sc)
    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    train_workers = [template_train_worker(layer, graph, config, worker_idx) for worker_idx in 1:config.workers]
    eval_workers = [template_eval_worker(layer, graph, config, worker_idx) for worker_idx in 1:config.workers]
    return ManualTimeAvgTrainer(layer, graph, params, opt_state, optimiser, train_workers, eval_workers)
end

"""
    prepare_train_worker!(worker, trainer::ManualTimeAvgTrainer, config, job)

Prepare one manual training worker using the same context updates as the
manager-backed trainer.
"""
function prepare_train_worker!(worker::Process, trainer::ManualTimeAvgTrainer, config::TimeAverageXorConfig, job::TrainJob)
    StatefulAlgorithms.isdone(worker) && close(worker)
    sync_worker_params!(worker, trainer.params)
    seed_worker!(worker, job.seed)
    IsingLearning.zero_buffer!(StatefulAlgorithms.context(worker)._state.buffers)
    IsingLearning._write_example!(worker, job.x, job.y)
    StatefulAlgorithms.reset!(worker)
    return worker
end

"""
    prepare_eval_worker!(worker, trainer::ManualTimeAvgTrainer, config, job)

Prepare one manual validation worker and reset only its output-averager
subcontext.
"""
function prepare_eval_worker!(worker::Process, trainer::ManualTimeAvgTrainer, config::TimeAverageXorConfig, job::EvalJob)
    StatefulAlgorithms.isdone(worker) && close(worker)
    sync_worker_params!(worker, trainer.params)
    graph = StatefulAlgorithms.context(worker).dynamics.model
    II.temp!(graph, config.temp)
    Random.seed!(job.seed)
    simple_initstate!(graph, simple_config(config))
    IsingLearning.apply_input(graph, job.x)
    hasproperty(StatefulAlgorithms.context(worker).dynamics, :rng) && Random.seed!(StatefulAlgorithms.context(worker).dynamics.rng, job.seed + 1)
    StatefulAlgorithms.context(worker, StatefulAlgorithms.initcontext(StatefulAlgorithms.context(worker), :output_averager))
    StatefulAlgorithms.reset!(worker)
    return worker
end

"""
    run_worker_batch!(workers, jobs, prepare!, consume!)

Start a batch of reusable process workers, wait for all of them, then consume
their finalized contexts. This is the direct manual equivalent of one manager
slot batch.
"""
function run_worker_batch!(workers, jobs, prepare!, consume!)
    active = Process[]
    for (worker, job) in zip(workers, jobs)
        prepare!(worker, job)
        run(worker)
        push!(active, worker)
    end
    for (worker, job) in zip(active, jobs)
        wait(worker)
        close(worker)
        consume!(worker, job)
    end
    return nothing
end

"""
    manual_train_epoch!(trainer, x, y, batch_gradient, epoch, config)

Train one epoch with explicit worker batching and no `ProcessManager`.
"""
function manual_train_epoch!(trainer::ManualTimeAvgTrainer, x, y, batch_gradient, epoch::Integer, config::TimeAverageXorConfig)
    IsingLearning.zero_buffer!(batch_gradient)
    responses = FT[]
    jobs = train_jobs(x, y, config, epoch)
    nworkers = length(trainer.train_workers)
    for first_idx in 1:nworkers:length(jobs)
        batch = jobs[first_idx:min(first_idx + nworkers - 1, length(jobs))]
        run_worker_batch!(
            view(trainer.train_workers, 1:length(batch)),
            batch,
            (worker, job) -> prepare_train_worker!(worker, trainer, config, job),
            (worker, job) -> collect_train_result!(batch_gradient, responses, (; context = StatefulAlgorithms.context(worker), error = nothing)),
        )
    end
    ntraj = length(jobs)
    IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
    config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
    config.grad_clip > 0 && clamp!(batch_gradient.w, -config.grad_clip, config.grad_clip)
    config.grad_clip > 0 && clamp!(batch_gradient.b, -config.grad_clip, config.grad_clip)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    return (; grad_norm, response_norm = isempty(responses) ? zero(FT) : mean(responses))
end

"""
    manual_evaluate_timeavg!(trainer, x, y, config; seed_offset)

Evaluate time-averaged scalar outputs with explicit worker batching and no
`ProcessManager`.
"""
function manual_evaluate_timeavg!(trainer::ManualTimeAvgTrainer, x, y, config::TimeAverageXorConfig; seed_offset::Integer)
    jobs = eval_jobs(x, config; seed_offset)
    sample_outputs = [NamedTuple{(:mean, :std),Tuple{FT,FT}}[] for _ in axes(x, 2)]
    nworkers = length(trainer.eval_workers)
    for first_idx in 1:nworkers:length(jobs)
        batch = jobs[first_idx:min(first_idx + nworkers - 1, length(jobs))]
        run_worker_batch!(
            view(trainer.eval_workers, 1:length(batch)),
            batch,
            (worker, job) -> prepare_eval_worker!(worker, trainer, config, job),
            (worker, job) -> begin
                ctx = StatefulAlgorithms.context(worker).output_averager
                push!(sample_outputs[job.sample_idx], (;
                    mean = reusable_average_mean(ctx),
                    std = reusable_average_std(ctx),
                ))
            end,
        )
    end
    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        sample_means = [output.mean for output in sample_outputs[sample_idx]]
        means[sample_idx] = mean(sample_means)
        stds[sample_idx] = std(sample_means)
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

"""
    timed!(rows, route, phase, epoch) do

Run a block and append its elapsed time to the benchmark row list.
"""
function timed!(f, rows, route::String, phase::String, epoch::Integer)
    result_ref = Ref{Any}()
    seconds = @elapsed result_ref[] = f()
    push!(rows, Dict{String,Any}(
        "route" => route,
        "phase" => phase,
        "epoch" => epoch,
        "seconds" => seconds,
    ))
    return result_ref[]
end

"""
    benchmark_manager!(rows, config, x, y)

Run the persistent-manager route and record init, eval, and train timings.
"""
function benchmark_manager!(rows, config::TimeAverageXorConfig, x, y)
    trainer = timed!(rows, "manager", "init", 0) do
        init_timeavg_trainer(config)
    end
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    timed!(rows, "manager", "eval", 0) do
        evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    for epoch in 1:config.epochs
        timed!(rows, "manager", "train", epoch) do
            train_epoch_managed!(trainer, x, y, batch_gradient, epoch, config)
        end
    end
    timed!(rows, "manager", "eval", config.epochs) do
        evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    return trainer
end

"""
    benchmark_manual!(rows, config, x, y)

Run the direct manual worker route and record init, eval, and train timings.
"""
function benchmark_manual!(rows, config::TimeAverageXorConfig, x, y)
    trainer = timed!(rows, "manual", "init", 0) do
        init_manual_trainer(config)
    end
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    timed!(rows, "manual", "eval", 0) do
        manual_evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    for epoch in 1:config.epochs
        timed!(rows, "manual", "train", epoch) do
            manual_train_epoch!(trainer, x, y, batch_gradient, epoch, config)
        end
    end
    timed!(rows, "manual", "eval", config.epochs) do
        manual_evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    return trainer
end

"""
    run_cached_manager!(rows, cache, config)

Run eval/train/eval through the cached `ProcessManager` trainer. No managers,
workers, or contexts are rebuilt by this function.
"""
function run_cached_manager!(rows, cache::BenchmarkCache, config::TimeAverageXorConfig)
    trainer = cache.manager_trainer
    x, y = cache.x, cache.y
    timed!(rows, "manager_cached", "eval", 0) do
        evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    for epoch in 1:config.epochs
        timed!(rows, "manager_cached", "train", epoch) do
            train_epoch_managed!(trainer, x, y, cache.manager_gradient, epoch, config)
        end
    end
    timed!(rows, "manager_cached", "eval", config.epochs) do
        evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    return trainer
end

"""
    run_cached_manual!(rows, cache, config)

Run eval/train/eval through the cached manual worker trainer. This keeps the
direct-worker baseline in the same REPL-friendly shape as the manager route.
"""
function run_cached_manual!(rows, cache::BenchmarkCache, config::TimeAverageXorConfig)
    trainer = cache.manual_trainer
    x, y = cache.x, cache.y
    timed!(rows, "manual_cached", "eval", 0) do
        manual_evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    for epoch in 1:config.epochs
        timed!(rows, "manual_cached", "train", epoch) do
            manual_train_epoch!(trainer, x, y, cache.manual_gradient, epoch, config)
        end
    end
    timed!(rows, "manual_cached", "eval", config.epochs) do
        manual_evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    end
    return trainer
end

"""
    print_summary(rows)

Print compact totals for manager/manual timing comparison.
"""
function print_summary(rows)
    println("Timing summary:")
    for route in unique(row["route"] for row in rows)
        route_rows = [row for row in rows if row["route"] == route]
        total = sum((row["seconds"] for row in route_rows); init = 0.0)
        init = sum((row["seconds"] for row in route_rows if row["phase"] == "init"); init = 0.0)
        evals = sum((row["seconds"] for row in route_rows if row["phase"] == "eval"); init = 0.0)
        trains = [row["seconds"] for row in route_rows if row["phase"] == "train"]
        println(route, ": total=", round(total, digits = 4),
            " init=", round(init, digits = 4),
            " eval_total=", round(evals, digits = 4),
            " train_mean=", isempty(trains) ? "n/a" : string(round(mean(trains), digits = 4)),
            " train_min=", isempty(trains) ? "n/a" : string(round(minimum(trains), digits = 4)),
            " train_max=", isempty(trains) ? "n/a" : string(round(maximum(trains), digits = 4)))
    end
end

"""
    main(; config=CONFIG, cache=CACHE, rebuild=false, mode=:cached)

Benchmark by default through cached manager/manual trainers. In a REPL or
DaemonMode session, mutate `CONFIG` fields and call `main()` again; cached
workers are reused unless `rebuild=true` or a worker-baked config field changed.

Use `mode=:cold` to run the original rebuild-every-route timing, and
`mode=:both` to run both cached and cold timings.
"""
function main(; config::BenchmarkConfig=CONFIG, cache::BenchmarkCache=CACHE, rebuild::Bool=false, mode::Symbol=:cached)
    tconfig = timeavg_config(config)
    mkpath(config.outdir)
    rows = Dict{String,Any}[]

    if mode in (:cached, :both)
        timed!(rows, "cache", "init", 0) do
            ensure_cache!(cache, config; rebuild)
        end
        cached_config = timeavg_config(config)
        run_cached_manager!(rows, cache, cached_config)
        run_cached_manual!(rows, cache, cached_config)
    end

    if mode in (:cold, :both)
        x, y = simple_dataset()
        benchmark_manager!(rows, tconfig, x, y)
        benchmark_manual!(rows, tconfig, x, y)
    end

    csv_path = write_csv(joinpath(config.outdir, "manager_vs_manual_timings.csv"), rows)
    print_summary(rows)
    println("Saved timings: ", csv_path)
    return (; rows, csv_path, config, cache)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
