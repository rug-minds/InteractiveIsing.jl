include(joinpath(@__DIR__, "latency_utils.jl"))

const ROWS = Dict{String,Any}[]
const TARGET = joinpath(@__DIR__, "..", "simple_timeavg_xor", "simple_2_4_1_timeavg_learning.jl")

measure!(ROWS, "include manager experiment") do
    include(TARGET)
end

"""Return the manager-backed timing profile configuration."""
function manager_profile_config()
    return TimeAverageXorConfig(
        epochs = parse(Int, get(ENV, "ISING_LATENCY_EPOCHS", "2")),
        log_every = 1,
        minit = parse(Int, get(ENV, "ISING_LATENCY_MINIT", "8")),
        eval_repeats = parse(Int, get(ENV, "ISING_LATENCY_EVAL_REPEATS", "16")),
        workers = parse(Int, get(ENV, "ISING_LATENCY_WORKERS", string(max(1, min(Threads.nthreads(), 8))))),
        free_relaxation = parse(Int, get(ENV, "ISING_LATENCY_FREE", "600")),
        nudged_relaxation = parse(Int, get(ENV, "ISING_LATENCY_NUDGED", "600")),
        β = parse(FT, get(ENV, "ISING_LATENCY_BETA", "2.0")),
        lr = parse(FT, get(ENV, "ISING_LATENCY_LR", "0.002")),
        weight_decay = parse(FT, get(ENV, "ISING_LATENCY_WEIGHT_DECAY", "1e-4")),
        grad_clip = parse(FT, get(ENV, "ISING_LATENCY_GRAD_CLIP", "50")),
        temp = parse(FT, get(ENV, "ISING_LATENCY_TEMP", "0.005")),
        stepsize = parse(FT, get(ENV, "ISING_LATENCY_STEPSIZE", "0.4")),
        max_drift_fraction = parse(FT, get(ENV, "ISING_LATENCY_MAX_DRIFT", "0.6")),
        eval_burnin_sweeps = parse(Int, get(ENV, "ISING_LATENCY_EVAL_BURNIN", "600")),
        eval_average_sweeps = parse(Int, get(ENV, "ISING_LATENCY_EVAL_AVG", "50")),
        eval_sample_every_sweeps = parse(Int, get(ENV, "ISING_LATENCY_EVAL_EVERY", "1")),
        outdir = latency_outdir("manager_timeavg_target_output"),
    )
end

"""Profile one manager-backed validation call in explicit subphases."""
function profile_manager_eval!(rows, trainer, x, y, config; seed_offset)
    jobs = measure!(rows, "manager eval: build jobs") do
        eval_jobs(x, config; seed_offset)
    end
    measure!(rows, "manager eval: sync/reinit workers") do
        sync_manager_workers!(trainer.eval_manager, trainer.params, (:dynamics,))
    end
    measure!(rows, "manager eval: allocate output buckets") do
        trainer.current_sample_outputs = [NamedTuple{(:mean, :std),Tuple{FT,FT}}[] for _ in axes(x, 2)]
    end
    measure!(rows, "manager eval: ProcessManager.run!") do
        StatefulAlgorithms.run!(trainer.eval_manager, jobs)
    end
    return measure!(rows, "manager eval: reduce metrics") do
        means = zeros(FT, size(x, 2))
        stds = zeros(FT, size(x, 2))
        for sample_idx in axes(x, 2)
            sample_means = [output.mean for output in trainer.current_sample_outputs[sample_idx]]
            means[sample_idx] = mean(sample_means)
            stds[sample_idx] = std(sample_means)
        end
        targets = vec(y)
        (;
            mse = mean(abs2, means .- targets),
            acc = mean(sign.(means) .== sign.(targets)),
            margin = minimum(abs.(means)),
            means,
            stds,
        )
    end
end

"""Profile one manager-backed training epoch in explicit subphases."""
function profile_manager_train_epoch!(rows, trainer, x, y, batch_gradient, epoch, config)
    measure!(rows, "manager train$(epoch): zero buffers") do
        IsingLearning.zero_buffer!(batch_gradient)
        reset_manager_buffers!(trainer.train_manager)
        trainer.current_batch_gradient = batch_gradient
        empty!(trainer.current_responses)
    end
    jobs = measure!(rows, "manager train$(epoch): build jobs") do
        train_jobs(x, y, config, epoch)
    end
    measure!(rows, "manager train$(epoch): ProcessManager.run!") do
        StatefulAlgorithms.run!(trainer.train_manager, jobs)
    end
    grad = measure!(rows, "manager train$(epoch): optimizer update") do
        ntraj = length(jobs)
        IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
        config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
        config.grad_clip > 0 && clamp!(batch_gradient.w, -config.grad_clip, config.grad_clip)
        config.grad_clip > 0 && clamp!(batch_gradient.b, -config.grad_clip, config.grad_clip)
        grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
        trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
        IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
        (; grad_norm, response_norm = isempty(trainer.current_responses) ? zero(FT) : mean(trainer.current_responses))
    end
    measure!(rows, "manager train$(epoch): sync/reinit workers") do
        sync_manager_workers!(trainer.train_manager, trainer.params, (:dynamics,))
    end
    return grad
end

"""Run and save a manager-backed latency characterization."""
function main()
    config = manager_profile_config()
    outdir = latency_outdir("manager_timeavg_latency")
    trainer = measure!(ROWS, "manager init_timeavg_trainer") do
        init_timeavg_trainer(config)
    end
    x, y = measure!(ROWS, "manager simple_dataset") do
        simple_dataset()
    end
    batch_gradient = measure!(ROWS, "manager gradient_buffer") do
        IsingLearning.gradient_buffer(trainer.prototype_graph)
    end
    profile_manager_eval!(ROWS, trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    profile_manager_train_epoch!(ROWS, trainer, x, y, batch_gradient, 1, config)

    warm_trainer = measure!(ROWS, "manager warm init_timeavg_trainer") do
        init_timeavg_trainer(config)
    end
    warm_batch_gradient = measure!(ROWS, "manager warm gradient_buffer") do
        IsingLearning.gradient_buffer(warm_trainer.prototype_graph)
    end
    profile_manager_eval!(ROWS, warm_trainer, x, y, config; seed_offset = config.base_seed + 60_000_000)
    profile_manager_train_epoch!(ROWS, warm_trainer, x, y, warm_batch_gradient, 101, config)

    csv_path = write_latency_csv(joinpath(outdir, "manager_timeavg_latency.csv"), ROWS)
    md_path = write_latency_md(
        joinpath(outdir, "README.md"),
        "Manager Time-Averaged XOR Latency",
        latency_settings(
            workers = config.workers,
            minit = config.minit,
            eval_repeats = config.eval_repeats,
            free_relaxation = config.free_relaxation,
            nudged_relaxation = config.nudged_relaxation,
            eval_burnin_sweeps = config.eval_burnin_sweeps,
            eval_average_sweeps = config.eval_average_sweeps,
            temp = config.temp,
            stepsize = config.stepsize,
            beta = config.β,
        ),
        ROWS,
        [
            "`ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.",
            "The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.",
            "Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.",
        ],
    )
    println("Wrote CSV: ", csv_path)
    println("Wrote report: ", md_path)
    return (; rows = ROWS, csv_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
