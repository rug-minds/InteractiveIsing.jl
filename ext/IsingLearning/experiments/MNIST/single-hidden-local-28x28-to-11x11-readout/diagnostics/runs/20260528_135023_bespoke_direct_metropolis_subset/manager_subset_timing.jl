using Dates

const MANAGER_T0 = time()
const MANAGER_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const MANAGER_FILE = joinpath(MANAGER_ARCH, "mnist_local_manager_grid.jl")
const MANAGER_OUTDIR = @__DIR__

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "manager_metropolis_subset"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = get(ENV, "ISING_MNIST_PM_WORKERS", "32")
ENV["ISING_MNIST_PM_EPOCHS"] = "1"
ENV["ISING_MNIST_PM_BATCHSIZE"] = get(ENV, "ISING_MNIST_PM_BATCHSIZE", "32")
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")
ENV["ISING_MNIST_PM_OPTIMIZER"] = "adam"
ENV["ISING_MNIST_PM_LR_W0"] = get(ENV, "ISING_MNIST_PM_LR_W0", "0.004")
ENV["ISING_MNIST_PM_LR_W12"] = get(ENV, "ISING_MNIST_PM_LR_W12", "0.004")
ENV["ISING_MNIST_PM_LR_W2O"] = get(ENV, "ISING_MNIST_PM_LR_W2O", "0.004")
ENV["ISING_MNIST_PM_LR_B"] = get(ENV, "ISING_MNIST_PM_LR_B", "0.0004")
ENV["ISING_MNIST_PM_GRADIENT_NORMALIZATION"] = get(ENV, "ISING_MNIST_PM_GRADIENT_NORMALIZATION", "mean")

include(MANAGER_FILE)

"""Print one timestamped manager-subset progress line."""
function manager_subset_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Run a fixed training subset through the existing ProcessManager minibatch path."""
function run_manager_subset!(
    manager::M,
    xtrain::X,
    ytrain::Y,
    nsamples::I,
) where {M<:StatefulAlgorithms.ProcessManager,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    config = manager.config
    elapsed = @elapsed begin
        ncorrect = 0
        nskipped = 0
        total_loss = 0f0
        nbatches = 0

        for batch_start in 1:config.batchsize:Int(nsamples)
            batch_end = min(Int(nsamples), batch_start + config.batchsize - 1)
            jobs = batch_jobs(xtrain, ytrain, collect(batch_start:batch_end))
            stats = run_minibatch!(manager, jobs; log_progress = false)
            ncorrect += round(Int, stats.accuracy * stats.nsamples)
            nskipped += stats.skipped
            total_loss += stats.loss * stats.nsamples
            nbatches += 1
            manager_subset_log(
                "manager subset progress";
                batch = nbatches,
                samples = batch_end,
                nsamples = Int(nsamples),
                elapsed_s = round(time() - MANAGER_T0; digits = 3),
            )
        end

        global MANAGER_LAST_STATS = (;
            nsamples = Int(nsamples),
            batch_count = nbatches,
            accuracy = ncorrect / Int(nsamples),
            loss = total_loss / Int(nsamples),
            skipped = nskipped,
        )
    end
    return merge(MANAGER_LAST_STATS, (; elapsed_seconds = elapsed))
end

"""Build a fresh manager and time only the subset execution, not model construction."""
function timed_manager_subset(config::C, xtrain::X, ytrain::Y, nsamples::I) where {C<:LocalMNISTManagerConfig,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    model = init_model(config, config.seed)
    manager = local_manager(model)
    try
        return run_manager_subset!(manager, xtrain, ytrain, nsamples)
    finally
        close(manager)
    end
end

"""Run warmup and measured manager subset diagnostics."""
function main()
    config = LocalMNISTManagerConfig(;
        name = "manager_metropolis_subset",
        workers = parse(Int, get(ENV, "ISING_MNIST_PM_WORKERS", "32")),
        epochs = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = MANAGER_OUTDIR,
    )
    Threads.nthreads() < config.workers &&
        @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers

    manager_subset_log("loading balanced MNIST subset")
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    full_n = size(xtrain, 2)
    default_subset = ceil(Int, full_n * 60 / 220)
    subset_n = min(full_n, parse(Int, get(ENV, "ISING_MNIST_MANAGER_SUBSET", string(default_subset))))
    warmup_n = min(subset_n, parse(Int, get(ENV, "ISING_MNIST_MANAGER_WARMUP_SUBSET", "1")))
    manager_subset_log("manager diagnostic configured"; workers = config.workers, threads = Threads.nthreads(), full_n, subset_n, warmup_n)

    manager_subset_log("manager warmup starting")
    warmup_stats = timed_manager_subset(config, xtrain, ytrain, warmup_n)
    manager_subset_log("manager warmup finished"; elapsed_s = round(warmup_stats.elapsed_seconds; digits = 3), samples = warmup_n)

    manager_subset_log("manager measured run starting")
    stats = timed_manager_subset(config, xtrain, ytrain, subset_n)
    estimated_epoch_seconds = stats.elapsed_seconds * full_n / subset_n
    seconds_per_sample = stats.elapsed_seconds / subset_n

    csv_path = joinpath(MANAGER_OUTDIR, "manager_summary.csv")
    open(csv_path, "w") do io
        println(io, "workers,threads,subset_samples,full_epoch_samples,elapsed_seconds,estimated_epoch_seconds,seconds_per_sample,batches,accuracy,loss,skipped")
        println(io, join((
            config.workers,
            Threads.nthreads(),
            subset_n,
            full_n,
            round(stats.elapsed_seconds; digits = 6),
            round(estimated_epoch_seconds; digits = 6),
            round(seconds_per_sample; digits = 9),
            stats.batch_count,
            round(stats.accuracy; digits = 6),
            round(stats.loss; digits = 6),
            stats.skipped,
        ), ","))
    end

    manager_subset_log(
        "manager measured subset finished";
        workers = config.workers,
        subset_n,
        elapsed_s = round(stats.elapsed_seconds; digits = 3),
        estimated_epoch_s = round(estimated_epoch_seconds; digits = 3),
        seconds_per_sample = round(seconds_per_sample; digits = 4),
        summary = csv_path,
    )
    return stats
end

main()
