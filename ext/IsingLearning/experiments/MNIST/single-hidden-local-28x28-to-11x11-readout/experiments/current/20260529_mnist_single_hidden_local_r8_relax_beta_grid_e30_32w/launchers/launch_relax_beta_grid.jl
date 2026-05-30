using Dates

const EXP_ROOT = normpath(joinpath(@__DIR__, ".."))
const ARCH_ROOT = normpath(joinpath(EXP_ROOT, "..", "..", ".."))
const MANAGER_FILE = joinpath(ARCH_ROOT, "mnist_local_manager_grid.jl")
const SUMMARY_PATH = joinpath(EXP_ROOT, "grid_summary.csv")

delete!(ENV, "ISING_MNIST_PM_RESUME_CHECKPOINT")
ENV["ISING_MNIST_PM_PROGRESS"] = get(ENV, "ISING_MNIST_PM_PROGRESS", "true")
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = get(ENV, "ISING_MNIST_PM_PROGRESS_BAR", "false")
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"

include(MANAGER_FILE)

"""Append one grid-level status row after a run finishes or fails."""
function append_grid_row!(row::R) where {R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(SUMMARY_PATH) || filesize(SUMMARY_PATH) == 0
    open(SUMMARY_PATH, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return row
end

"""Return a filesystem-safe beta label."""
function beta_label(beta::T) where {T<:Real}
    return replace(string(beta), "." => "p")
end

"""Run one relaxation/beta grid point and append the root summary."""
function run_grid_point!(sweeps::I, beta::T) where {I<:Integer,T<:Real}
    name = "r8_s$(Int(sweeps))_beta$(beta_label(beta))_e30"
    outdir = joinpath(EXP_ROOT, name)
    config = LocalMNISTManagerConfig(;
        name,
        outdir,
        workers = 32,
        epochs = 30,
        batchsize = 32,
        train_per_class = 100,
        test_per_class = 20,
        local_radius = 8,
        free_sweeps = Int(sweeps),
        nudge_sweeps = Int(sweeps),
        free_reads = 3,
        nudge_reads = 3,
        β = PMNIST_FT(beta),
        optimizer = "adam",
        gradient_normalization = :mean,
        progress = true,
        progress_bar = false,
        progress_every = 5,
    )

    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] grid run started name=", name)
    flush(stdout)
    started = time()
    try
        result = run_config!(config)
        last_row = result.rows[end]
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "ok",
            name,
            sweeps = Int(sweeps),
            beta = Float64(beta),
            elapsed_seconds = round(time() - started; digits = 3),
            best_accuracy = result.best_accuracy,
            final_test_accuracy = last_row.test_accuracy,
            final_test_loss = last_row.test_loss,
            final_train_accuracy = last_row.train_accuracy,
            final_train_loss = last_row.train_loss,
            skipped = last_row.skipped,
            latest_checkpoint = last_row.latest_path,
            final_checkpoint = last_row.final_path,
            outdir,
            error = "",
        ))
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] grid run finished name=", name, " best_accuracy=", result.best_accuracy)
        flush(stdout)
        return result
    catch err
        error_path = joinpath(outdir, "error.txt")
        mkpath(outdir)
        open(error_path, "w") do io
            showerror(io, err, catch_backtrace())
        end
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "error",
            name,
            sweeps = Int(sweeps),
            beta = Float64(beta),
            elapsed_seconds = round(time() - started; digits = 3),
            best_accuracy = missing,
            final_test_accuracy = missing,
            final_test_loss = missing,
            final_train_accuracy = missing,
            final_train_loss = missing,
            skipped = missing,
            latest_checkpoint = "",
            final_checkpoint = "",
            outdir,
            error = sprint(showerror, err),
        ))
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] grid run failed name=", name, " error=", err)
        flush(stdout)
        return nothing
    end
end

"""Run the full relaxation/beta grid sequentially in one Julia session."""
function main()
    mkpath(EXP_ROOT)
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] relaxation/beta grid launcher started root=", EXP_ROOT)
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] threads=", Threads.nthreads())
    flush(stdout)

    sweeps_grid = (10, 25, 50, 100)
    beta_grid = (2.5f0, 5.0f0, 10.0f0)
    for sweeps in sweeps_grid
        for beta in beta_grid
            run_grid_point!(sweeps, beta)
            GC.gc()
        end
    end
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] relaxation/beta grid launcher finished")
    flush(stdout)
    return nothing
end

main()
