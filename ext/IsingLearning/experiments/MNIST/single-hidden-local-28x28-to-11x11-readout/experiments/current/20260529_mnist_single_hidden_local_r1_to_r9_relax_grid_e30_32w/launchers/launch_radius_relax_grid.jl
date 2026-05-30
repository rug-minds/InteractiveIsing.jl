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

"""Append one radius-grid summary row."""
function append_grid_row!(row::R) where {R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(SUMMARY_PATH) || filesize(SUMMARY_PATH) == 0
    open(SUMMARY_PATH, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return row
end

"""Parse comma-separated integer values from an environment variable."""
function parse_grid_ints(name::S, default::V) where {S<:AbstractString,V<:Tuple}
    value = strip(get(ENV, name, ""))
    isempty(value) && return default
    return Tuple(parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part)))
end

"""Run one radius/relaxation grid point and append root-level status."""
function run_grid_point!(sweeps::I, radius::J) where {I<:Integer,J<:Integer}
    name = "r$(Int(radius))_e30"
    outdir = joinpath(EXP_ROOT, "s$(Int(sweeps))", name)
    config = LocalMNISTManagerConfig(;
        name,
        outdir,
        workers = 32,
        epochs = 30,
        batchsize = 32,
        train_per_class = 100,
        test_per_class = 20,
        local_radius = Int(radius),
        free_sweeps = Int(sweeps),
        nudge_sweeps = Int(sweeps),
        free_reads = 3,
        nudge_reads = 3,
        β = 5.0f0,
        optimizer = "adam",
        gradient_normalization = :mean,
        progress = true,
        progress_bar = false,
        progress_every = 5,
    )

    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] radius grid run started sweeps=", sweeps, " radius=", radius)
    flush(stdout)
    started = time()
    try
        result = run_config!(config)
        last_row = result.rows[end]
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "ok",
            sweeps = Int(sweeps),
            radius = Int(radius),
            beta = 5.0,
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
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] radius grid run finished sweeps=", sweeps, " radius=", radius, " best_accuracy=", result.best_accuracy)
        flush(stdout)
        return result
    catch err
        mkpath(outdir)
        open(joinpath(outdir, "error.txt"), "w") do io
            showerror(io, err, catch_backtrace())
        end
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "error",
            sweeps = Int(sweeps),
            radius = Int(radius),
            beta = 5.0,
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
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] radius grid run failed sweeps=", sweeps, " radius=", radius, " error=", err)
        flush(stdout)
        return nothing
    end
end

"""Run the radius grid sequentially, grouped by relaxation sweeps."""
function main()
    mkpath(EXP_ROOT)
    sweeps_grid = parse_grid_ints("ISING_MNIST_RADIUS_SWEEPS", (25, 50))
    radius_grid = parse_grid_ints("ISING_MNIST_RADIUS_GRID", (1, 2, 3, 4, 5, 6, 7, 8, 9))
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] radius/relax grid launcher started root=", EXP_ROOT)
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] threads=", Threads.nthreads(), " sweeps=", sweeps_grid, " radii=", radius_grid)
    flush(stdout)

    for sweeps in sweeps_grid
        mkpath(joinpath(EXP_ROOT, "s$(Int(sweeps))"))
        for radius in radius_grid
            run_grid_point!(sweeps, radius)
            GC.gc()
        end
    end
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] radius/relax grid launcher finished")
    flush(stdout)
    return nothing
end

main()
