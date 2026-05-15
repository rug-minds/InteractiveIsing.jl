using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "simple_2_4_1_timeavg_learning.jl"))

"""
    grid_timestamp()

Return the timestamp suffix used for a single time-averaged XOR grid run.
"""
grid_timestamp() = Dates.format(now(), "yyyymmdd_HHMMSS")

"""
    make_grid_config(root, spec)

Convert a named tuple of hyperparameters into a `TimeAverageXorConfig`. The
defaults intentionally match the scalar `2 -> 4 -> 1` Langevin recipe that
already reached low MSE without derivative time averaging.
"""
function make_grid_config(root::AbstractString, spec)
    label = spec.label
    return TimeAverageXorConfig(
        epochs = get(spec, :epochs, parse(Int, get(ENV, "ISING_TIMEAVG_GRID_EPOCHS", "1600"))),
        log_every = get(spec, :log_every, parse(Int, get(ENV, "ISING_TIMEAVG_GRID_LOG_EVERY", "100"))),
        minit = get(spec, :minit, parse(Int, get(ENV, "ISING_TIMEAVG_GRID_MINIT", "8"))),
        eval_repeats = get(spec, :eval_repeats, parse(Int, get(ENV, "ISING_TIMEAVG_GRID_EVAL_REPEATS", "16"))),
        workers = get(spec, :workers, parse(Int, get(ENV, "ISING_TIMEAVG_GRID_WORKERS", string(max(1, min(Threads.nthreads(), 8)))))),
        free_relaxation = spec.free,
        nudged_relaxation = spec.nudged,
        β = get(spec, :β, FT(2.0)),
        lr = spec.lr,
        weight_decay = get(spec, :weight_decay, zero(FT)),
        grad_clip = get(spec, :grad_clip, FT(50)),
        temp = get(spec, :temp, FT(0.005)),
        eval_temp = get(spec, :eval_temp, get(spec, :temp, FT(0.005))),
        stepsize = get(spec, :stepsize, FT(0.4)),
        max_drift_fraction = get(spec, :max_drift_fraction, FT(1.0)),
        weight_scale = get(spec, :weight_scale, FT(0.18)),
        bias_scale = get(spec, :bias_scale, FT(0.02)),
        train_average_sweeps = spec.train_avg,
        train_sample_every_sweeps = get(spec, :train_every, 1),
        eval_burnin_sweeps = get(spec, :eval_burnin, 20),
        eval_average_sweeps = get(spec, :eval_avg, 80),
        eval_sample_every_sweeps = get(spec, :eval_every, 1),
        weight_seed = get(spec, :weight_seed, 31),
        bias_seed = get(spec, :bias_seed, 37),
        base_seed = get(spec, :base_seed, 84000),
        outdir = joinpath(root, string(label)),
    )
end

"""
    grid_specs()

Return the compact search around the known-good scalar Langevin recipe. The
`avg1` cases are the closest comparison to endpoint-gradient EqProp; larger
`train_avg` values test whether derivative time averaging improves the signal.
"""
function grid_specs()
    base = (
        β = FT(2.0),
        temp = FT(0.005),
        stepsize = FT(0.4),
        max_drift_fraction = FT(1.0),
        weight_decay = zero(FT),
        minit = 8,
        eval_repeats = 16,
        eval_burnin = 20,
        eval_avg = 80,
    )
    return [
        merge(base, (; label = "f300_n300_avg1_lr002", free = 300, nudged = 300, train_avg = 1, lr = FT(0.002))),
        merge(base, (; label = "f300_n300_avg5_lr002", free = 300, nudged = 300, train_avg = 5, lr = FT(0.002))),
        merge(base, (; label = "f300_n300_avg20_lr002", free = 300, nudged = 300, train_avg = 20, lr = FT(0.002))),
        merge(base, (; label = "f600_n600_avg1_lr002", free = 600, nudged = 600, train_avg = 1, lr = FT(0.002))),
        merge(base, (; label = "f600_n600_avg5_lr002", free = 600, nudged = 600, train_avg = 5, lr = FT(0.002))),
        merge(base, (; label = "f600_n600_avg20_lr002", free = 600, nudged = 600, train_avg = 20, lr = FT(0.002))),
        merge(base, (; label = "f600_n600_avg5_lr0015", free = 600, nudged = 600, train_avg = 5, lr = FT(0.0015))),
        merge(base, (; label = "f600_n600_avg5_lr003", free = 600, nudged = 600, train_avg = 5, lr = FT(0.003))),
    ]
end

"""
    summary_row(label, result)

Extract the best and final logged metrics from one `main` result.
"""
function summary_row(label::AbstractString, result)
    final = result.rows[end]
    return Dict{String,Any}(
        "label" => label,
        "best_epoch" => result.best.epoch,
        "best_mse" => result.best.mse,
        "best_accuracy" => result.best.acc,
        "best_means" => string(round.(result.best.means, digits = 4)),
        "final_epoch" => final["epoch"],
        "final_mse" => final["mse"],
        "final_accuracy" => final["accuracy"],
        "final_margin" => final["margin"],
        "final_grad_norm" => final["grad_norm"],
        "run_dir" => result.config.outdir,
    )
end

"""
    plot_grid_summary(path, rows)

Save a compact comparison plot for the grid summary rows.
"""
function plot_grid_summary(path::AbstractString, rows)
    labels = string.(getindex.(rows, "label"))
    best_mse = FT.(getindex.(rows, "best_mse"))
    final_mse = FT.(getindex.(rows, "final_mse"))
    best_acc = FT.(getindex.(rows, "best_accuracy"))
    xs = 1:length(rows)

    fig = Figure(size = (1200, 700))
    ax1 = Axis(fig[1, 1], title = "time-averaged 2->4->1 XOR grid", xlabel = "config", ylabel = "MSE")
    ax2 = Axis(fig[2, 1], title = "accuracy", xlabel = "config", ylabel = "accuracy")
    lines!(ax1, xs, best_mse, label = "best")
    scatter!(ax1, xs, best_mse)
    lines!(ax1, xs, final_mse, label = "final")
    scatter!(ax1, xs, final_mse)
    axislegend(ax1, position = :rt)
    lines!(ax2, xs, best_acc, label = "best accuracy")
    scatter!(ax2, xs, best_acc)
    ax1.xticks = (xs, labels)
    ax2.xticks = (xs, labels)
    ax1.xticklabelrotation = pi / 4
    ax2.xticklabelrotation = pi / 4
    save(path, fig)
    return path
end

"""
    main(; specs = grid_specs())

Run the time-averaged XOR grid in one Julia session and write a summary next to
the per-config run folders.
"""
function main(; specs = grid_specs())
    root = joinpath(@__DIR__, "runs", "timeavg_grid_" * grid_timestamp())
    mkpath(root)
    rows = Dict{String,Any}[]
    for (idx, spec) in enumerate(specs)
        println()
        println("grid ", idx, "/", length(specs), ": ", spec.label)
        config = make_grid_config(root, spec)
        result = run_timeavg_learning(config = config)
        push!(rows, summary_row(string(spec.label), result))
        println("grid result ", spec.label, ": best mse=", round(result.best.mse, digits = 6),
            " acc=", result.best.acc, " epoch=", result.best.epoch)
    end
    csv_path = write_csv(joinpath(root, "timeavg_grid_summary.csv"), rows)
    png_path = plot_grid_summary(joinpath(root, "timeavg_grid_summary.png"), rows)
    println()
    println("Saved grid summary: ", csv_path)
    println("Saved grid plot: ", png_path)
    return (; root, rows, csv_path, png_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
