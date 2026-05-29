using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Statistics
using Dates
using CairoMakie

include(joinpath(@__DIR__, "analytic_2_4_1.jl"))

"""
    RelaxationSweepConfig(; kwargs...)

Top-level configuration for a small relaxation-budget sweep on the random-init
`2 -> 4 -> 1` scalar XOR experiment.
"""
Base.@kwdef struct RelaxationSweepConfig
    epochs::Int = parse(Int, get(ENV, "ISING_241_SWEEP_EPOCHS", "1800"))
    log_every::Int = parse(Int, get(ENV, "ISING_241_SWEEP_LOG_EVERY", "300"))
    minit::Int = parse(Int, get(ENV, "ISING_241_SWEEP_MINIT", "8"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_241_SWEEP_EVAL_REPEATS", "16"))
    β::FT = parse(FT, get(ENV, "ISING_241_SWEEP_BETA", "2.0"))
    lr::FT = parse(FT, get(ENV, "ISING_241_SWEEP_LR", "0.002"))
    temp::FT = parse(FT, get(ENV, "ISING_241_SWEEP_TEMP", "0.003"))
    stepsize::FT = parse(FT, get(ENV, "ISING_241_SWEEP_STEPSIZE", "0.3"))
end

"""
    relaxation_pairs()

Return the free/nudged relaxation budgets to compare. The defaults test whether
the successful `1000/1000` recipe is actually necessary and whether asymmetric
budgets help.
"""
function relaxation_pairs()
    return (
        (free = 300, nudged = 300),
        (free = 600, nudged = 600),
        (free = 1000, nudged = 1000),
        (free = 1500, nudged = 1000),
        (free = 1000, nudged = 1500),
    )
end

"""
    make_config(base, pair)

Convert a relaxation-budget pair into the concrete `Analytic241Config` used by
the shared experiment code.
"""
function make_config(base::RelaxationSweepConfig, pair)
    return Analytic241Config(;
        epochs = base.epochs,
        log_every = base.log_every,
        minit = base.minit,
        eval_repeats = base.eval_repeats,
        free_relaxation = pair.free,
        nudged_relaxation = pair.nudged,
        β = base.β,
        lr = base.lr,
        weight_decay = zero(FT),
        temp = base.temp,
        stepsize = base.stepsize,
        init = :random,
    )
end

"""
    run_relaxation_case(config, label)

Train one random-init scalar XOR run and return all logged rows plus the best
logged MSE/accuracy seen during the run.
"""
function run_relaxation_case(config::Analytic241Config, label::AbstractString)
    trainer = trainer_241(config)
    x, yraw = xor_dataset_241()
    y = scaled_targets(yraw, config)
    xbatch, ybatch = repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]
    best = (mse = Inf, acc = zero(FT), epoch = 0)

    for epoch in 0:config.epochs
        if epoch > 0
            IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
            IsingLearning._broadcast_params!(trainer)
        end
        if epoch == 0 || epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_241!(trainer, x, y, config; seed_offset = config.base_seed + 40_000_000)
            grad_norm = epoch == 0 ? zero(FT) : sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            row = Dict{String,Any}(
                "label" => label,
                "epoch" => epoch,
                "free" => config.free_relaxation,
                "nudged" => config.nudged_relaxation,
                "mse" => metrics.mse,
                "accuracy" => metrics.acc,
                "margin" => metrics.margin,
                "grad_norm" => grad_norm,
            )
            for i in eachindex(metrics.means)
                row["mean_$i"] = metrics.means[i]
            end
            push!(rows, row)
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
            end
            println(label, " epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " margin=", round(metrics.margin, digits = 4))
        end
    end
    close_trainer!(trainer)
    return (; rows, best)
end

"""Write sweep rows to CSV."""
function write_sweep_csv(path, rows)
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

"""Save a compact comparison plot for the relaxation sweep."""
function plot_sweep(path, rows)
    fig = Figure(size = (1100, 800))
    ax1 = Axis(fig[1, 1], title = "MSE by relaxation budget", xlabel = "epoch", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy")
    ax3 = Axis(fig[2, 1], title = "margin", xlabel = "epoch", ylabel = "min |mean output|")
    ax4 = Axis(fig[2, 2], title = "gradient norm", xlabel = "epoch", ylabel = "||grad||")
    for label in unique(row["label"] for row in rows)
        subset = [row for row in rows if row["label"] == label]
        epochs = [row["epoch"] for row in subset]
        lines!(ax1, epochs, [row["mse"] for row in subset], label = label)
        lines!(ax2, epochs, [row["accuracy"] for row in subset], label = label)
        lines!(ax3, epochs, [row["margin"] for row in subset], label = label)
        lines!(ax4, epochs, [row["grad_norm"] for row in subset], label = label)
    end
    axislegend(ax1, position = :rt)
    save(path, fig)
    return path
end

"""Run the relaxation sweep and write CSV/PNG/README outputs."""
function main()
    base = RelaxationSweepConfig()
    outdir = get(
        ENV,
        "ISING_241_SWEEP_DIR",
        joinpath(@__DIR__, "runs", "relaxation_sweep_2_4_1_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summaries = []
    for pair in relaxation_pairs()
        label = "$(pair.free)/$(pair.nudged)"
        result = run_relaxation_case(make_config(base, pair), label)
        append!(all_rows, result.rows)
        push!(summaries, (label = label, best = result.best))
    end
    csv_path = write_sweep_csv(joinpath(outdir, "relaxation_sweep.csv"), all_rows)
    png_path = plot_sweep(joinpath(outdir, "relaxation_sweep.png"), all_rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# 2->4->1 Relaxation Sweep")
        println(io)
        println(io, "- epochs: `$(base.epochs)`")
        println(io, "- log_every: `$(base.log_every)`")
        println(io, "- Minit/eval repeats: `$(base.minit)` / `$(base.eval_repeats)`")
        println(io, "- β/lr/T/stepsize: `$(base.β)` / `$(base.lr)` / `$(base.temp)` / `$(base.stepsize)`")
        println(io)
        println(io, "| free/nudged | best epoch | best MSE | best acc |")
        println(io, "|---|---:|---:|---:|")
        for summary in summaries
            println(io, "| `$(summary.label)` | $(summary.best.epoch) | $(round(summary.best.mse, digits = 6)) | $(summary.best.acc) |")
        end
    end
    println("Saved sweep: ", outdir)
    println("CSV: ", csv_path)
    println("Plot: ", png_path)
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
