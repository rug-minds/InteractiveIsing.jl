using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using Dates
using Printf

function default_env!(key, value)
    haskey(ENV, key) || (ENV[key] = string(value))
    return ENV[key]
end

default_env!("ISING_XOR_SEARCH_EPOCHS", 10_000)
default_env!("ISING_XOR_SEARCH_LOG_EVERY", 100)
default_env!("ISING_XOR_SEARCH_RELAXATION", 100)
default_env!("ISING_XOR_SEARCH_EVAL_REPEATS", 5)
default_env!("ISING_XOR_SEARCH_MINIT", 5)
default_env!("ISING_XOR_SEARCH_HIDDEN", "16")
default_env!("ISING_XOR_SEARCH_OUTPUT", 4)
default_env!("ISING_XOR_SEARCH_OUTPUT_CODE", "orthogonal")
default_env!("ISING_XOR_SEARCH_TARGET_MODE", "readout_hterm")
default_env!("ISING_XOR_SEARCH_READOUT_TARGET", 1.0)
default_env!("ISING_XOR_SEARCH_DYNAMICS", "block")
default_env!("ISING_XOR_SEARCH_STATE_MODE", "continuous")
default_env!("ISING_XOR_SEARCH_BIAS_SCALE", 0.1)
default_env!("ISING_XOR_SEARCH_LR", 0.01)
default_env!("ISING_XOR_SEARCH_BETA", 0.1)
default_env!("ISING_XOR_SEARCH_WEIGHT_NORM", 0.2)
default_env!("ISING_XOR_SEARCH_INPUT_BIAS", true)
default_env!("ISING_XOR_SEARCH_WEIGHT_SEED", 2)
default_env!("ISING_XOR_SEARCH_BIAS_SEED", 11)
default_env!("ISING_XOR_SEARCH_BASE_SEED", 24_000)
default_env!("ISING_XOR_SEARCH_PRINT_OUTPUTS", true)

const OUTDIR = get(
    ENV,
    "ISING_XOR_CURVE_DIR",
    joinpath(@__DIR__, "..", "runs", "xor_10k_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
)
mkpath(OUTDIR)

function parse_history(log_path)
    epochs = Int[]
    mses = Float64[]
    accuracies = Float64[]
    margins = Float64[]

    epoch_re = r"epoch=(\d+) mse=([0-9.eE+-]+) acc=([0-9.eE+-]+) margin=([0-9.eE+-]+)"
    before_re = r"before: mse=([0-9.eE+-]+) acc=([0-9.eE+-]+)"

    for line in eachline(log_path)
        if (m = match(epoch_re, line)) !== nothing
            push!(epochs, parse(Int, m.captures[1]))
            push!(mses, parse(Float64, m.captures[2]))
            push!(accuracies, parse(Float64, m.captures[3]))
            push!(margins, parse(Float64, m.captures[4]))
        elseif isempty(epochs) && (m = match(before_re, line)) !== nothing
            push!(epochs, 0)
            push!(mses, parse(Float64, m.captures[1]))
            push!(accuracies, parse(Float64, m.captures[2]))
            push!(margins, NaN)
        end
    end

    return (; epochs, mses, accuracies, margins)
end

function write_csv(path, history)
    open(path, "w") do io
        println(io, "epoch,mse,accuracy,margin")
        for idx in eachindex(history.epochs)
            @printf(
                io,
                "%d,%.12g,%.12g,%.12g\n",
                history.epochs[idx],
                history.mses[idx],
                history.accuracies[idx],
                history.margins[idx],
            )
        end
    end
    return path
end

function plot_history(path, history)
    fig = Figure(size = (960, 560))
    ax = Axis(
        fig[1, 1],
        xlabel = "epoch",
        ylabel = "readout score MSE",
        title = "XOR readout MSE from random signed J",
    )
    lines!(ax, history.epochs, history.mses, color = :steelblue, linewidth = 2)
    scatter!(ax, history.epochs, history.mses, color = :steelblue, markersize = 5)

    ax2 = Axis(
        fig[2, 1],
        xlabel = "epoch",
        ylabel = "accuracy",
        limits = (nothing, (0, 1.05)),
    )
    lines!(ax2, history.epochs, history.accuracies, color = :darkorange, linewidth = 2)
    scatter!(ax2, history.epochs, history.accuracies, color = :darkorange, markersize = 5)

    save(path, fig)
    return path
end

log_path = joinpath(OUTDIR, "xor_10k.log")
open(log_path, "w") do logfile
    redirect_stdout(logfile) do
        include(joinpath(@__DIR__, "xor_bespoke_search.jl"))
    end
end

history = parse_history(log_path)
isempty(history.epochs) && error("no epoch history parsed from $log_path")

csv_path = write_csv(joinpath(OUTDIR, "xor_10k_mse.csv"), history)
plot_path = plot_history(joinpath(OUTDIR, "xor_10k_mse.png"), history)

println("Saved XOR 10k log: $log_path")
println("Saved XOR 10k CSV: $csv_path")
println("Saved XOR 10k plot: $plot_path")
