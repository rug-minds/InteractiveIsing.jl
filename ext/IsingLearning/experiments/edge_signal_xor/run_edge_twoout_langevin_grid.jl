using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "edge_2_8x8_2_langevin.jl"))

using Dates
using CairoMakie

"""Return a compact name for one edge-connected Langevin grid configuration."""
function edge_grid_name(config::EdgeTwoOutConfig)
    return join((
        "$(config.hidden_height)x$(config.hidden_width)",
        "nn$(config.hidden_nn)",
        "out$(2 * config.output_repeats)",
        "T$(config.temp)",
        "Tv$(config.validation_temp)",
        "eta$(config.stepsize)",
        "b$(config.β)",
        "lr$(config.lr)",
        "gm$(config.gradient_mode)",
        "m$(config.minit)",
        "f$(config.free_relaxation)",
        "n$(config.nudged_relaxation)",
        "is$(config.input_scale)",
        "hs$(config.hidden_scale)",
        "os$(config.output_scale)",
        "ois$(config.output_internal_scale)",
    ), "_")
end

"""Write the best result of each grid configuration to one CSV file."""
function edge_grid_write_summary(path, rows)
    headers = [
        "name", "best_mse", "best_acc", "best_epoch", "run_dir",
        "height", "width", "nn", "temp", "validation_temp", "stepsize", "beta", "lr",
        "gradient_mode", "minit", "free", "nudged", "output_repeats", "input_scale", "hidden_scale", "output_scale",
        "output_internal_scale",
    ]
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((row[h] for h in headers), ","))
        end
    end
    return path
end

"""Plot best MSE and best accuracy for each grid configuration."""
function edge_grid_plot(path, rows)
    fig = Figure(size = (1100, 520))
    names = [String(row["name"]) for row in rows]
    xs = 1:length(rows)
    ax1 = Axis(fig[1, 1], title = "Best edge XOR MSE", xlabel = "config", ylabel = "MSE", xticks = (xs, names), xticklabelrotation = pi / 3)
    ax2 = Axis(fig[1, 2], title = "Best edge XOR accuracy", xlabel = "config", ylabel = "accuracy", xticks = (xs, names), xticklabelrotation = pi / 3)
    barplot!(ax1, xs, [Float64(row["best_mse"]) for row in rows])
    barplot!(ax2, xs, [Float64(row["best_acc"]) for row in rows])
    ylims!(ax2, 0, 1.05)
    save(path, fig)
    return path
end

"""Build the first edge-connected Langevin grid: one 4x4 control and several 8x8 variants."""
function edge_grid_configs()
    common = (;
        dynamics = :block,
        epochs = 2000,
        log_every = 500,
        eval_repeats = 16,
        workers = 1,
        β = 1.0,
        weight_decay = 0.0,
        block_size = 8,
        bias_scale = 0.05,
        weight_seed = 1701,
        bias_seed = 1703,
        base_seed = 917000,
    )
    return EdgeTwoOutConfig[
        EdgeTwoOutConfig(; common..., hidden_height = 4, hidden_width = 4, hidden_nn = 5,
            minit = 4, free_relaxation = 1000, nudged_relaxation = 1000,
            temp = 0.001, stepsize = 0.10, lr = 0.005,
            input_scale = 0.4, hidden_scale = 0.2, output_scale = 0.4),

        EdgeTwoOutConfig(; common..., hidden_height = 8, hidden_width = 8, hidden_nn = 5,
            minit = 4, free_relaxation = 2500, nudged_relaxation = 2500,
            temp = 0.001, stepsize = 0.12, lr = 0.00025,
            input_scale = 0.5, hidden_scale = 0.2, output_scale = 0.5),

        EdgeTwoOutConfig(; common..., hidden_height = 8, hidden_width = 8, hidden_nn = 5,
            minit = 8, free_relaxation = 4000, nudged_relaxation = 2000,
            temp = 0.0008, stepsize = 0.12, lr = 0.00015,
            input_scale = 0.8, hidden_scale = 0.1, output_scale = 0.8),

        EdgeTwoOutConfig(; common..., hidden_height = 8, hidden_width = 8, hidden_nn = 3,
            minit = 8, free_relaxation = 4000, nudged_relaxation = 2000,
            temp = 0.0008, stepsize = 0.12, lr = 0.00015,
            input_scale = 0.8, hidden_scale = 0.1, output_scale = 0.8),

        EdgeTwoOutConfig(; common..., hidden_height = 8, hidden_width = 8, hidden_nn = 5,
            minit = 8, free_relaxation = 5000, nudged_relaxation = 3000,
            temp = 0.0005, stepsize = 0.10, lr = 0.0001,
            input_scale = 1.0, hidden_scale = 0.05, output_scale = 1.0),
    ]
end

"""Run the edge-connected Langevin grid and save a compact summary."""
function run_edge_twoout_langevin_grid(configs = edge_grid_configs())
    root = joinpath(@__DIR__, "runs", "edge_twoout_grid_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    mkpath(root)
    rows = Dict{String,Any}[]
    for (idx, config) in enumerate(configs)
        name = edge_grid_name(config)
        run_dir = joinpath(root, lpad(string(idx), 2, "0") * "_" * name)
        println("\n=== edge grid $idx/$(length(configs)): $name ===")
        result = train_edge_twoout(config; outdir = run_dir)
        row = Dict{String,Any}(
            "name" => name,
            "best_mse" => result.best.mse,
            "best_acc" => result.best.acc,
            "best_epoch" => result.best.epoch,
            "run_dir" => run_dir,
            "height" => config.hidden_height,
            "width" => config.hidden_width,
            "nn" => config.hidden_nn,
            "temp" => config.temp,
            "validation_temp" => config.validation_temp,
            "stepsize" => config.stepsize,
            "beta" => config.β,
            "lr" => config.lr,
            "gradient_mode" => config.gradient_mode,
            "minit" => config.minit,
            "free" => config.free_relaxation,
            "nudged" => config.nudged_relaxation,
            "output_repeats" => config.output_repeats,
            "input_scale" => config.input_scale,
            "hidden_scale" => config.hidden_scale,
            "output_scale" => config.output_scale,
            "output_internal_scale" => config.output_internal_scale,
        )
        push!(rows, row)
        edge_grid_write_summary(joinpath(root, "summary.csv"), rows)
        edge_grid_plot(joinpath(root, "summary.png"), rows)
    end
    println("\nGRID_RESULT outdir=", root)
    return (; rows, outdir = root)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_edge_twoout_langevin_grid()
end
