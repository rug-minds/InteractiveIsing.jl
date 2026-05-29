using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "local_checkerboard_xor", "local_checkerboard_xor.jl"))

const LANGEVIN_CHECKER_RUN_ROOT = joinpath(@__DIR__, "runs")

"""
    env_parse(name, default, T)

Read one experiment override from the environment. The defaults in this file are
chosen for a first Langevin pass; larger runs should override `epochs`,
`relaxation`, and `Minit` from the shell instead of editing this file.
"""
env_parse(name::AbstractString, default, ::Type{T}) where {T} =
    parse(T, get(ENV, name, string(default)))

"""
    langevin_checkerboard_configs()

Build checkerboard XOR configurations that keep the input representation
physical: two input bits freeze complementary checkerboard masks, not four
one-hot XOR cases. All configs use continuous spins and `BlockLangevin`.

The three default cases are:

- `lgv_4x4_global4`: 4x4 input, hidden, and output layers; the code uses every
  spin in the input/output layers.
- `lgv_8x8_global8`: 8x8 layers with global checkerboard input/output codes.
- `lgv_8x8_inlaid4`: 8x8 layers with a 4x4 code inlaid every two lattice sites.
- `lgv_4x4_hidden8`: 4x4 physical input/output codes with a larger 8x8
  hidden layer. This isolates "larger graph" from "larger readout code".
"""
function langevin_checkerboard_configs()
    epochs = env_parse("ISING_LGV_CHECKER_EPOCHS", 400, Int)
    log_every = env_parse("ISING_LGV_CHECKER_LOG_EVERY", 100, Int)
    minit = env_parse("ISING_LGV_CHECKER_MINIT", 4, Int)
    eval_repeats = env_parse("ISING_LGV_CHECKER_EVAL_REPEATS", 16, Int)
    workers = env_parse("ISING_LGV_CHECKER_THREADS", max(1, min(Threads.nthreads(), 8)), Int)
    relaxation = env_parse("ISING_LGV_CHECKER_RELAXATION", 1000, Int)
    nudged_relaxation = env_parse("ISING_LGV_CHECKER_NUDGED_RELAXATION", relaxation, Int)

    β = env_parse("ISING_LGV_CHECKER_BETA", 1.0, FT)
    lr = env_parse("ISING_LGV_CHECKER_LR", 0.003, FT)
    weight_decay = env_parse("ISING_LGV_CHECKER_WEIGHT_DECAY", 0.0, FT)
    grad_clip = env_parse("ISING_LGV_CHECKER_GRAD_CLIP", 80.0, FT)
    temp = env_parse("ISING_LGV_CHECKER_TEMP", 0.001, FT)
    stepsize = env_parse("ISING_LGV_CHECKER_STEPSIZE", 0.05, FT)

    inter_weight_scale = env_parse("ISING_LGV_CHECKER_INTER_SCALE", 0.10, FT)
    input_internal_scale = env_parse("ISING_LGV_CHECKER_INPUT_INTERNAL", 0.015, FT)
    hidden_internal_scale = env_parse("ISING_LGV_CHECKER_HIDDEN_INTERNAL", 0.025, FT)
    output_internal_scale = env_parse("ISING_LGV_CHECKER_OUTPUT_INTERNAL", 0.02, FT)
    bias_scale = env_parse("ISING_LGV_CHECKER_BIAS_SCALE", 0.02, FT)

    common = (;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation = relaxation,
        nudged_relaxation,
        β,
        lr,
        weight_decay,
        grad_clip,
        temp,
        temp_is_factor = false,
        stepsize,
        inter_weight_scale,
        input_internal_scale,
        hidden_internal_scale,
        output_internal_scale,
        bias_scale,
        weight_seed = env_parse("ISING_LGV_CHECKER_WEIGHT_SEED", 2, Int),
        internal_seed = env_parse("ISING_LGV_CHECKER_INTERNAL_SEED", 3, Int),
        bias_seed = env_parse("ISING_LGV_CHECKER_BIAS_SEED", 11, Int),
        base_seed = env_parse("ISING_LGV_CHECKER_BASE_SEED", 110_000, Int),
        dynamics_mode = :langevin,
        state_mode = :continuous,
        init_mode = :zero,
    )

    return [
        LocalCheckerboardConfig(;
            name = "lgv_4x4_global4",
            side = 4,
            hidden_side = 4,
            code_side = 4,
            code_stride = 1,
            internal_nn = env_parse("ISING_LGV_CHECKER_4_INTERNAL_NN", 1, Int),
            inter_radius = env_parse("ISING_LGV_CHECKER_4_INTER_RADIUS", sqrt(2.0) + 1e-6, FT),
            block_size = env_parse("ISING_LGV_CHECKER_4_BLOCK", 16, Int),
            common...,
        ),
        LocalCheckerboardConfig(;
            name = "lgv_8x8_global8",
            side = 8,
            hidden_side = 8,
            code_side = 8,
            code_stride = 1,
            internal_nn = env_parse("ISING_LGV_CHECKER_8_INTERNAL_NN", 2, Int),
            inter_radius = env_parse("ISING_LGV_CHECKER_8_INTER_RADIUS", 2.05, FT),
            block_size = env_parse("ISING_LGV_CHECKER_8_BLOCK", 64, Int),
            common...,
        ),
        LocalCheckerboardConfig(;
            name = "lgv_8x8_inlaid4",
            side = 8,
            hidden_side = 8,
            code_side = 4,
            code_stride = 2,
            code_offset = (1, 1),
            internal_nn = env_parse("ISING_LGV_CHECKER_INLAID_INTERNAL_NN", 2, Int),
            inter_radius = env_parse("ISING_LGV_CHECKER_INLAID_INTER_RADIUS", 2.55, FT),
            block_size = env_parse("ISING_LGV_CHECKER_INLAID_BLOCK", 64, Int),
            common...,
        ),
        LocalCheckerboardConfig(;
            name = "lgv_4x4_hidden8",
            side = 4,
            hidden_side = 8,
            code_side = 4,
            code_stride = 1,
            internal_nn = env_parse("ISING_LGV_CHECKER_H8_INTERNAL_NN", 2, Int),
            inter_radius = env_parse("ISING_LGV_CHECKER_H8_INTER_RADIUS", 2.55, FT),
            block_size = env_parse("ISING_LGV_CHECKER_H8_BLOCK", 32, Int),
            common...,
        ),
    ]
end

function selected_langevin_checkerboard_configs()
    wanted = split(get(ENV, "ISING_LGV_CHECKER_CONFIGS", "lgv_4x4_global4,lgv_4x4_hidden8,lgv_8x8_global8,lgv_8x8_inlaid4"), ",")
    all = langevin_checkerboard_configs()
    return [cfg for cfg in all if cfg.name in wanted]
end

function best_mse_row(rows)
    isempty(rows) && error("no rows")
    return sort(rows, by = row -> (row["mse"], -row["accuracy"]))[1]
end

function write_langevin_summary(path, summary_rows)
    keys_order = [
        "config", "side", "hidden_side", "code_side", "code_stride",
        "relaxation", "nudged_relaxation", "temp", "stepsize",
        "best_epoch", "best_mse", "best_accuracy", "best_min_margin",
        "final_mse", "final_accuracy", "graph",
    ]
    open(path, "w") do io
        println(io, join(keys_order, ","))
        for row in summary_rows
            println(io, join((row[k] for k in keys_order), ","))
        end
    end
    return path
end

function plot_langevin_summary(path, rows)
    mkpath(dirname(path))
    configs = unique(row["config"] for row in rows)
    fig = Figure(size = (1350, 900))
    panels = [
        ("mse", "readout MSE"),
        ("accuracy", "accuracy"),
        ("min_margin", "|score| margin"),
        ("response_norm", "free-to-nudged response"),
        ("grad_norm", "gradient norm"),
        ("param_delta", "parameter displacement"),
    ]
    for (i, (key, label)) in enumerate(panels)
        ax = Axis(fig[cld(i, 2), mod1(i, 2)], title = label, xlabel = "epoch", ylabel = label)
        for cfg in configs
            sub = [row for row in rows if row["config"] == cfg]
            isempty(sub) && continue
            lines!(ax, [row["epoch"] for row in sub], [row[key] for row in sub], label = cfg)
        end
        axislegend(ax, position = :rt)
    end
    save(path, fig)
    return path
end

"""
    main()

Run the selected Langevin checkerboard XOR experiments and save metrics, plots,
trained graphs, and a short run README.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_LGV_CHECKER_DIR", joinpath(LANGEVIN_CHECKER_RUN_ROOT, "langevin_checkerboard_xor_$timestamp"))
    mkpath(outdir)

    configs = selected_langevin_checkerboard_configs()
    isempty(configs) && error("No selected configs; check ISING_LGV_CHECKER_CONFIGS")

    all_rows = Dict{String,Any}[]
    summary_rows = Dict{String,Any}[]
    for config in configs
        println("Running $(config.name): ", config)
        result = run_config(config, joinpath(outdir, config.name))
        append!(all_rows, result.rows)
        best = best_mse_row(result.rows)
        final = result.rows[end]
        push!(
            summary_rows,
            Dict{String,Any}(
                "config" => config.name,
                "side" => config.side,
                "hidden_side" => config.hidden_side,
                "code_side" => config.code_side,
                "code_stride" => config.code_stride,
                "relaxation" => config.free_relaxation,
                "nudged_relaxation" => config.nudged_relaxation,
                "temp" => config.temp,
                "stepsize" => config.stepsize,
                "best_epoch" => best["epoch"],
                "best_mse" => best["mse"],
                "best_accuracy" => best["accuracy"],
                "best_min_margin" => best["min_margin"],
                "final_mse" => final["mse"],
                "final_accuracy" => final["accuracy"],
                "graph" => result.graph_path,
            ),
        )
    end

    metrics_csv = write_csv(joinpath(outdir, "langevin_checkerboard_xor_metrics.csv"), all_rows)
    summary_csv = write_langevin_summary(joinpath(outdir, "langevin_checkerboard_xor_summary.csv"), summary_rows)
    progress_png = plot_langevin_summary(joinpath(outdir, "langevin_checkerboard_xor_progress.png"), all_rows)

    readme = joinpath(outdir, "README.md")
    open(readme, "w") do io
        println(io, "# Langevin Checkerboard XOR")
        println(io)
        println(io, "This run tests physical checkerboard input masks with continuous `BlockLangevin` dynamics.")
        println(io, "The input is not four-case one-hot: bit A freezes one checkerboard mask to `+1`, bit B freezes the complementary mask to `+1`.")
        println(io)
        println(io, "## Summary")
        for row in summary_rows
            println(io, "- `$(row["config"])`: best mse=$(round(row["best_mse"], digits=6)), best acc=$(round(row["best_accuracy"], digits=3)), final mse=$(round(row["final_mse"], digits=6)), final acc=$(round(row["final_accuracy"], digits=3))")
        end
        println(io)
        println(io, "## Files")
        println(io, "- Metrics: `$(basename(metrics_csv))`")
        println(io, "- Summary: `$(basename(summary_csv))`")
        println(io, "- Plot: `$(basename(progress_png))`")
        println(io, "- Per-config folders contain best graph JLD2 files and parameter SVGs.")
    end

    println("Saved metrics: $metrics_csv")
    println("Saved summary: $summary_csv")
    println("Saved plot: $progress_png")
    println("Saved run docs: $readme")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
