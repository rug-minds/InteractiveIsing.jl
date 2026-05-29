using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "langevin_checkerboard_xor.jl"))

"""
    parse_list(name, default, T)

Read a comma-separated sweep list from the environment.
"""
function parse_list(name::AbstractString, default, ::Type{T}) where {T}
    raw = get(ENV, name, join(string.(default), ","))
    return T.(parse.(T, split(raw, ",")))
end

function sweep_base_config(name; side, hidden_side, code_side, code_stride, inter_radius, block_size)
    return LocalCheckerboardConfig(
        name = name,
        side = side,
        hidden_side = hidden_side,
        code_side = code_side,
        code_stride = code_stride,
        code_offset = (1, 1),
        epochs = env_parse("ISING_LGV_SWEEP_EPOCHS", 150, Int),
        log_every = env_parse("ISING_LGV_SWEEP_LOG_EVERY", 50, Int),
        minit = env_parse("ISING_LGV_SWEEP_MINIT", 2, Int),
        eval_repeats = env_parse("ISING_LGV_SWEEP_EVAL_REPEATS", 4, Int),
        workers = env_parse("ISING_LGV_SWEEP_THREADS", max(1, min(Threads.nthreads(), 8)), Int),
        free_relaxation = 250,
        nudged_relaxation = 250,
        β = env_parse("ISING_LGV_SWEEP_BETA", 1.0, FT),
        lr = env_parse("ISING_LGV_SWEEP_LR", 0.01, FT),
        weight_decay = env_parse("ISING_LGV_SWEEP_WEIGHT_DECAY", 0.0, FT),
        grad_clip = env_parse("ISING_LGV_SWEEP_GRAD_CLIP", 120.0, FT),
        temp = 0.005,
        temp_is_factor = false,
        stepsize = 0.05,
        block_size = block_size,
        inter_radius = inter_radius,
        internal_nn = env_parse("ISING_LGV_SWEEP_INTERNAL_NN", 5, Int),
        inter_weight_scale = env_parse("ISING_LGV_SWEEP_INTER_SCALE", 0.25, FT),
        input_internal_scale = env_parse("ISING_LGV_SWEEP_INPUT_INTERNAL", 0.08, FT),
        hidden_internal_scale = env_parse("ISING_LGV_SWEEP_HIDDEN_INTERNAL", 0.08, FT),
        output_internal_scale = env_parse("ISING_LGV_SWEEP_OUTPUT_INTERNAL", 0.08, FT),
        bias_scale = env_parse("ISING_LGV_SWEEP_BIAS_SCALE", 0.02, FT),
        weight_seed = env_parse("ISING_LGV_SWEEP_WEIGHT_SEED", 2, Int),
        internal_seed = env_parse("ISING_LGV_SWEEP_INTERNAL_SEED", 3, Int),
        bias_seed = env_parse("ISING_LGV_SWEEP_BIAS_SEED", 11, Int),
        base_seed = env_parse("ISING_LGV_SWEEP_BASE_SEED", 130_000, Int),
        init_mode = :zero,
        state_mode = :continuous,
        dynamics_mode = :langevin,
    )
end

function sweep_prototypes()
    target = Symbol(get(ENV, "ISING_LGV_SWEEP_TARGET", "inlaid8"))
    if target === :global8
        return [sweep_base_config("global8"; side = 8, hidden_side = 8, code_side = 8, code_stride = 1, inter_radius = 3.05, block_size = 64)]
    elseif target === :inlaid8
        return [sweep_base_config("inlaid8"; side = 8, hidden_side = 8, code_side = 4, code_stride = 2, inter_radius = 3.05, block_size = 64)]
    elseif target === :hidden8
        return [sweep_base_config("hidden8"; side = 4, hidden_side = 8, code_side = 4, code_stride = 1, inter_radius = 3.05, block_size = 32)]
    elseif target === :all
        return [
            sweep_base_config("global8"; side = 8, hidden_side = 8, code_side = 8, code_stride = 1, inter_radius = 3.05, block_size = 64),
            sweep_base_config("inlaid8"; side = 8, hidden_side = 8, code_side = 4, code_stride = 2, inter_radius = 3.05, block_size = 64),
            sweep_base_config("hidden8"; side = 4, hidden_side = 8, code_side = 4, code_stride = 1, inter_radius = 3.05, block_size = 32),
        ]
    else
        throw(ArgumentError("ISING_LGV_SWEEP_TARGET must be global8, inlaid8, hidden8, or all"))
    end
end

function sweep_configs()
    temps = parse_list("ISING_LGV_SWEEP_TEMPS", [0.001, 0.005, 0.02], FT)
    stepsizes = parse_list("ISING_LGV_SWEEP_STEPSIZES", [0.03, 0.08], FT)
    relaxations = parse_list("ISING_LGV_SWEEP_RELAXATIONS", [250, 1000], Int)
    configs = LocalCheckerboardConfig[]
    for proto in sweep_prototypes(), T in temps, η in stepsizes, relaxation in relaxations
        name = "$(proto.name)_T$(replace(string(T), "." => "p"))_eta$(replace(string(η), "." => "p"))_r$(relaxation)"
        push!(
            configs,
            LocalCheckerboardConfig(;
                name,
                side = proto.side,
                hidden_side = proto.hidden_side,
                code_side = proto.code_side,
                code_stride = proto.code_stride,
                code_offset = proto.code_offset,
                epochs = proto.epochs,
                log_every = proto.log_every,
                minit = proto.minit,
                eval_repeats = proto.eval_repeats,
                workers = proto.workers,
                free_relaxation = relaxation,
                nudged_relaxation = relaxation,
                β = proto.β,
                lr = proto.lr,
                weight_decay = proto.weight_decay,
                grad_clip = proto.grad_clip,
                temp = T,
                temp_is_factor = proto.temp_is_factor,
                stepsize = η,
                block_size = proto.block_size,
                inter_radius = proto.inter_radius,
                internal_nn = proto.internal_nn,
                inter_weight_scale = proto.inter_weight_scale,
                input_internal_scale = proto.input_internal_scale,
                hidden_internal_scale = proto.hidden_internal_scale,
                output_internal_scale = proto.output_internal_scale,
                bias_scale = proto.bias_scale,
                weight_seed = proto.weight_seed,
                internal_seed = proto.internal_seed,
                bias_seed = proto.bias_seed,
                base_seed = proto.base_seed,
                init_mode = proto.init_mode,
                state_mode = proto.state_mode,
                dynamics_mode = proto.dynamics_mode,
            ),
        )
    end
    return configs
end

function sweep_main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_LGV_SWEEP_DIR", joinpath(LANGEVIN_CHECKER_RUN_ROOT, "langevin_checkerboard_sweep_$timestamp"))
    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summary_rows = Dict{String,Any}[]
    for config in sweep_configs()
        println("Sweep config $(config.name): T=$(config.temp) stepsize=$(config.stepsize) relaxation=$(config.free_relaxation)")
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
    metrics_csv = write_csv(joinpath(outdir, "langevin_checkerboard_sweep_metrics.csv"), all_rows)
    summary_csv = write_langevin_summary(joinpath(outdir, "langevin_checkerboard_sweep_summary.csv"), summary_rows)
    progress_png = plot_langevin_summary(joinpath(outdir, "langevin_checkerboard_sweep_progress.png"), all_rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Langevin Checkerboard Sweep")
        println(io)
        println(io, "- Metrics: `$(basename(metrics_csv))`")
        println(io, "- Summary: `$(basename(summary_csv))`")
        println(io, "- Plot: `$(basename(progress_png))`")
        println(io)
        println(io, "## Results")
        for row in sort(summary_rows, by = row -> row["best_mse"])
            println(io, "- `$(row["config"])`: best mse=$(round(row["best_mse"], digits=6)), acc=$(round(row["best_accuracy"], digits=3)), final mse=$(round(row["final_mse"], digits=6))")
        end
    end
    println("Saved sweep metrics: $metrics_csv")
    println("Saved sweep summary: $summary_csv")
    println("Saved sweep plot: $progress_png")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    sweep_main()
end
