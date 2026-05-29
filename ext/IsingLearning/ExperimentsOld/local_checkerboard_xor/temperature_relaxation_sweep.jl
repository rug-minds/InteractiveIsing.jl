using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_xor.jl"))

"""
    sweep_values(env_name, default_values, T)

Parse a comma-separated environment variable into a vector of values. This keeps
the sweep file reproducible while making one-off tuning runs cheap from the
shell.
"""
function sweep_values(env_name::AbstractString, default_values, ::Type{T}) where {T}
    raw = get(ENV, env_name, join(string.(default_values), ","))
    return T.(parse.(T, split(raw, ",")))
end

"""
    relaxation_sweep_configs()

Build a temperature/relaxation grid for the local checkerboard XOR setup. The
default sweep uses the `2x2 -> 2x2 -> 2x2` case because it is the cheapest place
to determine whether the dynamics can separate the four input cases at all.
"""
function relaxation_sweep_configs()
    side = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_SIDE", "2"))
    hidden_side = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_HIDDEN_SIDE", string(side)))
    code_side = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_CODE_SIDE", string(side)))
    stride = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_CODE_STRIDE", "1"))
    temps = sweep_values("ISING_LOCAL_XOR_SWEEP_TEMPS", [0.001, 0.003, 0.01, 0.03], FT)
    relaxations = sweep_values("ISING_LOCAL_XOR_SWEEP_RELAX", [100, 250, 500], Int)

    epochs = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_EPOCHS", "150"))
    log_every = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_LOG_EVERY", "50"))
    minit = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_MINIT", "4"))
    eval_repeats = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_EVAL_REPEATS", "8"))
    workers = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    temp_is_factor = parse(Bool, get(ENV, "ISING_LOCAL_XOR_SWEEP_TEMP_IS_FACTOR", "false"))
    weight_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_WEIGHT_SEED", "2"))
    internal_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_INTERNAL_SEED", "3"))
    bias_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_BIAS_SEED", "11"))
    base_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_BASE_SEED", "91000"))

    configs = LocalCheckerboardConfig[]
    for T in temps, relaxation in relaxations
        push!(
            configs,
            LocalCheckerboardConfig(
                name = "side$(side)_T$(replace(string(T), "." => "p"))_r$(relaxation)",
                side = side,
                hidden_side = hidden_side,
                code_side = code_side,
                code_stride = stride,
                epochs = epochs,
                log_every = log_every,
                minit = minit,
                eval_repeats = eval_repeats,
                workers = workers,
                free_relaxation = relaxation,
                nudged_relaxation = relaxation,
                β = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_BETA", "0.2")),
                lr = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_LR", "0.01")),
                weight_decay = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_WEIGHT_DECAY", "1e-4")),
                temp = T,
                temp_is_factor = temp_is_factor,
                stepsize = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_STEPSIZE", "0.05")),
                block_size = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_BLOCK_SIZE", string(side <= 2 ? 4 : 16))),
                inter_radius = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_INTER_RADIUS", string(sqrt(2.0) + 1e-6))),
                internal_nn = parse(Int, get(ENV, "ISING_LOCAL_XOR_SWEEP_INTERNAL_NN", "1")),
                inter_weight_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_INTER_WEIGHT_SCALE", "0.25")),
                input_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_INPUT_INTERNAL_SCALE", "0.1")),
                hidden_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_HIDDEN_INTERNAL_SCALE", "0.1")),
                output_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_OUTPUT_INTERNAL_SCALE", "0.1")),
                bias_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_SWEEP_BIAS_SCALE", "0.02")),
                weight_seed = weight_seed,
                internal_seed = internal_seed,
                bias_seed = bias_seed,
                base_seed = base_seed,
                dynamics_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_SWEEP_DYNAMICS", "langevin")),
                state_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_SWEEP_STATE", "continuous")),
            ),
        )
    end
    return configs
end

function best_row(rows)
    isempty(rows) && error("no rows")
    return sort(rows, by = row -> (row["accuracy"], -row["mse"]), rev = true)[1]
end

function write_sweep_summary(path, rows)
    keys_order = ["config", "temp", "relaxation", "best_epoch", "best_mse", "best_accuracy", "best_min_margin", "final_mse", "final_accuracy"]
    open(path, "w") do io
        println(io, join(keys_order, ","))
        for row in rows
            println(io, join((row[k] for k in keys_order), ","))
        end
    end
    return path
end

function plot_sweep_summary(path, summary_rows)
    temps = sort(unique(row["temp"] for row in summary_rows))
    relaxations = sort(unique(row["relaxation"] for row in summary_rows))
    mse = fill(NaN, length(relaxations), length(temps))
    acc = fill(NaN, length(relaxations), length(temps))
    for row in summary_rows
        i = findfirst(==(row["relaxation"]), relaxations)
        j = findfirst(==(row["temp"]), temps)
        mse[i, j] = row["best_mse"]
        acc[i, j] = row["best_accuracy"]
    end

    fig = Figure(size = (1100, 460))
    ax1 = Axis(fig[1, 1], title = "best readout MSE", xlabel = "T", ylabel = "relaxation")
    hm1 = heatmap!(ax1, 1:length(temps), 1:length(relaxations), mse')
    ax1.xticks = (1:length(temps), string.(temps))
    ax1.yticks = (1:length(relaxations), string.(relaxations))
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3], title = "best accuracy", xlabel = "T", ylabel = "relaxation")
    hm2 = heatmap!(ax2, 1:length(temps), 1:length(relaxations), acc')
    ax2.xticks = (1:length(temps), string.(temps))
    ax2.yticks = (1:length(relaxations), string.(relaxations))
    Colorbar(fig[1, 4], hm2)
    save(path, fig)
    return path
end

"""
    main()

Run a compact temperature/relaxation sweep. Each point still writes a full graph
and per-point metrics in a subfolder; the sweep root additionally contains
`temperature_relaxation_summary.csv` and `temperature_relaxation_summary.png`.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_LOCAL_XOR_SWEEP_DIR", joinpath(DEFAULT_RUN_ROOT, "temperature_relaxation_sweep_$timestamp"))
    mkpath(outdir)
    summary = Dict{String,Any}[]
    for config in relaxation_sweep_configs()
        result = run_config(config, joinpath(outdir, config.name))
        best = best_row(result.rows)
        final = result.rows[end]
        push!(
            summary,
            Dict{String,Any}(
                "config" => config.name,
                "temp" => config.temp,
                "relaxation" => config.free_relaxation,
                "best_epoch" => best["epoch"],
                "best_mse" => best["mse"],
                "best_accuracy" => best["accuracy"],
                "best_min_margin" => best["min_margin"],
                "final_mse" => final["mse"],
                "final_accuracy" => final["accuracy"],
            ),
        )
    end
    csv_path = write_sweep_summary(joinpath(outdir, "temperature_relaxation_summary.csv"), summary)
    png_path = plot_sweep_summary(joinpath(outdir, "temperature_relaxation_summary.png"), summary)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Temperature/Relaxation Sweep")
        println(io)
        println(io, "This sweep varies only temperature and free/nudged relaxation length for the local checkerboard XOR experiment.")
        println(io)
        println(io, "- Summary CSV: `$(basename(csv_path))`")
        println(io, "- Summary PNG: `$(basename(png_path))`")
    end
    println("Saved sweep summary: $csv_path")
    println("Saved sweep plot: $png_path")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
