using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "simple_2_4_1_langevin.jl"))

"""
    NormalGridSpec

One normal-EqProp `2 -> 4 -> 1` LocalLangevin hyperparameter point. This driver
does not run the split-snapshot route; it is only for getting the ordinary
free-then-nudged EqProp setup to learn first.
"""
Base.@kwdef struct NormalGridSpec
    name::String
    temp::FT
    stepsize::FT
    max_drift_fraction::FT
    free_relaxation::Int
    nudged_relaxation::Int
    β::FT
    lr::FT
    weight_scale::FT
    bias_scale::FT = FT(0.02)
end

"""
    grid_config(spec)

Convert a `NormalGridSpec` into a `SimpleXorConfig` while preserving the global
run-size environment overrides for epochs, repeats, and worker count.
"""
function grid_config(spec::NormalGridSpec)
    return SimpleXorConfig(
        name = spec.name,
        temp = spec.temp,
        stepsize = spec.stepsize,
        max_drift_fraction = spec.max_drift_fraction,
        free_relaxation = spec.free_relaxation,
        nudged_relaxation = spec.nudged_relaxation,
        β = spec.β,
        lr = spec.lr,
        weight_scale = spec.weight_scale,
        bias_scale = spec.bias_scale,
    )
end

"""
    normal_grid_specs()

Return a deliberately small first tuning grid around the knobs most likely to
matter for unadjusted LocalLangevin: temperature, step size, relaxation length,
nudging strength, and learning rate.
"""
function normal_grid_specs()
    return [
        NormalGridSpec(name = "T010_s020_b02_lr003", temp = 0.010, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 250, nudged_relaxation = 250, β = 0.20, lr = 0.003, weight_scale = 0.18),
        NormalGridSpec(name = "T020_s020_b02_lr003", temp = 0.020, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 250, nudged_relaxation = 250, β = 0.20, lr = 0.003, weight_scale = 0.18),
        NormalGridSpec(name = "T050_s020_b02_lr003", temp = 0.050, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 250, nudged_relaxation = 250, β = 0.20, lr = 0.003, weight_scale = 0.18),
        NormalGridSpec(name = "T020_s010_b02_lr003", temp = 0.020, stepsize = 0.10, max_drift_fraction = 0.50, free_relaxation = 300, nudged_relaxation = 300, β = 0.20, lr = 0.003, weight_scale = 0.18),
        NormalGridSpec(name = "T020_s040_b02_lr002", temp = 0.020, stepsize = 0.40, max_drift_fraction = 0.80, free_relaxation = 250, nudged_relaxation = 250, β = 0.20, lr = 0.002, weight_scale = 0.18),
        NormalGridSpec(name = "T020_s020_b05_lr002", temp = 0.020, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 250, nudged_relaxation = 250, β = 0.50, lr = 0.002, weight_scale = 0.18),
        NormalGridSpec(name = "T020_s020_b02_lr001_w025", temp = 0.020, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 300, nudged_relaxation = 300, β = 0.20, lr = 0.0015, weight_scale = 0.25),
        NormalGridSpec(name = "T050_s040_b05_lr001_w025", temp = 0.050, stepsize = 0.40, max_drift_fraction = 0.80, free_relaxation = 300, nudged_relaxation = 300, β = 0.50, lr = 0.0015, weight_scale = 0.25),
        NormalGridSpec(name = "T015_s015_b02_lr001_long", temp = 0.015, stepsize = 0.15, max_drift_fraction = 0.50, free_relaxation = 500, nudged_relaxation = 500, β = 0.20, lr = 0.0010, weight_scale = 0.22),
        NormalGridSpec(name = "T020_s020_b02_lr001_long", temp = 0.020, stepsize = 0.20, max_drift_fraction = 0.60, free_relaxation = 500, nudged_relaxation = 500, β = 0.20, lr = 0.0010, weight_scale = 0.22),
        NormalGridSpec(name = "T030_s025_b03_lr001_long", temp = 0.030, stepsize = 0.25, max_drift_fraction = 0.70, free_relaxation = 500, nudged_relaxation = 500, β = 0.30, lr = 0.0010, weight_scale = 0.22),
        NormalGridSpec(name = "T020_s030_b05_lr0007_long", temp = 0.020, stepsize = 0.30, max_drift_fraction = 0.80, free_relaxation = 600, nudged_relaxation = 600, β = 0.50, lr = 0.0007, weight_scale = 0.25),
        NormalGridSpec(name = "aggr_T005_s050_b10_lr0005", temp = 0.005, stepsize = 0.50, max_drift_fraction = 1.00, free_relaxation = 900, nudged_relaxation = 900, β = 1.00, lr = 0.0005, weight_scale = 0.25),
        NormalGridSpec(name = "aggr_T010_s050_b10_lr0005", temp = 0.010, stepsize = 0.50, max_drift_fraction = 1.00, free_relaxation = 900, nudged_relaxation = 900, β = 1.00, lr = 0.0005, weight_scale = 0.25),
        NormalGridSpec(name = "aggr_T020_s080_b10_lr0004", temp = 0.020, stepsize = 0.80, max_drift_fraction = 1.00, free_relaxation = 1000, nudged_relaxation = 1000, β = 1.00, lr = 0.0004, weight_scale = 0.30),
        NormalGridSpec(name = "aggr_T010_s100_b15_lr0003", temp = 0.010, stepsize = 1.00, max_drift_fraction = 1.00, free_relaxation = 1200, nudged_relaxation = 1200, β = 1.50, lr = 0.0003, weight_scale = 0.30),
        NormalGridSpec(name = "sweeps_T010_s050_b10_lr0005", temp = 0.010, stepsize = 0.50, max_drift_fraction = 1.00, free_relaxation = 7000, nudged_relaxation = 7000, β = 1.00, lr = 0.0005, weight_scale = 0.25),
        NormalGridSpec(name = "sweeps_T005_s050_b10_lr0005", temp = 0.005, stepsize = 0.50, max_drift_fraction = 1.00, free_relaxation = 7000, nudged_relaxation = 7000, β = 1.00, lr = 0.0005, weight_scale = 0.25),
        NormalGridSpec(name = "sweeps_T010_s080_b10_lr0004", temp = 0.010, stepsize = 0.80, max_drift_fraction = 1.00, free_relaxation = 7000, nudged_relaxation = 7000, β = 1.00, lr = 0.0004, weight_scale = 0.30),
        NormalGridSpec(name = "cold_T001_s080_b10_lr0004_w035", temp = 0.001, stepsize = 0.80, max_drift_fraction = 1.00, free_relaxation = 2500, nudged_relaxation = 2500, β = 1.00, lr = 0.0004, weight_scale = 0.35),
        NormalGridSpec(name = "cold_T002_s080_b10_lr0004_w035", temp = 0.002, stepsize = 0.80, max_drift_fraction = 1.00, free_relaxation = 2500, nudged_relaxation = 2500, β = 1.00, lr = 0.0004, weight_scale = 0.35),
        NormalGridSpec(name = "cold_T005_s100_b10_lr0003_w04", temp = 0.005, stepsize = 1.00, max_drift_fraction = 1.00, free_relaxation = 3000, nudged_relaxation = 3000, β = 1.00, lr = 0.0003, weight_scale = 0.40),
    ]
end

"""
    run_normal_spec(spec, outdir)

Train one normal EqProp point and return its result plus rows tagged by spec
name. The implementation reuses `run_route(...; split=false)` from the simple
experiment file.
"""
function run_normal_spec(spec::NormalGridSpec, outdir)
    config = grid_config(spec)
    println("normal grid: ", spec.name)
    result = run_route(config, spec.name; split = false)
    return result
end

"""
    write_grid_readme(path, specs, results, csv_path, png_path)

Write a compact summary of the normal-only grid search.
"""
function write_grid_readme(path, specs, results, csv_path, png_path)
    open(path, "w") do io
        println(io, "# Normal LocalLangevin 2->4->1 XOR Grid")
        println(io)
        println(io, "This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.")
        println(io)
        println(io, "| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |")
        println(io, "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        for (spec, result) in zip(specs, results)
            println(io, "| `$(spec.name)` | $(round(result.best_mse, digits = 6)) | $(round(result.best_acc, digits = 3)) | $(spec.temp) | $(spec.stepsize) | $(spec.max_drift_fraction) | $(spec.free_relaxation)/$(spec.nudged_relaxation) | $(spec.β) | $(spec.lr) | $(spec.weight_scale) |")
        end
        println(io)
        println(io, "CSV: `$(basename(csv_path))`")
        println(io, "Plot: `$(basename(png_path))`")
    end
    return path
end

"""
    main()

Run the normal-only LocalLangevin grid and save metrics, plot, and README.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_SIMPLE_XOR_GRID_DIR", joinpath(@__DIR__, "runs", "normal_grid_$timestamp"))
    mkpath(outdir)
    specs = normal_grid_specs()
    only = get(ENV, "ISING_SIMPLE_XOR_GRID_ONLY", "")
    if !isempty(only)
        specs = [spec for spec in specs if occursin(only, spec.name)]
    end
    limit = parse(Int, get(ENV, "ISING_SIMPLE_XOR_GRID_LIMIT", string(length(specs))))
    specs = specs[1:min(limit, length(specs))]
    all_rows = Dict{String,Any}[]
    results = []
    for spec in specs
        result = run_normal_spec(spec, outdir)
        append!(all_rows, result.rows)
        push!(results, result)
        println("best ", spec.name, ": mse=", round(result.best_mse, digits = 6), " acc=", round(result.best_acc, digits = 3))
    end
    csv_path = write_csv(joinpath(outdir, "normal_grid_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "normal_grid_progress.png"), all_rows)
    md_path = write_grid_readme(joinpath(outdir, "README.md"), specs, results, csv_path, png_path)
    println("Saved metrics: ", csv_path)
    println("Saved plot: ", png_path)
    println("Saved docs: ", md_path)
    return (; outdir, specs, results, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
