include("run_split_input_structured_search.jl")

"""Return growth-oriented settings after structured input reached weak full accuracy."""
function structured_growth_specs()
    common = (;
        epochs = 16000,
        log_every = 1000,
        minit = 4,
        eval_repeats = 8,
        free_sweeps = 35,
        nudged_sweeps = 90,
        validation_sweeps = 180,
        hidden_nn = 5,
        input_hidden_scale = FT(0.80),
        hidden_local_scale = FT(0.018),
        hidden_output_scale = FT(0.80),
        bias_scale = FT(0.12),
    )
    raw = [
        (; name = "growth_b0p45_T0p007_eta1p2_lr35e4_wd0", β = FT(0.45), temp_fraction = FT(0.007), stepsize = FT(1.2), lr = FT(0.00035), wd = FT(0.0), bump = FT(1.25), seed = 1_210_001),
        (; name = "growth_b0p30_T0p006_eta1p4_lr45e4_wd0", β = FT(0.30), temp_fraction = FT(0.006), stepsize = FT(1.4), lr = FT(0.00045), wd = FT(0.0), bump = FT(1.20), seed = 1_210_101),
        (; name = "growth_b0p40_T0p005_eta1p4_lr30e4_wd1e4", β = FT(0.40), temp_fraction = FT(0.005), stepsize = FT(1.4), lr = FT(0.00030), wd = FT(0.0001), bump = FT(1.30), seed = 1_210_201),
    ]
    specs = []
    for spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            temp_fraction = spec.temp_fraction,
            stepsize = spec.stepsize,
            lr = spec.lr,
            weight_decay = spec.wd,
            base_seed = spec.seed,
        )
        push!(specs, (; spec.name, config = cfg.config, active_count = cfg.active_count, nudged_temp_factor = spec.bump))
    end
    return specs
end

"""Run the growth-oriented structured-input search."""
function run_structured_growth_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_structured_growth_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 160
    eval_average_sweeps = 160
    for spec in structured_growth_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_structured_split_input_xor(
            spec.config;
            outdir,
            nudged_temp_factor = spec.nudged_temp_factor,
            eval_burnin_sweeps,
            eval_average_sweeps,
        )
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => spec.name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "beta" => spec.config.β,
            "temp_fraction" => spec.config.temp_fraction,
            "stepsize" => spec.config.stepsize,
            "lr" => spec.config.lr,
            "weight_decay" => spec.config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "summary.csv"), rows)
        if trained.best.acc == 1.0 && trained.best.mse < 0.12
            println("Stopping grid after successful setting: ", spec.name)
            break
        end
    end
    println("Saved structured growth run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_structured_growth_search()
end
