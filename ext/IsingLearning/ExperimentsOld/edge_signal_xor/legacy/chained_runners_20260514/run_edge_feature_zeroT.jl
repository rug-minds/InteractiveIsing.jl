include("run_edge_feature_bands.jl")

"""Return zero-temperature feature-band edge settings."""
function edge_feature_zeroT_specs()
    common = (;
        epochs = 10000,
        log_every = 500,
        minit = 8,
        eval_repeats = 8,
        free_sweeps = 40,
        nudged_sweeps = 80,
        validation_sweeps = 160,
        hidden_nn = 1,
        input_hidden_scale = FT(0.85),
        hidden_local_scale = FT(0.12),
        hidden_output_scale = FT(0.40),
        bias_scale = FT(0.03),
        temp_fraction = FT(0.0),
    )
    raw = [
        (; name = "zeroT_feature_b0p25_eta1p5_lr2e4", β = FT(0.25), stepsize = FT(1.5), lr = FT(0.0002), wd = FT(0.0005), bump = FT(1.0), seed = 1_240_001),
        (; name = "zeroT_feature_b0p15_eta2p0_lr3e4", β = FT(0.15), stepsize = FT(2.0), lr = FT(0.0003), wd = FT(0.0005), bump = FT(1.0), seed = 1_240_101),
    ]
    specs = []
    for spec in raw
        cfg = split_input_config_from_sweeps(;
            common...,
            β = spec.β,
            stepsize = spec.stepsize,
            lr = spec.lr,
            weight_decay = spec.wd,
            base_seed = spec.seed,
        )
        push!(specs, (; spec.name, config = cfg.config, active_count = cfg.active_count, nudged_temp_factor = spec.bump))
    end
    return specs
end

"""Run the zero-temperature feature-band search."""
function run_edge_feature_zeroT(; rootdir = joinpath(@__DIR__, "runs", "edge_feature_zeroT_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    eval_burnin_sweeps = 80
    eval_average_sweeps = 40
    for spec in edge_feature_zeroT_specs()
        println("\n=== ", spec.name, " ===")
        println("active spins=", spec.active_count,
            " free=", spec.config.free_relaxation,
            " nudged=", spec.config.nudged_relaxation,
            " eval burn/avg sweeps=", eval_burnin_sweeps, "/", eval_average_sweeps)
        outdir = joinpath(rootdir, spec.name)
        mkpath(outdir)
        trained = train_edge_feature_xor(
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
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "summary.csv"), rows)
        if trained.best.acc == 1.0 && trained.best.mse < 0.12
            println("Stopping grid after successful setting: ", spec.name)
            break
        end
    end
    println("Saved edge feature zero-temperature run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_edge_feature_zeroT()
end
