include("edge_signal_split_input_sweep_relax.jl")

"""
    split_input_high_sweep_trials()

Return higher-relaxation split-edge XOR trials.

The 80-sweep run was still weak because the information has to travel from the
first hidden edge to the last hidden edge. These trials explicitly test larger
free/nudged relaxation depths while keeping the architecture fixed:
`2 -> split first edge -> 8x8 hidden -> last edge -> 1`.
"""
function split_input_high_sweep_trials()
    common = (;
        epochs = 6000,
        log_every = 1000,
        minit = 3,
        eval_repeats = 12,
        β = FT(0.20),
        weight_decay = FT(0.003),
        nudged_temp_factor = FT(1.10),
    )
    return [
        ("sweeps160_160_val320", split_input_config_from_sweeps(;
            free_sweeps = 160,
            nudged_sweeps = 160,
            validation_sweeps = 320,
            lr = FT(0.00028),
            base_seed = 1_140_703,
            common...,
        )),
        ("sweeps240_240_val480", split_input_config_from_sweeps(;
            free_sweeps = 240,
            nudged_sweeps = 240,
            validation_sweeps = 480,
            lr = FT(0.00022),
            base_seed = 1_141_703,
            common...,
        )),
        ("sweeps320_320_val640", split_input_config_from_sweeps(;
            free_sweeps = 320,
            nudged_sweeps = 320,
            validation_sweeps = 640,
            lr = FT(0.00018),
            base_seed = 1_142_703,
            common...,
        )),
    ]
end

"""Run higher-sweep split-input trials and save a summary CSV."""
function run_split_input_high_sweeps(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_high_sweeps_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, spec) in split_input_high_sweep_trials()
        config = spec.config
        println("\n=== ", name, " ===")
        println("active spins=", spec.active_count,
            " free steps=", config.free_relaxation,
            " nudged steps=", config.nudged_relaxation,
            " validation steps=", config.validation_relaxation)
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_regularized_xor(config; outdir, nudged_temp_factor = spec.nudged_temp_factor)
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => name,
            "active_count" => spec.active_count,
            "free_sweeps" => spec.free_sweeps,
            "nudged_sweeps" => spec.nudged_sweeps,
            "validation_sweeps" => spec.validation_sweeps,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "lr" => config.lr,
            "weight_decay" => config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved high-sweep split-input search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_high_sweeps()
end
