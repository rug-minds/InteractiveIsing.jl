include("edge_signal_split_input_core.jl")

"""
    split_input_high_sweep_specs()

Return the current high-sweep split-edge XOR configurations.

Each configuration keeps the architecture fixed:
`2 input spins -> split first hidden edge -> 8x8 hidden layer -> last hidden edge -> 1 output spin`.
Only relaxation depth and learning rate are varied.
"""
function split_input_high_sweep_specs()
    common = (;
        epochs = 6000,
        log_every = 1000,
        minit = 3,
        eval_repeats = 12,
        β = FT(0.20),
        weight_decay = FT(0.003),
    )
    return [
        ("sweeps160_160_val320", split_input_config_from_sweeps(;
            free_sweeps = 160,
            nudged_sweeps = 160,
            validation_sweeps = 320,
            lr = FT(0.00028),
            base_seed = 1_140_703,
            common...,
        ), FT(1.10)),
        ("sweeps240_240_val480", split_input_config_from_sweeps(;
            free_sweeps = 240,
            nudged_sweeps = 240,
            validation_sweeps = 480,
            lr = FT(0.00022),
            base_seed = 1_141_703,
            common...,
        ), FT(1.10)),
        ("sweeps320_320_val640", split_input_config_from_sweeps(;
            free_sweeps = 320,
            nudged_sweeps = 320,
            validation_sweeps = 640,
            lr = FT(0.00018),
            base_seed = 1_142_703,
            common...,
        ), FT(1.10)),
    ]
end

"""Run the high-sweep split-input edge search and save a summary CSV."""
function run_split_input_high_sweeps(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_high_sweeps_clean_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, spec, nudged_temp_factor) in split_input_high_sweep_specs()
        config = spec.config
        println("\n=== ", name, " ===")
        println("active spins=", spec.active_count,
            " free sweeps=", spec.free_sweeps,
            " nudged sweeps=", spec.nudged_sweeps,
            " validation sweeps=", spec.validation_sweeps)
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_regularized_xor(config; outdir, nudged_temp_factor)
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
            "nudged_temp_factor" => nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved clean high-sweep split-input run: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_high_sweeps()
end
