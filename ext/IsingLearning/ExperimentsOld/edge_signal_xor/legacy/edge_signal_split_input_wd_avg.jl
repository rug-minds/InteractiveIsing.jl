include("edge_signal_split_input_regularized.jl")

"""
    split_input_weight_decay_avg_trials()

Return high-averaging follow-up trials around the best mild-decay run.

The previous best used `minit = 3`, `weight_decay = 0.003`, and a small
nudged-temperature bump. These trials increase the number of repeated starting
states per XOR sample to reduce stochastic gradient noise.
"""
function split_input_weight_decay_avg_trials()
    base = (;
        epochs = 9000,
        log_every = 900,
        minit = 6,
        eval_repeats = 32,
        free_relaxation = 1200,
        nudged_relaxation = 1200,
        validation_relaxation = 3500,
        stepsize = FT(0.8),
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = 5,
        β = FT(0.20),
        temp_fraction = FT(0.018),
        input_hidden_scale = FT(0.45),
        hidden_local_scale = FT(0.005),
        hidden_output_scale = FT(0.40),
        bias_scale = FT(0.0),
        skip_response = true,
        weight_seed = 703,
        bias_seed = 723,
    )
    return [
        ("avg6_wd003_Tx1p10", EdgeSignalXORConfig(; lr = FT(0.00040), weight_decay = FT(0.003), base_seed = 1_111_703, base...), FT(1.10)),
        ("avg6_wd005_Tx1p10", EdgeSignalXORConfig(; lr = FT(0.00035), weight_decay = FT(0.005), base_seed = 1_112_703, base...), FT(1.10)),
    ]
end

"""Run high-averaging split-input edge trials and save a summary CSV."""
function run_split_input_weight_decay_avg(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_wd_avg_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, config, nudged_temp_factor) in split_input_weight_decay_avg_trials()
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_regularized_xor(config; outdir, nudged_temp_factor)
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "lr" => config.lr,
            "weight_decay" => config.weight_decay,
            "minit" => config.minit,
            "eval_repeats" => config.eval_repeats,
            "nudged_temp_factor" => nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved high-averaging mild-weight-decay search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_weight_decay_avg()
end
