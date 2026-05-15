include("edge_signal_split_input_regularized.jl")

"""
    split_input_weight_decay_long_trials()

Return longer split-input edge runs with mild weight decay.

The previous regularization scout used only 3600 epochs and fairly strong decay.
These trials keep the same `2 -> split first edge -> 8x8 -> last edge -> 1`
architecture, but use smaller decay so the effective temperature rises only
when the trained couplings drift upward or stop improving.
"""
function split_input_weight_decay_long_trials()
    base = (;
        epochs = 9000,
        log_every = 900,
        minit = 3,
        eval_repeats = 16,
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
        ("wd_1e_minus_3_Tx1p15", EdgeSignalXORConfig(; lr = FT(0.00045), weight_decay = FT(1e-3), base_seed = 1_101_703, base...), FT(1.15)),
        ("wd_3e_minus_3_Tx1p15", EdgeSignalXORConfig(; lr = FT(0.00045), weight_decay = FT(3e-3), base_seed = 1_102_703, base...), FT(1.15)),
        ("wd_1e_minus_3_Tx1p10", EdgeSignalXORConfig(; lr = FT(0.00045), weight_decay = FT(1e-3), base_seed = 1_103_703, base...), FT(1.10)),
        ("wd_3e_minus_3_Tx1p10", EdgeSignalXORConfig(; lr = FT(0.00040), weight_decay = FT(3e-3), base_seed = 1_104_703, base...), FT(1.10)),
    ]
end

"""Run longer mild-weight-decay split-input trials and save a summary CSV."""
function run_split_input_weight_decay_long(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_wd_long_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, config, nudged_temp_factor) in split_input_weight_decay_long_trials()
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
            "nudged_temp_factor" => nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved long mild-weight-decay search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_weight_decay_long()
end
