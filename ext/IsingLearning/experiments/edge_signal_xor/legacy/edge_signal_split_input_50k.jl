include("edge_signal_split_input_regularized.jl")

"""
    split_input_50k_config()

Return the focused 50k-epoch split-input edge run.

This uses the best previous recipe: mild decoupled weight decay and a small
nudged-only temperature bump. Averaging is kept moderate so the run is long
enough to see whether the weak scalar output keeps improving.
"""
function split_input_50k_config()
    return EdgeSignalXORConfig(;
        epochs = 50_000,
        log_every = 2_500,
        minit = 3,
        eval_repeats = 16,
        free_relaxation = 1200,
        nudged_relaxation = 1200,
        validation_relaxation = 3500,
        β = FT(0.20),
        lr = FT(0.00040),
        weight_decay = FT(0.003),
        stepsize = FT(0.8),
        temp_fraction = FT(0.018),
        input_hidden_scale = FT(0.45),
        hidden_local_scale = FT(0.005),
        hidden_output_scale = FT(0.40),
        bias_scale = FT(0.0),
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = 5,
        target_scale = one(FT),
        skip_response = true,
        weight_seed = 703,
        bias_seed = 723,
        base_seed = 1_104_703,
    )
end

"""Run the focused 50k split-input edge learning experiment."""
function run_split_input_50k(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_50k_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    config = split_input_50k_config()
    trained = train_split_input_regularized_xor(config; outdir = rootdir, nudged_temp_factor = FT(1.10))
    final = trained.rows[end]
    rows = [Dict{String,Any}(
        "name" => "wd003_Tx1p10_50k",
        "best_mse" => trained.best.mse,
        "best_accuracy" => trained.best.acc,
        "best_epoch" => trained.best.epoch,
        "final_mse" => final["mse"],
        "final_accuracy" => final["accuracy"],
        "weight_decay" => config.weight_decay,
        "nudged_temp_factor" => FT(1.10),
        "outdir" => rootdir,
    )]
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved focused 50k split-input run: ", rootdir)
    return (; rootdir, config, trained)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_50k()
end
