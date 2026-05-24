include("edge_signal_split_input_search.jl")

"""
    split_input_scout_configs()

Return short split-input runs around the best previous edge result.

The goal is not a broad grid. It tests whether the previous `NN=5` result was
limited by bias, output-edge strength, temperature, or hidden smoothing.
"""
function split_input_scout_configs()
    base = (;
        epochs = 2400,
        log_every = 600,
        minit = 2,
        eval_repeats = 10,
        free_relaxation = 1000,
        nudged_relaxation = 1000,
        validation_relaxation = 2500,
        stepsize = FT(0.8),
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = 5,
        skip_response = true,
    )

    return [
        ("prev_best_repeat", EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.01), weight_seed = 703, bias_seed = 723, base_seed = 1_070_703, base...)),
        ("zero_bias_same", EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.0006), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.0), weight_seed = 703, bias_seed = 723, base_seed = 1_071_703, base...)),
        ("more_output_same", EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00055), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.55),
            bias_scale = FT(0.0), weight_seed = 703, bias_seed = 723, base_seed = 1_072_703, base...)),
        ("lower_T_same", EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00055), temp_fraction = FT(0.018),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.005), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.0), weight_seed = 703, bias_seed = 723, base_seed = 1_073_703, base...)),
        ("lower_hidden_same", EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00055), temp_fraction = FT(0.025),
            input_hidden_scale = FT(0.45), hidden_local_scale = FT(0.0025), hidden_output_scale = FT(0.40),
            bias_scale = FT(0.0), weight_seed = 703, bias_seed = 723, base_seed = 1_074_703, base...)),
    ]
end

"""Run the short split-input scout and save a summary CSV."""
function run_split_input_scout(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_scout_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, config) in split_input_scout_configs()
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_edge_xor(config; outdir)
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name" => name,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "best_epoch" => trained.best.epoch,
            "final_mse" => final["mse"],
            "final_accuracy" => final["accuracy"],
            "temp_fraction" => config.temp_fraction,
            "hidden_local_scale" => config.hidden_local_scale,
            "hidden_output_scale" => config.hidden_output_scale,
            "bias_scale" => config.bias_scale,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved split-input scout: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_scout()
end
