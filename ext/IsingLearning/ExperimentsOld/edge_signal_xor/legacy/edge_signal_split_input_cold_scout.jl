include("edge_signal_split_input_search.jl")

"""
    split_input_cold_scout_configs()

Return colder split-input edge runs after the first scout showed correct signs
but weak scalar-output magnitude at `T fraction = 0.018`.
"""
function split_input_cold_scout_configs()
    base = (; epochs=3600, log_every=600, minit=3, eval_repeats=16,
        free_relaxation=1200, nudged_relaxation=1200, validation_relaxation=4000,
        stepsize=FT(0.8), hidden_height=8, hidden_width=8, hidden_nn=5,
        input_hidden_scale=FT(0.45), hidden_local_scale=FT(0.005), bias_scale=FT(0.0),
        skip_response=true)
    return [
        ("T0015_out040", EdgeSignalXORConfig(; β=FT(0.20), lr=FT(0.00045), temp_fraction=FT(0.015),
            hidden_output_scale=FT(0.40), weight_seed=703, bias_seed=723, base_seed=1_083_703, base...)),
        ("T0012_out040", EdgeSignalXORConfig(; β=FT(0.20), lr=FT(0.00040), temp_fraction=FT(0.012),
            hidden_output_scale=FT(0.40), weight_seed=703, bias_seed=723, base_seed=1_084_703, base...)),
        ("T0015_out055", EdgeSignalXORConfig(; β=FT(0.20), lr=FT(0.00040), temp_fraction=FT(0.015),
            hidden_output_scale=FT(0.55), weight_seed=703, bias_seed=723, base_seed=1_085_703, base...)),
        ("T0018_beta035", EdgeSignalXORConfig(; β=FT(0.35), lr=FT(0.00035), temp_fraction=FT(0.018),
            hidden_output_scale=FT(0.40), weight_seed=703, bias_seed=723, base_seed=1_086_703, base...)),
    ]
end

"""Run the colder split-input edge scout and save a summary CSV."""
function run_split_input_cold_scout(; rootdir=joinpath(@__DIR__, "runs", "edge_split_input_cold_scout_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, config) in split_input_cold_scout_configs()
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_edge_xor(config; outdir)
        final = trained.rows[end]
        push!(rows, Dict{String,Any}(
            "name"=>name, "best_mse"=>trained.best.mse, "best_accuracy"=>trained.best.acc,
            "best_epoch"=>trained.best.epoch, "final_mse"=>final["mse"], "final_accuracy"=>final["accuracy"],
            "beta"=>config.β, "temp_fraction"=>config.temp_fraction, "hidden_output_scale"=>config.hidden_output_scale,
            "outdir"=>outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved cold split-input scout: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_cold_scout()
end
