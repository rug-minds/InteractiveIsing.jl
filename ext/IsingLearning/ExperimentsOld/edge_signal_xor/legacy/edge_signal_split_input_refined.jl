include("edge_signal_split_input_search.jl")

"""
    refined_split_input_trials()

Return a focused follow-up grid for the split-edge XOR architecture.

The architecture remains exactly `2 input spins -> split halves of the hidden
left edge -> hidden right edge -> 1 scalar output`. These trials keep the best
previous neighborhood (`NN=5`) and vary only the strength/noise balance:
hidden-local scale, output-edge scale, beta, temperature, and bias.
"""
function refined_split_input_trials()
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
        input_hidden_scale = FT(0.45),
        skip_response = true,
    )

    return [
        EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00045), temp_fraction = FT(0.020),
            hidden_local_scale = FT(0.003), hidden_output_scale = FT(0.55), bias_scale = FT(0.0),
            weight_seed = 803, bias_seed = 823, base_seed = 1_080_803, base...),
        EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00045), temp_fraction = FT(0.018),
            hidden_local_scale = FT(0.003), hidden_output_scale = FT(0.70), bias_scale = FT(0.0),
            weight_seed = 804, bias_seed = 824, base_seed = 1_080_804, base...),
        EdgeSignalXORConfig(; β = FT(0.35), lr = FT(0.00035), temp_fraction = FT(0.020),
            hidden_local_scale = FT(0.003), hidden_output_scale = FT(0.55), bias_scale = FT(0.0),
            weight_seed = 805, bias_seed = 825, base_seed = 1_080_805, base...),
        EdgeSignalXORConfig(; β = FT(0.20), lr = FT(0.00040), temp_fraction = FT(0.025),
            hidden_local_scale = FT(0.0015), hidden_output_scale = FT(0.70), bias_scale = FT(0.0),
            weight_seed = 806, bias_seed = 826, base_seed = 1_080_806, base...),
        EdgeSignalXORConfig(; β = FT(0.15), lr = FT(0.00050), temp_fraction = FT(0.020),
            hidden_local_scale = FT(0.002), hidden_output_scale = FT(0.85), bias_scale = FT(0.0),
            weight_seed = 807, bias_seed = 827, base_seed = 1_080_807, base...),
    ]
end

"""Return a compact name for a refined split-input trial."""
function refined_split_input_name(config::EdgeSignalXORConfig)
    beta = replace(string(config.β), "." => "p")
    temp = replace(string(config.temp_fraction), "." => "p")
    hidden = replace(string(config.hidden_local_scale), "." => "p")
    output = replace(string(config.hidden_output_scale), "." => "p")
    return "NN$(config.hidden_nn)_b$(beta)_T$(temp)_hl$(hidden)_out$(output)"
end

"""Run the refined split-input search and write a summary table."""
function run_refined_split_input_search(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_refined_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for config in refined_split_input_trials()
        name = refined_split_input_name(config)
        println("\n=== ", name, " ===")
        outdir = joinpath(rootdir, name)
        mkpath(outdir)
        trained = train_split_input_edge_xor(config; outdir)
        final = isempty(trained.rows) ? nothing : trained.rows[end]
        push!(summary, Dict{String,Any}(
            "name" => name,
            "hidden_nn" => config.hidden_nn,
            "beta" => config.β,
            "temp_fraction" => config.temp_fraction,
            "input_hidden_scale" => config.input_hidden_scale,
            "hidden_local_scale" => config.hidden_local_scale,
            "hidden_output_scale" => config.hidden_output_scale,
            "bias_scale" => config.bias_scale,
            "lr" => config.lr,
            "best_epoch" => trained.best.epoch,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "final_mse" => final === nothing ? missing : final["mse"],
            "final_accuracy" => final === nothing ? missing : final["accuracy"],
            "outdir" => outdir,
        ))
        write_run_readme(joinpath(outdir, "README.md"), config, trained, nothing)
        push!(results, (; config, trained, outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), summary)
    write_refined_split_input_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved refined split-input edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write a markdown summary for the refined split-input search."""
function write_refined_split_input_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Refined Split-Input Edge XOR")
        println(io)
        println(io, "Architecture: `2 -> split first edge -> 8x8 hidden -> last edge -> 1`.")
        println(io, "Input spin 1 drives only the upper half of the first hidden edge; input spin 2 drives only the lower half.")
        println(io)
        println(io, "| trial | beta | T fraction | hidden scale | output scale | bias scale | lr | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            println(io, "| `$(row["name"])` | $(row["beta"]) | $(row["temp_fraction"]) | $(row["hidden_local_scale"]) | $(row["hidden_output_scale"]) | $(row["bias_scale"]) | $(row["lr"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_refined_split_input_search()
end
