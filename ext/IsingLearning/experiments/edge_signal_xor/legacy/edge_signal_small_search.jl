include("edge_signal_xor.jl")

"""
    SmallEdgeTrial

One compact edge-signal XOR trial. These runs shorten the hidden propagation
path first, so failure is less likely to be caused by an 8-column diffusion
problem.
"""
Base.@kwdef struct SmallEdgeTrial
    name::String
    hidden_height::Int
    hidden_width::Int
    hidden_nn::Int
    β::FT
    lr::FT
    stepsize::FT
    temp_fraction::FT
    edge_scale::FT
    hidden_scale::FT
    free_relaxation::Int
    nudged_relaxation::Int
    validation_relaxation::Int
    minit::Int
    eval_repeats::Int
    epochs::Int
    log_every::Int
    weight_seed::Int
end

"""Build the standard edge config for one compact-trial row."""
function edge_config(trial::SmallEdgeTrial)
    return EdgeSignalXORConfig(
        epochs = trial.epochs,
        log_every = trial.log_every,
        minit = trial.minit,
        eval_repeats = trial.eval_repeats,
        free_relaxation = trial.free_relaxation,
        nudged_relaxation = trial.nudged_relaxation,
        validation_relaxation = trial.validation_relaxation,
        β = trial.β,
        lr = trial.lr,
        stepsize = trial.stepsize,
        temp_fraction = trial.temp_fraction,
        input_hidden_scale = trial.edge_scale,
        hidden_local_scale = trial.hidden_scale,
        hidden_output_scale = trial.edge_scale,
        bias_scale = FT(0.02),
        hidden_height = trial.hidden_height,
        hidden_width = trial.hidden_width,
        hidden_nn = trial.hidden_nn,
        target_scale = one(FT),
        skip_response = true,
        weight_seed = trial.weight_seed,
        bias_seed = trial.weight_seed + 19,
        base_seed = 720_000 + 1000 * trial.weight_seed,
    )
end

"""Return small hidden-layer edge trials ordered from shortest to longer paths."""
function small_edge_trials()
    base = (;
        epochs = 3000,
        log_every = 300,
        minit = 2,
        eval_repeats = 16,
        free_relaxation = 900,
        nudged_relaxation = 900,
        validation_relaxation = 1800,
        stepsize = FT(0.8),
    )
    return SmallEdgeTrial[
        SmallEdgeTrial(; name = "H2x2_NN1_beta0p20_T0p04_edge0p35_h0p030",
            hidden_height = 2, hidden_width = 2, hidden_nn = 1, β = FT(0.20), lr = FT(0.0008),
            temp_fraction = FT(0.04), edge_scale = FT(0.35), hidden_scale = FT(0.030),
            weight_seed = 201, base...),
        SmallEdgeTrial(; name = "H2x2_NN1_beta0p35_T0p04_edge0p35_h0p030",
            hidden_height = 2, hidden_width = 2, hidden_nn = 1, β = FT(0.35), lr = FT(0.0007),
            temp_fraction = FT(0.04), edge_scale = FT(0.35), hidden_scale = FT(0.030),
            weight_seed = 202, base...),
        SmallEdgeTrial(; name = "H4x4_NN1_beta0p20_T0p035_edge0p35_h0p025",
            hidden_height = 4, hidden_width = 4, hidden_nn = 1, β = FT(0.20), lr = FT(0.0008),
            temp_fraction = FT(0.035), edge_scale = FT(0.35), hidden_scale = FT(0.025),
            weight_seed = 203, base...),
        SmallEdgeTrial(; name = "H4x4_NN2_beta0p20_T0p035_edge0p40_h0p018",
            hidden_height = 4, hidden_width = 4, hidden_nn = 2, β = FT(0.20), lr = FT(0.0008),
            temp_fraction = FT(0.035), edge_scale = FT(0.40), hidden_scale = FT(0.018),
            weight_seed = 204, base...),
        SmallEdgeTrial(; name = "H4x4_NN2_beta0p35_T0p025_edge0p40_h0p018",
            hidden_height = 4, hidden_width = 4, hidden_nn = 2, β = FT(0.35), lr = FT(0.0007),
            temp_fraction = FT(0.025), edge_scale = FT(0.40), hidden_scale = FT(0.018),
            weight_seed = 205, base...),
    ]
end

"""Run the compact edge search and write a summary table."""
function run_small_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_small_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for trial in small_edge_trials()
        println("\n=== ", trial.name, " ===")
        config = edge_config(trial)
        outdir = joinpath(rootdir, trial.name)
        mkpath(outdir)
        trained = train_edge_xor(config; outdir)
        final = isempty(trained.rows) ? nothing : trained.rows[end]
        push!(summary, Dict{String,Any}(
            "name" => trial.name,
            "hidden_height" => trial.hidden_height,
            "hidden_width" => trial.hidden_width,
            "hidden_nn" => trial.hidden_nn,
            "beta" => trial.β,
            "temp_fraction" => trial.temp_fraction,
            "edge_scale" => trial.edge_scale,
            "hidden_scale" => trial.hidden_scale,
            "best_epoch" => trained.best.epoch,
            "best_mse" => trained.best.mse,
            "best_accuracy" => trained.best.acc,
            "final_mse" => final === nothing ? missing : final["mse"],
            "final_accuracy" => final === nothing ? missing : final["accuracy"],
            "outdir" => outdir,
        ))
        write_run_readme(joinpath(outdir, "README.md"), config, trained, nothing)
        push!(results, (; trial, config, trained, outdir))
    end
    write_csv(joinpath(rootdir, "summary.csv"), summary)
    write_small_edge_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved compact edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write the compact edge search markdown summary."""
function write_small_edge_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Compact Edge Signal XOR Search")
        println(io)
        println(io, "This search uses the same input-left-edge and output-right-edge scheme as the 8x8 edge experiment, but starts with 2x2 and 4x4 hidden layers.")
        println(io)
        println(io, "| trial | hidden | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            hidden = string(row["hidden_height"], "x", row["hidden_width"])
            println(io, "| `$(row["name"])` | $(hidden) | $(row["hidden_nn"]) | $(row["beta"]) | $(row["temp_fraction"]) | $(row["edge_scale"]) | $(row["hidden_scale"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_small_edge_search()
end
