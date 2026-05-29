include("edge_signal_xor.jl")

"""
    LowBetaEdgeTrial

One concrete edge-signal XOR trial. This keeps the architecture fixed and only
changes the parameters that determine signal propagation and nudging strength.
"""
Base.@kwdef struct LowBetaEdgeTrial
    name::String
    hidden_nn::Int
    β::FT
    lr::FT
    stepsize::FT
    temp_fraction::FT
    input_hidden_scale::FT
    hidden_local_scale::FT
    hidden_output_scale::FT
    free_relaxation::Int
    nudged_relaxation::Int
    validation_relaxation::Int
    minit::Int
    eval_repeats::Int
    epochs::Int
    log_every::Int
    weight_seed::Int
end

"""Convert a low-beta search row into the existing edge experiment config."""
function edge_config(trial::LowBetaEdgeTrial)
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
        weight_decay = zero(FT),
        stepsize = trial.stepsize,
        temp_fraction = trial.temp_fraction,
        input_hidden_scale = trial.input_hidden_scale,
        hidden_local_scale = trial.hidden_local_scale,
        hidden_output_scale = trial.hidden_output_scale,
        bias_scale = FT(0.02),
        hidden_nn = trial.hidden_nn,
        target_scale = one(FT),
        skip_response = true,
        weight_seed = trial.weight_seed,
        bias_seed = trial.weight_seed + 17,
        base_seed = 610_000 + 1000 * trial.weight_seed,
    )
end

"""Return a compact set of trials centered on the scalar XOR β≈0.2 result."""
function lowbeta_edge_trials()
    base = (;
        epochs = 2400,
        log_every = 240,
        minit = 2,
        eval_repeats = 12,
        free_relaxation = 700,
        nudged_relaxation = 700,
        validation_relaxation = 1400,
        stepsize = FT(0.8),
    )
    return LowBetaEdgeTrial[
        LowBetaEdgeTrial(; name = "NN1_beta0p20_T0p05_edge0p20_h0p020", hidden_nn = 1,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.05),
            input_hidden_scale = FT(0.20), hidden_local_scale = FT(0.020), hidden_output_scale = FT(0.20),
            weight_seed = 101, base...),
        LowBetaEdgeTrial(; name = "NN2_beta0p20_T0p05_edge0p20_h0p015", hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.05),
            input_hidden_scale = FT(0.20), hidden_local_scale = FT(0.015), hidden_output_scale = FT(0.20),
            weight_seed = 102, base...),
        LowBetaEdgeTrial(; name = "NN2_beta0p35_T0p05_edge0p20_h0p015", hidden_nn = 2,
            β = FT(0.35), lr = FT(0.0006), temp_fraction = FT(0.05),
            input_hidden_scale = FT(0.20), hidden_local_scale = FT(0.015), hidden_output_scale = FT(0.20),
            weight_seed = 103, base...),
        LowBetaEdgeTrial(; name = "NN2_beta0p20_T0p035_edge0p25_h0p012", hidden_nn = 2,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.012), hidden_output_scale = FT(0.25),
            weight_seed = 104, base...),
        LowBetaEdgeTrial(; name = "NN3_beta0p20_T0p05_edge0p25_h0p010", hidden_nn = 3,
            β = FT(0.20), lr = FT(0.0007), temp_fraction = FT(0.05),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.010), hidden_output_scale = FT(0.25),
            weight_seed = 105, base...),
        LowBetaEdgeTrial(; name = "NN3_beta0p35_T0p035_edge0p25_h0p010", hidden_nn = 3,
            β = FT(0.35), lr = FT(0.0006), temp_fraction = FT(0.035),
            input_hidden_scale = FT(0.25), hidden_local_scale = FT(0.010), hidden_output_scale = FT(0.25),
            weight_seed = 106, base...),
    ]
end

"""Return the last row of a learning-metric row vector."""
last_metric(rows) = isempty(rows) ? nothing : rows[end]

"""Run all low-beta edge trials and save a summary CSV."""
function run_lowbeta_edge_search(; rootdir = joinpath(@__DIR__, "runs", "edge_lowbeta_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    summary = Dict{String,Any}[]
    results = []
    for trial in lowbeta_edge_trials()
        println("\n=== ", trial.name, " ===")
        config = edge_config(trial)
        outdir = joinpath(rootdir, trial.name)
        mkpath(outdir)
        trained = train_edge_xor(config; outdir)
        final = last_metric(trained.rows)
        push!(summary, Dict{String,Any}(
            "name" => trial.name,
            "hidden_nn" => trial.hidden_nn,
            "beta" => trial.β,
            "lr" => trial.lr,
            "stepsize" => trial.stepsize,
            "temp_fraction" => trial.temp_fraction,
            "input_hidden_scale" => trial.input_hidden_scale,
            "hidden_local_scale" => trial.hidden_local_scale,
            "hidden_output_scale" => trial.hidden_output_scale,
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
    write_lowbeta_summary(joinpath(rootdir, "README.md"), summary)
    println("Saved low-beta edge search: ", rootdir)
    return (; rootdir, summary, results)
end

"""Write a short markdown summary for the low-beta edge search."""
function write_lowbeta_summary(path, summary)
    sorted = sort(summary; by = row -> (row["best_accuracy"], -row["best_mse"]), rev = true)
    open(path, "w") do io
        println(io, "# Low-Beta Edge Signal XOR Search")
        println(io)
        println(io, "This run keeps the edge architecture fixed and tests the scalar-XOR lesson that full `±1` targets need a smaller direct clamping `β` than the old edge runs used.")
        println(io)
        println(io, "| trial | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted
            println(io, "| `$(row["name"])` | $(row["hidden_nn"]) | $(row["beta"]) | $(row["temp_fraction"]) | $(row["input_hidden_scale"]) | $(row["hidden_local_scale"]) | $(round(row["best_mse"], digits = 6)) | $(row["best_accuracy"]) | $(row["best_epoch"]) |")
        end
        println(io)
        println(io, "The success criterion is accuracy `1.0` and scalar-output MSE below `0.1` from repeated validation starts.")
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lowbeta_edge_search()
end
