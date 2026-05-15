include("edge_signal_split_input_regularized.jl")

"""Return the number of active non-input spins for the split-edge graph."""
function split_input_active_count(config::EdgeSignalXORConfig)
    graph = split_input_edge_graph(config)
    II.off!(graph.index_set, 1)
    return length(II.sampling_indices(graph.index_set))
end

"""
    split_input_config_from_sweeps(; kwargs...)

Build an `EdgeSignalXORConfig` using full-sweep counts instead of raw
single-spin update counts.

For this edge task, information is injected only at the first hidden edge and
must propagate through the 8x8 layer before the output sees it. Expressing
relaxation in sweeps makes that propagation requirement explicit.
"""
function split_input_config_from_sweeps(;
    epochs::Int,
    log_every::Int,
    minit::Int,
    eval_repeats::Int,
    free_sweeps::Int,
    nudged_sweeps::Int,
    validation_sweeps::Int,
    β::FT,
    lr::FT,
    weight_decay::FT,
    nudged_temp_factor::FT,
    base_seed::Int,
)
    template = EdgeSignalXORConfig(;
        hidden_height = 8,
        hidden_width = 8,
        hidden_nn = 5,
        input_hidden_scale = FT(0.45),
        hidden_local_scale = FT(0.005),
        hidden_output_scale = FT(0.40),
        bias_scale = FT(0.0),
        stepsize = FT(0.8),
        temp_fraction = FT(0.018),
        weight_seed = 703,
        bias_seed = 723,
        skip_response = true,
    )
    active_count = split_input_active_count(template)
    config = EdgeSignalXORConfig(;
        epochs,
        log_every,
        minit,
        eval_repeats,
        free_relaxation = free_sweeps * active_count,
        nudged_relaxation = nudged_sweeps * active_count,
        validation_relaxation = validation_sweeps * active_count,
        β,
        lr,
        weight_decay,
        stepsize = template.stepsize,
        temp_fraction = template.temp_fraction,
        input_hidden_scale = template.input_hidden_scale,
        hidden_local_scale = template.hidden_local_scale,
        hidden_output_scale = template.hidden_output_scale,
        bias_scale = template.bias_scale,
        hidden_height = template.hidden_height,
        hidden_width = template.hidden_width,
        hidden_nn = template.hidden_nn,
        target_scale = one(FT),
        skip_response = true,
        weight_seed = template.weight_seed,
        bias_seed = template.bias_seed,
        base_seed,
    )
    return (; config, active_count, nudged_temp_factor, free_sweeps, nudged_sweeps, validation_sweeps)
end

"""Return long-relaxation split-input runs in true sweep units."""
function split_input_sweep_relax_trials()
    return [
        ("sweeps80_80_val160", split_input_config_from_sweeps(;
            epochs = 20_000,
            log_every = 1_000,
            minit = 3,
            eval_repeats = 16,
            free_sweeps = 80,
            nudged_sweeps = 80,
            validation_sweeps = 160,
            β = FT(0.20),
            lr = FT(0.00035),
            weight_decay = FT(0.003),
            nudged_temp_factor = FT(1.10),
            base_seed = 1_130_703,
        )),
        ("sweeps120_120_val240", split_input_config_from_sweeps(;
            epochs = 20_000,
            log_every = 1_000,
            minit = 3,
            eval_repeats = 16,
            free_sweeps = 120,
            nudged_sweeps = 120,
            validation_sweeps = 240,
            β = FT(0.20),
            lr = FT(0.00030),
            weight_decay = FT(0.003),
            nudged_temp_factor = FT(1.10),
            base_seed = 1_131_703,
        )),
    ]
end

"""Run split-input edge learning with relaxation specified in full sweeps."""
function run_split_input_sweep_relax(; rootdir = joinpath(@__DIR__, "runs", "edge_split_input_sweep_relax_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(rootdir)
    rows = Dict{String,Any}[]
    for (name, spec) in split_input_sweep_relax_trials()
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
            "weight_decay" => config.weight_decay,
            "nudged_temp_factor" => spec.nudged_temp_factor,
            "outdir" => outdir,
        ))
    end
    write_csv(joinpath(rootdir, "summary.csv"), rows)
    println("Saved sweep-relax split-input search: ", rootdir)
    return rootdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_split_input_sweep_relax()
end
