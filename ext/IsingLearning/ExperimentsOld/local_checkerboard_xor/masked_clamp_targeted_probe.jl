using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    targeted_masked_clamp_searches()

Small follow-up after `masked_clamp_beta_probe.jl`.

The first masked-clamping probe showed that readout clamping produced tiny
score margins, while output-pattern clamping produced the strongest movement.
This file therefore probes only a few nearby points: stronger `β`, longer
nudged relaxation, slightly different temperature factors, and one global
Langevin comparison.
"""
function targeted_masked_clamp_searches()
    epochs = parse(Int, get(ENV, "ISING_MASKED_TARGET_EPOCHS", "800"))
    log_every = parse(Int, get(ENV, "ISING_MASKED_TARGET_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_MASKED_TARGET_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_MASKED_TARGET_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_MASKED_TARGET_EVAL_REPEATS", "16"))

    common = (;
        side = 2,
        hidden_side = 2,
        code_side = 2,
        code_stride = 1,
        code_offset = (1, 1),
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        weight_decay = FT(0),
        grad_clip = FT(80),
        temp_is_factor = true,
        stepsize = FT(0.05),
        block_size = 4,
        inter_radius = sqrt(2.0) + 1e-6,
        internal_nn = 1,
        input_internal_scale = FT(0.10),
        hidden_internal_scale = FT(0.10),
        output_internal_scale = FT(0.10),
        bias_scale = FT(0.02),
        weight_seed = 13,
        internal_seed = 14,
        bias_seed = 22,
        base_seed = 94016,
        init_mode = :random,
        state_mode = :continuous,
        doublewell_barrier = FT(0),
        output_clamp_mode = :pattern,
    )

    specs = (
        (; name = "block_pattern_b15_T02_r300_900", dynamics = :langevin, β = FT(1.5), temp = FT(0.02), free = 300, nudged = 900, lr = FT(0.002), inter = FT(0.20)),
        (; name = "block_pattern_b20_T02_r300_900", dynamics = :langevin, β = FT(2.0), temp = FT(0.02), free = 300, nudged = 900, lr = FT(0.0015), inter = FT(0.20)),
        (; name = "block_pattern_b15_T03_r300_900", dynamics = :langevin, β = FT(1.5), temp = FT(0.03), free = 300, nudged = 900, lr = FT(0.002), inter = FT(0.20)),
        (; name = "block_pattern_b10_T015_r500_1000", dynamics = :langevin, β = FT(1.0), temp = FT(0.015), free = 500, nudged = 1000, lr = FT(0.002), inter = FT(0.20)),
        (; name = "block_pattern_b15_T02_J030_r300_900", dynamics = :langevin, β = FT(1.5), temp = FT(0.02), free = 300, nudged = 900, lr = FT(0.0015), inter = FT(0.30)),
        (; name = "global_pattern_b15_T02_r300_900", dynamics = :global_langevin, β = FT(1.5), temp = FT(0.02), free = 300, nudged = 900, lr = FT(0.002), inter = FT(0.20)),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                dynamics_mode = spec.dynamics,
                β = spec.β,
                temp = spec.temp,
                free_relaxation = spec.free,
                nudged_relaxation = spec.nudged,
                lr = spec.lr,
                inter_weight_scale = spec.inter,
            ),
            input_kick = true,
            save_threshold = FT(0.3),
            notes = "targeted masked clamp follow-up: $(spec.dynamics), output-pattern clamp, β=$(spec.β), Tfactor=$(spec.temp), relax=$(spec.free)/$(spec.nudged), J=$(spec.inter)",
        )
        for spec in specs
    ]
end

"""
    main()

Run the targeted masked-clamp checkerboard probes, then save a CSV, progress
plot, and short run README into one timestamped experiment folder.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_MASKED_TARGET_DIR", joinpath(DEFAULT_RUN_ROOT, "masked_clamp_targeted_probe_$timestamp"))
    searches = targeted_masked_clamp_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running targeted masked clamp probe $idx/$(length(searches)): $(search.config.name)")
        result = run_stabilized_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "masked_clamp_targeted_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "masked_clamp_targeted_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Targeted Masked-Clamp Checkerboard Probe")
        println(io)
        println(io, "Follow-up to the first masked-clamp beta probe. The first probe favored output-pattern clamping over scalar readout clamping, so this run concentrates on high-β output-pattern Langevin points.")
        println(io)
        println(io, "| Config | Best MSE | Best Acc | Saved | Notes |")
        println(io, "|---|---:|---:|---|---|")
        for (search, result) in zip(searches, results)
            saved = isnothing(result.graph_path) ? "no" : "yes"
            println(io, "| `$(search.config.name)` | $(round(result.best_mse, digits=6)) | $(round(result.best_acc, digits=3)) | $saved | $(search.notes) |")
        end
        println(io)
        println(io, "Metrics CSV: `$(basename(csv_path))`")
        println(io, "Progress PNG: `$(basename(png_path))`")
    end
    println("Saved metrics: $csv_path")
    println("Saved plot: $png_path")
    println("Saved docs: $md_path")
    return (; outdir, searches, results, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
