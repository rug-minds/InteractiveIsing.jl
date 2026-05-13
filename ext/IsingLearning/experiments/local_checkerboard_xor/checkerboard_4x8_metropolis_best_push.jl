using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "checkerboard_4x8_metropolis_tight_probe.jl"))

"""
    best_push_4x8_searches()

Return a small follow-up search around the best corrected checkerboard
Metropolis recipe. This file intentionally stays narrow: the previous tight
probe found that the useful region is stronger inter-layer coupling, moderate
temperature, stronger pattern clamping, and longer free/nudged relaxation.
"""
function best_push_4x8_searches()
    epochs = parse(Int, get(ENV, "ISING_4X8_PUSH_EPOCHS", "2600"))
    log_every = parse(Int, get(ENV, "ISING_4X8_PUSH_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_4X8_PUSH_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_4X8_PUSH_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_4X8_PUSH_EVAL_REPEATS", "24"))

    common = (;
        side = 4,
        hidden_side = 8,
        code_side = 4,
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
        inter_radius = FT(2.25),
        internal_nn = 1,
        input_internal_scale = FT(0),
        hidden_internal_scale = FT(0.06),
        output_internal_scale = FT(0.06),
        bias_scale = FT(0.02),
        weight_seed = 31,
        internal_seed = 37,
        bias_seed = 41,
        base_seed = 104000,
        init_mode = :random,
        doublewell_barrier = FT(0),
        dynamics_mode = :metropolis,
        state_mode = :discrete,
        output_clamp_mode = :pattern,
        inter_weight_scale = FT(0.12),
        free_relaxation = 500,
        nudged_relaxation = 500,
    )

    specs = (
        (; name = "push_T020_b075_lr0025_J012", temp = FT(0.020), beta = FT(0.75), lr = FT(0.0025), inter = FT(0.12), relax = 500),
        (; name = "push_T018_b075_lr0025_J012", temp = FT(0.018), beta = FT(0.75), lr = FT(0.0025), inter = FT(0.12), relax = 500),
        (; name = "push_T020_b100_lr0020_J012", temp = FT(0.020), beta = FT(1.00), lr = FT(0.0020), inter = FT(0.12), relax = 500),
        (; name = "push_T020_b075_lr0020_J014", temp = FT(0.020), beta = FT(0.75), lr = FT(0.0020), inter = FT(0.14), relax = 500),
        (; name = "push_T020_b075_lr0020_J012_rel700", temp = FT(0.020), beta = FT(0.75), lr = FT(0.0020), inter = FT(0.12), relax = 700),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                temp = spec.temp,
                β = spec.beta,
                lr = spec.lr,
                inter_weight_scale = spec.inter,
                free_relaxation = spec.relax,
                nudged_relaxation = spec.relax,
            ),
            full_bipolar_input = true,
            input_kick = false,
            freeze_inactive_input = false,
            save_threshold = FT(0.13),
            notes = "best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=$(spec.temp), β=$(spec.beta), lr=$(spec.lr), relax=$(spec.relax)/$(spec.relax), J=$(spec.inter)",
        )
        for spec in specs
    ]
end

"""
    main()

Run the narrow best-push search and save metrics, plot, README, and any graph
whose best MSE crosses the configured threshold.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_4X8_PUSH_DIR", joinpath(DEFAULT_RUN_ROOT, "checkerboard_4x8_metropolis_best_push_$timestamp"))
    searches = best_push_4x8_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running best-push Metropolis 4x8 $idx/$(length(searches)): $(search.config.name)")
        result = run_tight_metropolis_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "checkerboard_4x8_metropolis_best_push_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "checkerboard_4x8_metropolis_best_push_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Checkerboard 4x8 Metropolis Best Push")
        println(io)
        println(io, "Narrow follow-up around the first corrected checkerboard run that approached `0.1` MSE. All configs use fully frozen bipolar input, no input internal weights, pattern output clamping, no polynomial/double-well local potentials, and symmetric Metropolis dynamics.")
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
