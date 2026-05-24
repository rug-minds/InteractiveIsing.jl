using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "checkerboard_4x8_no_seed_grid.jl"))

"""
    tight_metropolis_4x8_searches()

Return a focused Metropolis-only search around the best corrected checkerboard
recipe found so far. The experiment keeps the full bipolar input code frozen,
keeps input-layer internal weights disabled, uses pattern output clamping, and
does not add polynomial or double-well local potentials.
"""
function tight_metropolis_4x8_searches()
    epochs = parse(Int, get(ENV, "ISING_4X8_TIGHT_EPOCHS", "1400"))
    log_every = parse(Int, get(ENV, "ISING_4X8_TIGHT_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_4X8_TIGHT_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_4X8_TIGHT_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_4X8_TIGHT_EVAL_REPEATS", "24"))

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
        base_seed = 101000,
        init_mode = :random,
        doublewell_barrier = FT(0),
        dynamics_mode = :metropolis,
        state_mode = :discrete,
        output_clamp_mode = :pattern,
    )

    specs = (
        (; name = "metro_T020_b05_lr003_rel300_J010", temp = FT(0.020), beta = FT(0.50), lr = FT(0.0030), free = 300, nudged = 300, inter = FT(0.10), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T025_b05_lr003_rel300_J010", temp = FT(0.025), beta = FT(0.50), lr = FT(0.0030), free = 300, nudged = 300, inter = FT(0.10), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T030_b05_lr003_rel300_J010", temp = FT(0.030), beta = FT(0.50), lr = FT(0.0030), free = 300, nudged = 300, inter = FT(0.10), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T025_b075_lr0025_rel300_J010", temp = FT(0.025), beta = FT(0.75), lr = FT(0.0025), free = 300, nudged = 300, inter = FT(0.10), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T025_b05_lr0025_rel500_J010", temp = FT(0.025), beta = FT(0.50), lr = FT(0.0025), free = 500, nudged = 500, inter = FT(0.10), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T025_b05_lr003_rel300_J012", temp = FT(0.025), beta = FT(0.50), lr = FT(0.0030), free = 300, nudged = 300, inter = FT(0.12), hidden = FT(0.06), output = FT(0.06)),
        (; name = "metro_T025_b05_lr003_rel300_internal008", temp = FT(0.025), beta = FT(0.50), lr = FT(0.0030), free = 300, nudged = 300, inter = FT(0.10), hidden = FT(0.08), output = FT(0.08)),
        (; name = "metro_T020_b075_lr0025_rel500_J012", temp = FT(0.020), beta = FT(0.75), lr = FT(0.0025), free = 500, nudged = 500, inter = FT(0.12), hidden = FT(0.06), output = FT(0.06)),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                temp = spec.temp,
                β = spec.beta,
                lr = spec.lr,
                free_relaxation = spec.free,
                nudged_relaxation = spec.nudged,
                inter_weight_scale = spec.inter,
                hidden_internal_scale = spec.hidden,
                output_internal_scale = spec.output,
            ),
            full_bipolar_input = true,
            input_kick = false,
            freeze_inactive_input = false,
            save_threshold = FT(0.18),
            notes = "tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=$(spec.temp), β=$(spec.beta), lr=$(spec.lr), relax=$(spec.free)/$(spec.nudged), J=$(spec.inter), hidden/output internal=$(spec.hidden)/$(spec.output)",
        )
        for spec in specs
    ]
end

"""
    run_tight_metropolis_config(search, outdir)

Run one focused Metropolis checkerboard configuration after checking that the
Hamiltonian contains no unwanted polynomial or double-well local potential.
"""
function run_tight_metropolis_config(search::StabilizedSearchConfig, outdir)
    graph = checkerboard_graph(search.config)
    assert_no_extra_local_potentials!(graph)
    return run_stabilized_config(search, outdir)
end

"""
    main()

Execute the focused Metropolis probe and write per-epoch metrics, a progress
plot, saved best graphs under the configured threshold, and a compact README.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_4X8_TIGHT_DIR", joinpath(DEFAULT_RUN_ROOT, "checkerboard_4x8_metropolis_tight_$timestamp"))
    searches = tight_metropolis_4x8_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running tight Metropolis 4x8 probe $idx/$(length(searches)): $(search.config.name)")
        result = run_tight_metropolis_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "checkerboard_4x8_metropolis_tight_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "checkerboard_4x8_metropolis_tight_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Checkerboard 4x8 Tight Metropolis Probe")
        println(io)
        println(io, "This run tightens the search around the corrected checkerboard recipe that first reached MSE near `0.13`: fully frozen bipolar input, no input-layer internal weights, pattern output clamping, no polynomial/double-well local potentials, symmetric Metropolis dynamics.")
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
