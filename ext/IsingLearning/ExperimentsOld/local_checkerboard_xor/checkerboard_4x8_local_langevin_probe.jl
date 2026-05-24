using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "checkerboard_4x8_no_seed_grid.jl"))

"""
    local_langevin_4x8_searches()

Build a small LocalLangevin-only probe for the explicit bipolar
`4x4 -> 8x8 -> 4x4` checkerboard setup. The input code is fully frozen and the
input layer has no internal weights, matching `checkerboard_4x8_no_seed_grid.jl`.
"""
function local_langevin_4x8_searches()
    epochs = parse(Int, get(ENV, "ISING_4X8_LOCAL_EPOCHS", "600"))
    log_every = parse(Int, get(ENV, "ISING_4X8_LOCAL_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_4X8_LOCAL_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_4X8_LOCAL_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_4X8_LOCAL_EVAL_REPEATS", "16"))

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
        base_seed = 99000,
        init_mode = :random,
        state_mode = :continuous,
        dynamics_mode = :local_langevin,
        output_clamp_mode = :pattern,
        doublewell_barrier = FT(0),
        inter_weight_scale = FT(0.10),
    )

    specs = (
        (; name = "local_T025_s005_b10_rel500_1000", temp = FT(0.025), stepsize = FT(0.05), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000),
        (; name = "local_T05_s005_b10_rel500_1000", temp = FT(0.05), stepsize = FT(0.05), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000),
        (; name = "local_T025_s010_b10_rel500_1000", temp = FT(0.025), stepsize = FT(0.10), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000),
        (; name = "local_T05_s010_b10_rel500_1000", temp = FT(0.05), stepsize = FT(0.10), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000),
        (; name = "local_T025_s005_b05_rel300_600", temp = FT(0.025), stepsize = FT(0.05), beta = FT(0.5), lr = FT(0.003), free = 300, nudged = 600),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                temp = spec.temp,
                stepsize = spec.stepsize,
                β = spec.beta,
                lr = spec.lr,
                free_relaxation = spec.free,
                nudged_relaxation = spec.nudged,
            ),
            full_bipolar_input = true,
            save_threshold = FT(0.25),
            notes = "LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=$(spec.temp), stepsize=$(spec.stepsize), β=$(spec.beta), relax=$(spec.free)/$(spec.nudged)",
        )
        for spec in specs
    ]
end

"""
    main()

Run the LocalLangevin-only no-seed checkerboard probe and save metrics, plot,
and run notes.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_4X8_LOCAL_DIR", joinpath(DEFAULT_RUN_ROOT, "checkerboard_4x8_local_langevin_$timestamp"))
    searches = local_langevin_4x8_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running LocalLangevin 4x8 probe $idx/$(length(searches)): $(search.config.name)")
        result = run_no_seed_4x8_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "checkerboard_4x8_local_langevin_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "checkerboard_4x8_local_langevin_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Checkerboard 4x8 LocalLangevin Probe")
        println(io)
        println(io, "Topology: `4x4 input -> 8x8 hidden -> 4x4 output`. Input is fully frozen bipolar checkerboard embedding; input layer has no internal weights; LocalLangevin uses `adjusted=false`; Hamiltonian terms are asserted to be only `Bilinear`, `MagField`, and output clamping.")
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
