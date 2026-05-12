using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    masked_clamp_beta_probe_searches()

Focused checkerboard XOR probes after adding a mask to the generic
`InteractiveIsing.Clamping`.

This is intentionally not a broad grid. It checks a few high-β continuous
Langevin points where the old full-graph clamping bug would have made larger β
dangerous by pulling every untargeted unit toward zero.
"""
function masked_clamp_beta_probe_searches()
    epochs = parse(Int, get(ENV, "ISING_MASKED_CLAMP_PROBE_EPOCHS", "600"))
    log_every = parse(Int, get(ENV, "ISING_MASKED_CLAMP_PROBE_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_MASKED_CLAMP_PROBE_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_MASKED_CLAMP_PROBE_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_MASKED_CLAMP_PROBE_EVAL_REPEATS", "16"))
    lr = parse(FT, get(ENV, "ISING_MASKED_CLAMP_PROBE_LR", "0.003"))

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
        lr,
        weight_decay = FT(0),
        grad_clip = FT(80),
        temp_is_factor = true,
        stepsize = FT(0.05),
        block_size = 4,
        inter_radius = sqrt(2.0) + 1e-6,
        internal_nn = 1,
        inter_weight_scale = FT(0.20),
        input_internal_scale = FT(0.10),
        hidden_internal_scale = FT(0.10),
        output_internal_scale = FT(0.10),
        bias_scale = FT(0.02),
        weight_seed = 13,
        internal_seed = 14,
        bias_seed = 22,
        base_seed = 93016,
        init_mode = :random,
        state_mode = :continuous,
        doublewell_barrier = FT(0),
    )

    specs = (
        (; name = "block_readout_b05_T02_r150_300", dynamics = :langevin, clamp = :readout, β = FT(0.5), temp = FT(0.02), free = 150, nudged = 300, lr = lr),
        (; name = "block_readout_b10_T02_r150_300", dynamics = :langevin, clamp = :readout, β = FT(1.0), temp = FT(0.02), free = 150, nudged = 300, lr = lr),
        (; name = "block_readout_b20_T02_r150_300", dynamics = :langevin, clamp = :readout, β = FT(2.0), temp = FT(0.02), free = 150, nudged = 300, lr = lr / 2),
        (; name = "block_readout_b10_T05_r150_300", dynamics = :langevin, clamp = :readout, β = FT(1.0), temp = FT(0.05), free = 150, nudged = 300, lr = lr),
        (; name = "block_pattern_b10_T02_r300_600", dynamics = :langevin, clamp = :pattern, β = FT(1.0), temp = FT(0.02), free = 300, nudged = 600, lr = lr),
        (; name = "global_readout_b10_T02_r150_300", dynamics = :global_langevin, clamp = :readout, β = FT(1.0), temp = FT(0.02), free = 150, nudged = 300, lr = lr),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                dynamics_mode = spec.dynamics,
                output_clamp_mode = spec.clamp,
                β = spec.β,
                temp = spec.temp,
                free_relaxation = spec.free,
                nudged_relaxation = spec.nudged,
                lr = spec.lr,
            ),
            input_kick = true,
            save_threshold = FT(0.3),
            notes = "masked-clamping beta probe: $(spec.dynamics), clamp=$(spec.clamp), β=$(spec.β), Tfactor=$(spec.temp), relax=$(spec.free)/$(spec.nudged)",
        )
        for spec in specs
    ]
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_MASKED_CLAMP_PROBE_DIR", joinpath(DEFAULT_RUN_ROOT, "masked_clamp_beta_probe_$timestamp"))
    searches = masked_clamp_beta_probe_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running masked clamp beta probe $idx/$(length(searches)): $(search.config.name)")
        result = run_stabilized_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "masked_clamp_beta_probe_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "masked_clamp_beta_probe_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Masked Clamping Beta Probe")
        println(io)
        println(io, "Focused checkerboard XOR probes after adding a clamping mask.")
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
