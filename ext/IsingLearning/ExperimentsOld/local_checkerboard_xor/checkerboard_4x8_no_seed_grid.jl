using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    assert_no_extra_local_potentials!(graph)

Check that the no-seed 4x8 checkerboard grid uses only the intended energy
terms: `Bilinear`, trainable `MagField`, and one output clamping term. `MagField`
is local but explicitly allowed here; polynomial/double-well local potentials
are not.
"""
function assert_no_extra_local_potentials!(graph)
    allowed = (II.Bilinear, II.MagField, IsingLearning.LinearReadoutClamping, MultiLinearReadoutClamping, OutputPatternClamping)
    for hterm in II.hamiltonians(graph.hamiltonian)
        any(T -> hterm isa T, allowed) ||
            error("unexpected Hamiltonian term in no-seed grid: $(typeof(hterm))")
    end
    return graph
end

"""
    no_seed_4x8_searches()

Return a compact grid for the local checkerboard XOR problem with topology
`4x4 input -> 8x8 hidden -> 4x4 output`. Hidden and output layers have learned
internal local interactions. The input layer has no internal weights because the
full input code is externally fixed. There is no explicit XOR feature seed and
no local potential term besides the allowed trainable magnetic field.

Input bits use the bipolar fixed-input encoding for this grid: bit value `0`
sets its checkerboard mask to `-1`, bit value `1` sets it to `+1`, and the full
input code is removed from the sampler in both free and nudged phases.
"""
function no_seed_4x8_searches()
    epochs = parse(Int, get(ENV, "ISING_4X8_NOSEED_EPOCHS", "900"))
    log_every = parse(Int, get(ENV, "ISING_4X8_NOSEED_LOG_EVERY", "100"))
    workers = parse(Int, get(ENV, "ISING_4X8_NOSEED_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_4X8_NOSEED_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_4X8_NOSEED_EVAL_REPEATS", "16"))

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
        base_seed = 97000,
        init_mode = :random,
        doublewell_barrier = FT(0),
    )

    specs = (
        (; name = "metro_pattern_T015_b05_rel300", dynamics = :metropolis, state = :discrete, clamp = :pattern, temp = FT(0.015), beta = FT(0.5), lr = FT(0.003), free = 300, nudged = 300, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "metro_pattern_T025_b05_rel300", dynamics = :metropolis, state = :discrete, clamp = :pattern, temp = FT(0.025), beta = FT(0.5), lr = FT(0.003), free = 300, nudged = 300, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "metro_tworeadout_T015_b03_rel300", dynamics = :metropolis, state = :discrete, clamp = :two_readout, temp = FT(0.015), beta = FT(0.3), lr = FT(0.003), free = 300, nudged = 300, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "metro_pattern_nn2_T015_b05_rel300", dynamics = :metropolis, state = :discrete, clamp = :pattern, temp = FT(0.015), beta = FT(0.5), lr = FT(0.003), free = 300, nudged = 300, inter = FT(0.10), nn = 2, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "block_pattern_T015_b10_rel500_1000", dynamics = :langevin, state = :continuous, clamp = :pattern, temp = FT(0.015), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "block_pattern_T03_b10_rel500_1000", dynamics = :langevin, state = :continuous, clamp = :pattern, temp = FT(0.03), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "global_pattern_T015_b10_rel500_1000", dynamics = :global_langevin, state = :continuous, clamp = :pattern, temp = FT(0.015), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "global_pattern_T025_b10_rel500_1000", dynamics = :global_langevin, state = :continuous, clamp = :pattern, temp = FT(0.025), beta = FT(1.0), lr = FT(0.002), free = 500, nudged = 1000, inter = FT(0.10), nn = 1, stepsize = FT(0.05), radius = FT(2.25)),
        (; name = "global_pattern_wide_T015_b10_rel500_1000", dynamics = :global_langevin, state = :continuous, clamp = :pattern, temp = FT(0.015), beta = FT(1.0), lr = FT(0.0015), free = 500, nudged = 1000, inter = FT(0.08), nn = 2, stepsize = FT(0.05), radius = FT(3.05)),
    )

    return [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = spec.name,
                dynamics_mode = spec.dynamics,
                state_mode = spec.state,
                output_clamp_mode = spec.clamp,
                temp = spec.temp,
                β = spec.beta,
                lr = spec.lr,
                free_relaxation = spec.free,
                nudged_relaxation = spec.nudged,
                inter_weight_scale = spec.inter,
                internal_nn = spec.nn,
                stepsize = spec.stepsize,
                inter_radius = spec.radius,
            ),
            full_bipolar_input = true,
            input_kick = false,
            freeze_inactive_input = false,
            save_threshold = FT(0.25),
            notes = "no-seed 4x8 grid with fully frozen bipolar input: $(spec.dynamics), clamp=$(spec.clamp), state=$(spec.state), Tfactor=$(spec.temp), β=$(spec.beta), relax=$(spec.free)/$(spec.nudged), J=$(spec.inter), NN=$(spec.nn), radius=$(spec.radius)",
        )
        for spec in specs
    ]
end

"""
    run_no_seed_4x8_config(search, outdir)

Run one no-seed 4x8 configuration after asserting that the graph has no
unwanted local-potential Hamiltonian terms.
"""
function run_no_seed_4x8_config(search::StabilizedSearchConfig, outdir)
    graph = checkerboard_graph(search.config)
    assert_no_extra_local_potentials!(graph)
    return run_stabilized_config(search, outdir)
end

"""
    main()

Run the no-seed `4x4 -> 8x8 -> 4x4` checkerboard grid and write metrics, plots,
and a short README into a run folder.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_4X8_NOSEED_DIR", joinpath(DEFAULT_RUN_ROOT, "checkerboard_4x8_no_seed_$timestamp"))
    searches = no_seed_4x8_searches()
    all_rows = Dict{String,Any}[]
    results = []

    for (idx, search) in enumerate(searches)
        println("Running no-seed 4x8 grid $idx/$(length(searches)): $(search.config.name)")
        result = run_no_seed_4x8_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end

    csv_path = write_csv(joinpath(outdir, "checkerboard_4x8_no_seed_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "checkerboard_4x8_no_seed_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Checkerboard 4x8 No-Seed Grid")
        println(io)
        println(io, "Topology: `4x4 input -> 8x8 hidden -> 4x4 output`. Hidden and output layers have local internal interactions. The input layer has no internal weights because the full input code is externally fixed. No explicit XOR feature seed and no polynomial/double-well local potential are used.")
        println(io)
        println(io, "Input encoding is fully clamped and bipolar: each checkerboard input mask is fixed to `-1` for bit `0` and `+1` for bit `1`, so the complete input code is held fixed in free and nudged phases.")
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
