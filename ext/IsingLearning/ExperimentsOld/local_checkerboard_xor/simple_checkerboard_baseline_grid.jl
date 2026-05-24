using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    simple_baseline_searches(; kwargs...)

Return a compact grid for the plain local checkerboard task.

This file is intentionally experiment-local. It keeps the shared toolbox
unchanged and reuses the stabilized no-anneal worker path from
`local_checkerboard_stabilized_search.jl`.

The grid focuses on the simplest question: can a local checkerboard graph with
same-layer connections learn XOR from random initial states?
"""
function simple_baseline_searches(;
    epochs,
    log_every,
    minit,
    eval_repeats,
    workers,
    free_relaxation,
    nudged_relaxation,
)
    common = (;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation,
        nudged_relaxation,
        weight_decay = FT(0),
        grad_clip = FT(20),
        bias_scale = FT(0.02),
        init_mode = Symbol(get(ENV, "ISING_SIMPLE_CHECKER_INIT", "random")),
        dynamics_mode = Symbol(get(ENV, "ISING_SIMPLE_CHECKER_DYNAMICS", "metropolis")),
        state_mode = Symbol(get(ENV, "ISING_SIMPLE_CHECKER_STATE", "discrete")),
        output_clamp_mode = :readout,
        doublewell_barrier = FT(0),
        temp_is_factor = true,
        stepsize = parse(FT, get(ENV, "ISING_SIMPLE_CHECKER_STEPSIZE", "0.05")),
    )

    configs = StabilizedSearchConfig[]
    idx = 0

    # These values are deliberately centered on the earlier best true-symmetric
    # window, but do not use the invalid discrete zero-init recipe.
    side_specs = (
        (2, 2, 2, 1, FT(sqrt(2) + 1e-6), 4, "2x2"),
        (4, 4, 4, 1, FT(sqrt(2) + 1e-6), 8, "4x4"),
        (4, 6, 4, 1, FT(2.05), 8, "4x4_h6"),
    )
    nn_specs = (1, 2)
    temp_factors = (FT(0.005), FT(0.01), FT(0.02), FT(0.04))
    inter_scales = (FT(0.08), FT(0.12), FT(0.20), FT(0.30))
    beta_lr_specs = ((FT(0.1), FT(0.003)), (FT(0.2), FT(0.003)), (FT(0.2), FT(0.006)))
    internal_scales = (FT(0.05), FT(0.10))
    bias_scales = (FT(0.02), FT(0.10), FT(0.20))

    for (side, hidden_side, code_side, code_stride, radius, block_size, label) in side_specs
        for nn in nn_specs
            for internal_scale in internal_scales
                for temp_factor in temp_factors
                    for inter_scale in inter_scales
                        for (beta, lr) in beta_lr_specs
                            for bias_scale in bias_scales
                                idx += 1
                                seed = 10_000 + 31 * idx
                                name = "simple_$(label)_nn$(nn)_is$(internal_scale)_bs$(bias_scale)_Tf$(temp_factor)_J$(inter_scale)_b$(beta)_lr$(lr)"
                                cfg = LocalCheckerboardConfig(;
                                    common...,
                                    name,
                                    side,
                                    hidden_side,
                                    code_side,
                                    code_stride,
                                    code_offset = (1, 1),
                                    block_size,
                                    inter_radius = radius,
                                    internal_nn = nn,
                                    temp = temp_factor,
                                    β = beta,
                                    lr,
                                    inter_weight_scale = inter_scale,
                                    input_internal_scale = internal_scale,
                                    hidden_internal_scale = internal_scale,
                                    output_internal_scale = internal_scale,
                                    bias_scale,
                                    weight_seed = seed,
                                    internal_seed = seed + 1,
                                    bias_seed = seed + 2,
                                    base_seed = 500_000 + 10_000 * idx,
                                )
                                push!(configs, StabilizedSearchConfig(
                                    config = cfg,
                                    save_threshold = FT(0.2),
                                    notes = "plain local checkerboard baseline grid: random init, symmetric weights, in-layer NN=$nn, bias_scale=$bias_scale",
                                ))
                            end
                        end
                    end
                end
            end
        end
    end

    return configs
end

function parse_names(names::String)
    wanted = strip.(split(names, ","))
    return filter(!isempty, wanted)
end

function write_simple_grid_summary(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "rank,config,best_acc,best_mse,saved,notes")
        for (rank, row) in enumerate(rows)
            println(io, join((
                rank,
                row.config,
                row.best_acc,
                row.best_mse,
                row.saved,
                row.notes,
            ), ","))
        end
    end
    return path
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_SIMPLE_CHECKER_GRID_DIR", joinpath(DEFAULT_RUN_ROOT, "simple_checkerboard_grid_$timestamp"))
    epochs = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_EPOCHS", "800"))
    log_every = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_LOG_EVERY", "200"))
    minit = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_MINIT", "4"))
    eval_repeats = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_EVAL_REPEATS", "16"))
    workers = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    free_relaxation = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_FREE_RELAXATION", "150"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_NUDGED_RELAXATION", "150"))
    limit = parse(Int, get(ENV, "ISING_SIMPLE_CHECKER_LIMIT", "0"))
    names = parse_names(get(ENV, "ISING_SIMPLE_CHECKER_CONFIGS", ""))

    searches = simple_baseline_searches(;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation,
        nudged_relaxation,
    )
    if !isempty(names)
        searches = [s for s in searches if s.config.name in names]
    elseif limit > 0
        searches = searches[1:min(limit, length(searches))]
    end
    isempty(searches) && error("no simple checkerboard configs selected")

    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summaries = NamedTuple[]

    println("Running simple checkerboard baseline grid with $(length(searches)) config(s)")
    for (idx, search) in enumerate(searches)
        println("[$idx/$(length(searches))] $(search.config.name)")
        result = run_stabilized_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        saved = isnothing(result.graph_path) ? "no" : "yes"
        push!(summaries, (;
            config = search.config.name,
            best_acc = result.best_acc,
            best_mse = result.best_mse,
            saved,
            notes = search.notes,
        ))
        best = first(sort(summaries; by = row -> (-row.best_acc, row.best_mse)))
        println("current best: $(best.config), acc=$(best.best_acc), mse=$(best.best_mse)")
    end

    sorted = sort(summaries; by = row -> (-row.best_acc, row.best_mse))
    csv_path = write_csv(joinpath(outdir, "simple_checkerboard_grid_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "simple_checkerboard_grid_progress.png"), all_rows)
    summary_path = write_simple_grid_summary(joinpath(outdir, "simple_checkerboard_grid_summary.csv"), sorted)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Simple Checkerboard Baseline Grid")
        println(io)
        println(io, "Random-init local checkerboard XOR grid. No toolbox code was changed.")
        println(io)
        println(io, "## Top Results")
        println(io)
        println(io, "| Rank | Config | Best MSE | Best Acc | Saved |")
        println(io, "|---:|---|---:|---:|---|")
        for (rank, row) in enumerate(first(sorted, min(20, length(sorted))))
            println(io, "| $rank | `$(row.config)` | $(round(row.best_mse, digits=6)) | $(round(row.best_acc, digits=3)) | $(row.saved) |")
        end
        println(io)
        println(io, "Metrics CSV: `$(basename(csv_path))`")
        println(io, "Summary CSV: `$(basename(summary_path))`")
        println(io, "Progress PNG: `$(basename(png_path))`")
    end

    println("Saved metrics: $csv_path")
    println("Saved summary: $summary_path")
    println("Saved plot: $png_path")
    println("Saved docs: $md_path")
    return (; outdir, summaries = sorted, csv_path, summary_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
