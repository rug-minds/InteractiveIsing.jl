using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    structured_and_seed_searches(; kwargs...)

Experiment-local test of a minimal local architecture that contains an explicit
hidden `A`, `B`, and `AND` feature path. The graph still uses the checkerboard
input protocol and same-layer local connections. The structured path is only an
initialization; unless later projected, the normal EqProp update can change it.
"""
function structured_and_seed_searches(;
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
        weight_decay = parse(FT, get(ENV, "ISING_STRUCTURED_WEIGHT_DECAY", "0")),
        grad_clip = parse(FT, get(ENV, "ISING_STRUCTURED_GRAD_CLIP", "20")),
        init_mode = Symbol(get(ENV, "ISING_STRUCTURED_INIT", "random")),
        dynamics_mode = Symbol(get(ENV, "ISING_STRUCTURED_DYNAMICS", "metropolis")),
        state_mode = Symbol(get(ENV, "ISING_STRUCTURED_STATE", "discrete")),
        output_clamp_mode = Symbol(get(ENV, "ISING_STRUCTURED_OUTPUT_CLAMP", "two_readout")),
        doublewell_barrier = FT(0),
        temp_is_factor = false,
        stepsize = parse(FT, get(ENV, "ISING_STRUCTURED_STEPSIZE", "0.05")),
    )

    configs = StabilizedSearchConfig[]
    idx = 0
    side_specs = (
        (; side = 2, hidden_side = 2, code_side = 2, code_stride = 1, radius = FT(sqrt(2) + 1e-6), block_size = 4, label = "2x2"),
        (; side = 4, hidden_side = 4, code_side = 4, code_stride = 1, radius = FT(sqrt(2) + 1e-6), block_size = 8, label = "4x4"),
    )
    temps = FT.(parse.(Float64, split(get(ENV, "ISING_STRUCTURED_TEMPS", "0.02,0.05,0.10"), ",")))
    input_biases = FT.(parse.(Float64, split(get(ENV, "ISING_STRUCTURED_INPUT_BIASES", "0.3,0.6"), ",")))
    feature_scales = FT.(parse.(Float64, split(get(ENV, "ISING_STRUCTURED_FEATURE_SCALES", "0.8,1.2"), ",")))
    output_scales = FT.(parse.(Float64, split(get(ENV, "ISING_STRUCTURED_OUTPUT_SCALES", "0.4,0.8"), ",")))
    output_biases = FT.(parse.(Float64, split(get(ENV, "ISING_STRUCTURED_OUTPUT_BIASES", "0.2,0.5"), ",")))
    beta_lr_specs = ((FT(0.1), FT(0.002)), (FT(0.2), FT(0.002)), (FT(0.2), FT(0.004)))

    for spec in side_specs
        for temp in temps
            for input_bias in input_biases
                for feature_scale in feature_scales
                    for output_scale in output_scales
                        for output_bias in output_biases
                            for (beta, lr) in beta_lr_specs
                                idx += 1
                                seed = 9_000_000 + 41 * idx
                                cfg = LocalCheckerboardConfig(;
                                    common...,
                                    name = "structured_$(spec.label)_T$(temp)_ib$(input_bias)_f$(feature_scale)_o$(output_scale)_ob$(output_bias)_b$(beta)_lr$(lr)",
                                    side = spec.side,
                                    hidden_side = spec.hidden_side,
                                    code_side = spec.code_side,
                                    code_stride = spec.code_stride,
                                    code_offset = (1, 1),
                                    block_size = spec.block_size,
                                    inter_radius = spec.radius,
                                    internal_nn = parse(Int, get(ENV, "ISING_STRUCTURED_INTERNAL_NN", "1")),
                                    temp,
                                    β = beta,
                                    lr,
                                    inter_weight_scale = FT(0.0),
                                    input_internal_scale = parse(FT, get(ENV, "ISING_STRUCTURED_INTERNAL_SCALE", "0.05")),
                                    hidden_internal_scale = parse(FT, get(ENV, "ISING_STRUCTURED_INTERNAL_SCALE", "0.05")),
                                    output_internal_scale = parse(FT, get(ENV, "ISING_STRUCTURED_INTERNAL_SCALE", "0.05")),
                                    bias_scale = FT(0.0),
                                    weight_seed = seed,
                                    internal_seed = seed + 1,
                                    bias_seed = seed + 2,
                                    base_seed = 6_000_000 + 10_000 * idx,
                                )
                                push!(configs, StabilizedSearchConfig(
                                    config = cfg,
                                    save_threshold = FT(0.2),
                                    input_default_bias = input_bias,
                                    fix_input_default_bias = true,
                                    structured_and_seed = true,
                                    structured_feature_scale = feature_scale,
                                    structured_output_scale = output_scale,
                                    structured_output_bias = output_bias,
                                    structured_zero_interlayer = true,
                                    notes = "explicit local A/B/AND hidden seed with fixed input default bias",
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

function selected_structured_searches(searches)
    wanted = filter(!isempty, strip.(split(get(ENV, "ISING_STRUCTURED_CONFIGS", ""), ",")))
    limit = parse(Int, get(ENV, "ISING_STRUCTURED_LIMIT", "0"))
    if !isempty(wanted)
        selected = [s for s in searches if s.config.name in wanted]
        isempty(selected) && error("no configs matched ISING_STRUCTURED_CONFIGS")
        return selected
    elseif limit > 0
        return searches[1:min(limit, length(searches))]
    else
        return searches
    end
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_STRUCTURED_DIR", joinpath(DEFAULT_RUN_ROOT, "structured_and_seed_$timestamp"))
    epochs = parse(Int, get(ENV, "ISING_STRUCTURED_EPOCHS", "1000"))
    log_every = parse(Int, get(ENV, "ISING_STRUCTURED_LOG_EVERY", "200"))
    minit = parse(Int, get(ENV, "ISING_STRUCTURED_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_STRUCTURED_EVAL_REPEATS", "32"))
    workers = parse(Int, get(ENV, "ISING_STRUCTURED_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    free_relaxation = parse(Int, get(ENV, "ISING_STRUCTURED_FREE_RELAXATION", "150"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_STRUCTURED_NUDGED_RELAXATION", "150"))

    searches = selected_structured_searches(structured_and_seed_searches(;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation,
        nudged_relaxation,
    ))
    isempty(searches) && error("no structured configs selected")

    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summaries = NamedTuple[]
    println("Running structured AND-seed checkerboard search with $(length(searches)) config(s)")
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
    csv_path = write_csv(joinpath(outdir, "structured_and_seed_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "structured_and_seed_progress.png"), all_rows)
    summary_path = joinpath(outdir, "structured_and_seed_summary.csv")
    open(summary_path, "w") do io
        println(io, "rank,config,best_acc,best_mse,saved,notes")
        for (rank, row) in enumerate(sorted)
            println(io, join((rank, row.config, row.best_acc, row.best_mse, row.saved, row.notes), ","))
        end
    end
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Structured AND-Seed Checkerboard Search")
        println(io)
        println(io, "This run tests whether the local checkerboard task needs an explicit local `A`, `B`, and `AND` feature route. The graph still uses symmetric weights, same-layer local connections, and the normal contrastive-gradient worker path.")
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
