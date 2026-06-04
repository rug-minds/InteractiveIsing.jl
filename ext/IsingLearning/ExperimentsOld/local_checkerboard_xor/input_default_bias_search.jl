using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_stabilized_search.jl"))

"""
    input_default_bias_searches(; kwargs...)

Return a focused set of local-checkerboard XOR configs where inactive input
bits are not frozen, but the input-code sites have a fixed negative magnetic
field. This gives `0` bits a physical `-1` default while preserving the intended
input protocol:

- `(0, 0)`: no input sites frozen, input layer relaxes under the default field;
- `(1, 0)`: pattern A frozen to `+1`, pattern B relaxes under the default field;
- `(0, 1)`: pattern B frozen to `+1`, pattern A relaxes under the default field;
- `(1, 1)`: both patterns frozen to `+1`.

The bias is fixed after every update in this experiment-local file. This avoids
letting the optimizer erase the input convention while still using the normal
StatefulAlgorithms/contrastive-gradient training path.
"""
function input_default_bias_searches(;
    epochs,
    log_every,
    minit,
    eval_repeats,
    workers,
    free_relaxation,
    nudged_relaxation,
)
    dynamics = Symbol(get(ENV, "ISING_INPUT_DEFAULT_DYNAMICS", "metropolis"))
    state_mode = Symbol(get(ENV, "ISING_INPUT_DEFAULT_STATE", dynamics === :metropolis ? "discrete" : "continuous"))
    init_mode = Symbol(get(ENV, "ISING_INPUT_DEFAULT_INIT", "random"))
    common = (;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation,
        nudged_relaxation,
        weight_decay = parse(FT, get(ENV, "ISING_INPUT_DEFAULT_WEIGHT_DECAY", "0")),
        grad_clip = parse(FT, get(ENV, "ISING_INPUT_DEFAULT_GRAD_CLIP", "20")),
        init_mode,
        dynamics_mode = dynamics,
        state_mode,
        output_clamp_mode = Symbol(get(ENV, "ISING_INPUT_DEFAULT_OUTPUT_CLAMP", "readout")),
        doublewell_barrier = FT(0),
        temp_is_factor = parse(Bool, get(ENV, "ISING_INPUT_DEFAULT_TEMP_IS_FACTOR", "false")),
        stepsize = parse(FT, get(ENV, "ISING_INPUT_DEFAULT_STEPSIZE", "0.05")),
    )

    configs = StabilizedSearchConfig[]
    idx = 0
    side_specs = (
        (; side = 2, hidden_side = 2, code_side = 2, code_stride = 1, radius = FT(sqrt(2) + 1e-6), block_size = 4, label = "2x2"),
        (; side = 4, hidden_side = 4, code_side = 4, code_stride = 1, radius = FT(sqrt(2) + 1e-6), block_size = 8, label = "4x4"),
        (; side = 4, hidden_side = 6, code_side = 4, code_stride = 1, radius = FT(2.05), block_size = 8, label = "4x4_h6"),
    )
    input_biases = FT.(parse.(Float64, split(get(ENV, "ISING_INPUT_DEFAULT_BIASES", "0.15,0.3,0.6"), ",")))
    temps = FT.(parse.(Float64, split(get(ENV, "ISING_INPUT_DEFAULT_TEMPS", "0.02,0.05,0.10"), ",")))
    inter_scales = FT.(parse.(Float64, split(get(ENV, "ISING_INPUT_DEFAULT_INTER_SCALES", "0.10,0.20,0.35"), ",")))
    internal_scales = FT.(parse.(Float64, split(get(ENV, "ISING_INPUT_DEFAULT_INTERNAL_SCALES", "0.05,0.10"), ",")))
    bias_scales = FT.(parse.(Float64, split(get(ENV, "ISING_INPUT_DEFAULT_OTHER_BIAS_SCALES", "0.0,0.02"), ",")))
    beta_lr_specs = ((FT(0.2), FT(0.003)), (FT(0.3), FT(0.003)), (FT(0.3), FT(0.006)))

    for spec in side_specs
        for input_bias in input_biases
            for temp in temps
                for inter_scale in inter_scales
                    for internal_scale in internal_scales
                        for other_bias_scale in bias_scales
                            for (beta, lr) in beta_lr_specs
                                idx += 1
                                seed = 700_000 + 37 * idx
                                name = "inputdefault_$(spec.label)_ib$(input_bias)_T$(temp)_J$(inter_scale)_is$(internal_scale)_obs$(other_bias_scale)_b$(beta)_lr$(lr)"
                                cfg = LocalCheckerboardConfig(;
                                    common...,
                                    name,
                                    side = spec.side,
                                    hidden_side = spec.hidden_side,
                                    code_side = spec.code_side,
                                    code_stride = spec.code_stride,
                                    code_offset = (1, 1),
                                    block_size = spec.block_size,
                                    inter_radius = spec.radius,
                                    internal_nn = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_INTERNAL_NN", "1")),
                                    temp,
                                    β = beta,
                                    lr,
                                    inter_weight_scale = inter_scale,
                                    input_internal_scale = internal_scale,
                                    hidden_internal_scale = internal_scale,
                                    output_internal_scale = internal_scale,
                                    bias_scale = other_bias_scale,
                                    weight_seed = seed,
                                    internal_seed = seed + 1,
                                    bias_seed = seed + 2,
                                    base_seed = 3_000_000 + 10_000 * idx,
                                )
                                push!(configs, StabilizedSearchConfig(
                                    config = cfg,
                                    save_threshold = FT(0.2),
                                    input_default_bias = input_bias,
                                    fix_input_default_bias = true,
                                    notes = "fixed negative input-code bias gives inactive input bits a physical -1 default; active bits are still the only frozen sites",
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

function selected_input_default_searches(searches)
    wanted = filter(!isempty, strip.(split(get(ENV, "ISING_INPUT_DEFAULT_CONFIGS", ""), ",")))
    limit = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_LIMIT", "0"))
    if !isempty(wanted)
        selected = [s for s in searches if s.config.name in wanted]
        isempty(selected) && error("no configs matched ISING_INPUT_DEFAULT_CONFIGS")
        return selected
    elseif limit > 0
        return searches[1:min(limit, length(searches))]
    else
        return searches
    end
end

function write_input_default_summary(path, summaries)
    open(path, "w") do io
        println(io, "rank,config,best_acc,best_mse,saved,notes")
        for (rank, row) in enumerate(summaries)
            println(io, join((rank, row.config, row.best_acc, row.best_mse, row.saved, row.notes), ","))
        end
    end
    return path
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_INPUT_DEFAULT_DIR", joinpath(DEFAULT_RUN_ROOT, "input_default_bias_$timestamp"))
    epochs = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_EPOCHS", "1200"))
    log_every = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_LOG_EVERY", "200"))
    minit = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_EVAL_REPEATS", "32"))
    workers = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    free_relaxation = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_FREE_RELAXATION", "150"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_INPUT_DEFAULT_NUDGED_RELAXATION", "150"))

    searches = selected_input_default_searches(input_default_bias_searches(;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation,
        nudged_relaxation,
    ))
    isempty(searches) && error("no input-default configs selected")

    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summaries = NamedTuple[]
    println("Running input-default checkerboard search with $(length(searches)) config(s)")
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
    csv_path = write_csv(joinpath(outdir, "input_default_bias_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "input_default_bias_progress.png"), all_rows)
    summary_path = write_input_default_summary(joinpath(outdir, "input_default_bias_summary.csv"), sorted)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Input Default Bias Local Checkerboard Search")
        println(io)
        println(io, "Inactive input bits are not frozen. The input-code sites have a fixed negative magnetic field, so the dynamics can relax unforced input bits toward `-1`.")
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
