using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "local_checkerboard_xor.jl"))

"""
    grid_config(; kwargs...)

Build one focused 2x2 local-checkerboard XOR configuration for short grid
searches. This intentionally keeps the physical input code, output-pattern
clamping, symmetric weights, random initialization, and no double well fixed.
"""
function grid_config(;
    name,
    dynamics_mode,
    state_mode,
    temp,
    stepsize = FT(0.08),
    β = FT(0.2),
    lr = FT(0.004),
    inter_weight_scale = FT(0.1),
    inter_radius = sqrt(FT(2)) + FT(1e-6),
    internal_scale = FT(0.01),
    internal_nn = 1,
    free_relaxation,
    nudged_relaxation,
    epochs,
    log_every,
    minit,
    eval_repeats,
    workers,
    seed_offset = 0,
)
    return LocalCheckerboardConfig(
        name = name,
        side = 2,
        hidden_side = 2,
        code_side = 2,
        code_stride = 1,
        code_offset = (1, 1),
        epochs = epochs,
        log_every = log_every,
        minit = minit,
        eval_repeats = eval_repeats,
        workers = workers,
        free_relaxation = free_relaxation,
        nudged_relaxation = nudged_relaxation,
        β = β,
        lr = lr,
        weight_decay = FT(1e-4),
        grad_clip = FT(20),
        temp = temp,
        temp_is_factor = false,
        stepsize = stepsize,
        block_size = 4,
        inter_radius = FT(inter_radius),
        internal_nn = internal_nn,
        inter_weight_scale = inter_weight_scale,
        input_internal_scale = internal_scale,
        hidden_internal_scale = internal_scale,
        output_internal_scale = internal_scale,
        bias_scale = FT(0.02),
        weight_seed = 2 + seed_offset,
        internal_seed = 3 + seed_offset,
        bias_seed = 11 + seed_offset,
        base_seed = 91_000 + 10_000 * seed_offset,
        init_mode = :random,
        state_mode = state_mode,
        dynamics_mode = dynamics_mode,
        output_clamp_mode = :pattern,
        doublewell_barrier = zero(FT),
        free_temp_start_factor = one(FT),
        free_temp_stop_factor = one(FT),
        nudged_temp_start_factor = one(FT),
        nudged_temp_stop_factor = one(FT),
        temp_schedule_power = one(FT),
    )
end

function candidate_configs(; epochs, log_every, minit, eval_repeats, workers, free_relaxation, nudged_relaxation, include_metropolis = false)
    configs = LocalCheckerboardConfig[]

    # Block Langevin probes. These scan temperature, step size, beta, layer-to-layer
    # scale, same-layer connectivity, and inter-layer radius/fanout while keeping
    # the physical local checkerboard setup.
    langevin_specs = [
        (0.0100, 0.08, 0.2, 0.25),
        (0.0100, 0.12, 0.5, 0.50),
        (0.0200, 0.08, 0.2, 0.50),
        (0.0400, 0.12, 0.5, 0.50),
    ]
    internal_nns = (1, 2, 3)
    radius_specs = (
        (1.01, "r1"),
        (sqrt(2) + 1e-6, "r141"),
        (sqrt(3) + 1e-6, "r173"),
        (10.0, "rall"),
    )
    idx = 0
    for (temp, stepsize, beta, inter_scale) in langevin_specs
        for nn in internal_nns
            for (radius, radius_label) in radius_specs
                idx += 1
                push!(configs, grid_config(
                    name = "grid_block_$(idx)_T$(temp)_s$(stepsize)_b$(beta)_J$(inter_scale)_NN$(nn)_$(radius_label)",
                    dynamics_mode = :langevin,
                    state_mode = :continuous,
                    temp = FT(temp),
                    stepsize = FT(stepsize),
                    β = FT(beta),
                    inter_weight_scale = FT(inter_scale),
                    inter_radius = FT(radius),
                    internal_nn = nn,
                    epochs = epochs,
                    log_every = log_every,
                    minit = minit,
                    eval_repeats = eval_repeats,
                    workers = workers,
                    free_relaxation = free_relaxation,
                    nudged_relaxation = nudged_relaxation,
                    seed_offset = idx,
                ))
            end
        end
    end

    # Global Langevin minimization probes. These are included because global
    # gradients may find lower-energy states faster in the tiny 2x2 test.
    global_specs = [
        (0.0025, 0.05, 0.2, 0.25, 1),
        (0.0100, 0.08, 0.5, 0.25, 1),
        (0.0400, 0.12, 0.2, 0.40, 1),
    ]
    for (idx, (temp, stepsize, beta, inter_scale, nn)) in enumerate(global_specs)
        push!(configs, grid_config(
            name = "grid_global_$(idx)_T$(temp)_s$(stepsize)_b$(beta)_J$(inter_scale)_NN$(nn)_r141",
            dynamics_mode = :global_langevin,
            state_mode = :continuous,
            temp = FT(temp),
            stepsize = FT(stepsize),
            β = FT(beta),
            inter_weight_scale = FT(inter_scale),
            inter_radius = sqrt(FT(2)) + FT(1e-6),
            internal_nn = nn,
            epochs = epochs,
            log_every = log_every,
            minit = minit,
            eval_repeats = eval_repeats,
            workers = workers,
            free_relaxation = free_relaxation,
            nudged_relaxation = nudged_relaxation,
            seed_offset = 100 + idx,
        ))
    end

    if include_metropolis
        # Discrete Metropolis controls. If these separate while Langevin does not,
        # the bottleneck is the continuous relaxation/noise scale rather than the
        # physical checkerboard encoding itself.
        metropolis_specs = [
            (0.05, 0.2, 0.10, 1),
            (0.20, 0.2, 0.25, 1),
            (0.50, 0.5, 0.25, 1),
            (1.00, 0.5, 0.40, 1),
        ]
        for (idx, (temp, beta, inter_scale, nn)) in enumerate(metropolis_specs)
            push!(configs, grid_config(
                name = "grid_metro_$(idx)_T$(temp)_b$(beta)_J$(inter_scale)_NN$(nn)_r141",
                dynamics_mode = :metropolis,
                state_mode = :discrete,
                temp = FT(temp),
                β = FT(beta),
                inter_weight_scale = FT(inter_scale),
                inter_radius = sqrt(FT(2)) + FT(1e-6),
                internal_scale = FT(0.01),
                internal_nn = nn,
                epochs = epochs,
                log_every = log_every,
                minit = minit,
                eval_repeats = eval_repeats,
                workers = workers,
                free_relaxation = free_relaxation,
                nudged_relaxation = nudged_relaxation,
                seed_offset = 200 + idx,
            ))
        end
    end

    return configs
end

function final_row(rows, name)
    sub = [row for row in rows if row["config"] == name]
    isempty(sub) && error("no rows for config $name")
    return sub[argmax(row["epoch"] for row in sub)]
end

function write_grid_summary(path, summaries)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "rank,config,best_acc,best_mse,final_acc,final_mse,graph")
        for (rank, row) in enumerate(summaries)
            println(io, join((
                rank,
                row.config,
                row.best_acc,
                row.best_mse,
                row.final_acc,
                row.final_mse,
                row.graph_path,
            ), ","))
        end
    end
    return path
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_LOCAL_GRID_DIR", joinpath(DEFAULT_RUN_ROOT, "local_checkerboard_grid_$timestamp"))
    epochs = parse(Int, get(ENV, "ISING_LOCAL_GRID_EPOCHS", "150"))
    log_every = parse(Int, get(ENV, "ISING_LOCAL_GRID_LOG_EVERY", "50"))
    minit = parse(Int, get(ENV, "ISING_LOCAL_GRID_MINIT", "2"))
    eval_repeats = parse(Int, get(ENV, "ISING_LOCAL_GRID_EVAL_REPEATS", "8"))
    workers = parse(Int, get(ENV, "ISING_LOCAL_GRID_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    free_relaxation = parse(Int, get(ENV, "ISING_LOCAL_GRID_FREE_RELAXATION", "250"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_LOCAL_GRID_NUDGED_RELAXATION", "250"))
    limit = parse(Int, get(ENV, "ISING_LOCAL_GRID_LIMIT", "0"))
    include_metropolis = parse(Bool, get(ENV, "ISING_LOCAL_GRID_INCLUDE_METROPOLIS", "false"))

    configs = candidate_configs(; epochs, log_every, minit, eval_repeats, workers, free_relaxation, nudged_relaxation, include_metropolis)
    if limit > 0
        configs = configs[1:min(limit, length(configs))]
    end

    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    summaries = NamedTuple[]

    println("Running local checkerboard grid with $(length(configs)) config(s)")
    for (idx, cfg) in enumerate(configs)
        println("[$idx/$(length(configs))] $(cfg.name)")
        result = run_config(cfg, joinpath(outdir, cfg.name))
        append!(all_rows, result.rows)
        final = final_row(result.rows, cfg.name)
        push!(summaries, (;
            config = cfg.name,
            best_acc = result.best_acc,
            best_mse = result.best_mse,
            final_acc = final["accuracy"],
            final_mse = final["mse"],
            graph_path = result.graph_path,
        ))
        sorted_now = sort(summaries; by = row -> (-row.best_acc, row.best_mse))
        best = first(sorted_now)
        println("current best: $(best.config), acc=$(best.best_acc), mse=$(best.best_mse)")
    end

    csv_path = write_csv(joinpath(outdir, "local_checkerboard_grid_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "local_checkerboard_grid_progress.png"), all_rows)
    summary_path = write_grid_summary(joinpath(outdir, "local_checkerboard_grid_summary.csv"), sort(summaries; by = row -> (-row.best_acc, row.best_mse)))

    println("Saved metrics: $csv_path")
    println("Saved plot: $png_path")
    println("Saved summary: $summary_path")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
