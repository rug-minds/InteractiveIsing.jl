using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "langevin_checkerboard_xor.jl"))

"""
    parse_sweep_list(name, default, T)

Read a comma-separated environment override for a field-scaled checkerboard
sweep. Empty entries are ignored so shell overrides can be edited quickly.
"""
function parse_sweep_list(name::AbstractString, default, ::Type{T}) where {T}
    raw = get(ENV, name, join(string.(default), ","))
    vals = filter(!isempty, strip.(split(raw, ",")))
    return T.(parse.(T, vals))
end

function parse_string_list(name::AbstractString, default)
    raw = get(ENV, name, join(string.(default), ","))
    return filter(!isempty, strip.(split(raw, ",")))
end

layer_ranges(config::LocalCheckerboardConfig) = begin
    n1 = config.side^2
    n2 = config.hidden_side^2
    n3 = config.side^2
    (; input = 1:n1, hidden = (n1 + 1):(n1 + n2), output = (n1 + n2 + 1):(n1 + n2 + n3))
end

function _copy_checker_config(config::LocalCheckerboardConfig; kwargs...)
    vals = Dict{Symbol,Any}(name => getfield(config, name) for name in fieldnames(LocalCheckerboardConfig))
    for (k, v) in kwargs
        vals[k] = v
    end
    return LocalCheckerboardConfig(; vals...)
end

function _mean_row_degree(A, rows_range, cols_range)
    rows, cols, _ = SparseArrays.findnz(A)
    rowset = Set(rows_range)
    colset = Set(cols_range)
    counts = Dict{Int,Int}()
    for k in eachindex(rows)
        r = rows[k]
        c = cols[k]
        (r in rowset && c in colset) || continue
        counts[r] = get(counts, r, 0) + 1
    end
    isempty(counts) && return 0.0
    return mean(values(counts))
end

"""
    coupling_fanouts(config)

Measure the mean nonzero row degree of each connection class for this exact
geometry. This is used to choose edge RMS values from desired local-field RMS
targets: for independent weights, `field_rms ≈ edge_scale * sqrt(fanout)`.
"""
function coupling_fanouts(config::LocalCheckerboardConfig)
    probe = _copy_checker_config(
        config;
        inter_weight_scale = one(FT),
        input_internal_scale = one(FT),
        hidden_internal_scale = one(FT),
        output_internal_scale = one(FT),
        bias_scale = zero(FT),
        temp = one(FT),
        temp_is_factor = false,
    )
    g = checkerboard_graph(probe)
    A = II.adj(g).sp
    r = layer_ranges(probe)
    return (;
        input_internal = _mean_row_degree(A, r.input, r.input),
        hidden_internal = _mean_row_degree(A, r.hidden, r.hidden),
        output_internal = _mean_row_degree(A, r.output, r.output),
        input_hidden = _mean_row_degree(A, r.input, r.hidden),
        hidden_input = _mean_row_degree(A, r.hidden, r.input),
        hidden_output = _mean_row_degree(A, r.hidden, r.output),
        output_hidden = _mean_row_degree(A, r.output, r.hidden),
    )
end

edge_scale_for_field(field_rms, fanout) =
    fanout <= 0 ? zero(FT) : FT(field_rms) / sqrt(FT(fanout))

"""
    field_scaled_config(proto; ...)

Return a config whose random edge scales are normalized by measured fanout.
`inter_field` is the intended RMS contribution of each adjacent layer block to
one spin's local field. Internal layer targets are `internal_ratio * inter_field`.
"""
function field_scaled_config(
    proto::LocalCheckerboardConfig;
    name,
    inter_field::Real,
    internal_ratio::Real,
    temp_factor::Real,
    stepsize::Real,
    dynamics_mode::Symbol,
)
    f = coupling_fanouts(proto)
    input_internal_scale = edge_scale_for_field(inter_field * internal_ratio, f.input_internal)
    hidden_internal_scale = edge_scale_for_field(inter_field * internal_ratio, f.hidden_internal)
    output_internal_scale = edge_scale_for_field(inter_field * internal_ratio, f.output_internal)
    inter_fanout = mean((f.input_hidden, f.hidden_input, f.hidden_output, f.output_hidden))
    inter_weight_scale = edge_scale_for_field(inter_field, inter_fanout)
    return _copy_checker_config(
        proto;
        name,
        temp = FT(temp_factor),
        temp_is_factor = true,
        stepsize = FT(stepsize),
        dynamics_mode,
        inter_weight_scale,
        input_internal_scale,
        hidden_internal_scale,
        output_internal_scale,
        bias_scale = FT(0.05) * FT(inter_field),
    )
end

function field_scaled_prototypes()
    target = Symbol(get(ENV, "ISING_FIELD_LGV_TARGET", "inlaid8"))
    epochs = parse(Int, get(ENV, "ISING_FIELD_LGV_EPOCHS", "300"))
    log_every = parse(Int, get(ENV, "ISING_FIELD_LGV_LOG_EVERY", "100"))
    minit = parse(Int, get(ENV, "ISING_FIELD_LGV_MINIT", "4"))
    eval_repeats = parse(Int, get(ENV, "ISING_FIELD_LGV_EVAL_REPEATS", "16"))
    workers = parse(Int, get(ENV, "ISING_FIELD_LGV_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    relaxation = parse(Int, get(ENV, "ISING_FIELD_LGV_RELAXATION", "1000"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_FIELD_LGV_NUDGED_RELAXATION", string(relaxation)))
    lr = parse(FT, get(ENV, "ISING_FIELD_LGV_LR", "0.004"))
    β = parse(FT, get(ENV, "ISING_FIELD_LGV_BETA", "1.0"))
    grad_clip = parse(FT, get(ENV, "ISING_FIELD_LGV_GRAD_CLIP", "100.0"))
    weight_decay = parse(FT, get(ENV, "ISING_FIELD_LGV_WEIGHT_DECAY", "0.0"))
    internal_nn = parse(Int, get(ENV, "ISING_FIELD_LGV_INTERNAL_NN", "5"))
    base_seed = parse(Int, get(ENV, "ISING_FIELD_LGV_BASE_SEED", "170000"))

    common = (;
        epochs,
        log_every,
        minit,
        eval_repeats,
        workers,
        free_relaxation = relaxation,
        nudged_relaxation,
        β,
        lr,
        weight_decay,
        grad_clip,
        internal_nn,
        weight_seed = parse(Int, get(ENV, "ISING_FIELD_LGV_WEIGHT_SEED", "2")),
        internal_seed = parse(Int, get(ENV, "ISING_FIELD_LGV_INTERNAL_SEED", "3")),
        bias_seed = parse(Int, get(ENV, "ISING_FIELD_LGV_BIAS_SEED", "11")),
        base_seed,
        init_mode = :random,
        state_mode = :continuous,
        output_clamp_mode = Symbol(get(ENV, "ISING_FIELD_LGV_OUTPUT_CLAMP", "readout")),
    )

    if target === :global4
        return [LocalCheckerboardConfig(; name = "field_global4", side = 4, hidden_side = 4, code_side = 4, code_stride = 1, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS", "2.05")), block_size = 16, common...)]
    elseif target === :global8
        return [LocalCheckerboardConfig(; name = "field_global8", side = 8, hidden_side = 8, code_side = 8, code_stride = 1, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS", "3.05")), block_size = 64, common...)]
    elseif target === :inlaid8
        return [LocalCheckerboardConfig(; name = "field_inlaid8", side = 8, hidden_side = 8, code_side = 4, code_stride = 2, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS", "3.05")), block_size = 64, common...)]
    elseif target === :all
        return [
            LocalCheckerboardConfig(; name = "field_global4", side = 4, hidden_side = 4, code_side = 4, code_stride = 1, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS_4", "2.05")), block_size = 16, common...),
            LocalCheckerboardConfig(; name = "field_global8", side = 8, hidden_side = 8, code_side = 8, code_stride = 1, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS_8", "3.05")), block_size = 64, common...),
            LocalCheckerboardConfig(; name = "field_inlaid8", side = 8, hidden_side = 8, code_side = 4, code_stride = 2, code_offset = (1, 1), inter_radius = parse(FT, get(ENV, "ISING_FIELD_LGV_INTER_RADIUS_INLAID", "3.05")), block_size = 64, common...),
        ]
    else
        throw(ArgumentError("ISING_FIELD_LGV_TARGET must be global4, global8, inlaid8, or all"))
    end
end

function field_scaled_configs()
    prototypes = field_scaled_prototypes()
    inter_fields = parse_sweep_list("ISING_FIELD_LGV_INTER_FIELDS", [0.30, 0.60], FT)
    internal_ratios = parse_sweep_list("ISING_FIELD_LGV_INTERNAL_RATIOS", [0.25, 0.50], FT)
    temp_factors = parse_sweep_list("ISING_FIELD_LGV_TEMP_FACTORS", [0.005, 0.02], FT)
    stepsizes = parse_sweep_list("ISING_FIELD_LGV_STEPSIZES", [0.05, 0.10], FT)
    modes = Symbol.(parse_string_list("ISING_FIELD_LGV_DYNAMICS", ["global_langevin"]))

    configs = LocalCheckerboardConfig[]
    for proto in prototypes, mode in modes, inter_field in inter_fields, ratio in internal_ratios, temp_factor in temp_factors, stepsize in stepsizes
        name = join((
            proto.name,
            string(mode),
            "F$(replace(string(inter_field), "." => "p"))",
            "R$(replace(string(ratio), "." => "p"))",
            "Tf$(replace(string(temp_factor), "." => "p"))",
            "eta$(replace(string(stepsize), "." => "p"))",
        ), "_")
        push!(
            configs,
            field_scaled_config(
                proto;
                name,
                inter_field,
                internal_ratio = ratio,
                temp_factor,
                stepsize,
                dynamics_mode = mode,
            ),
        )
    end
    return configs
end

function field_scaled_summary_row(config::LocalCheckerboardConfig, result)
    g = checkerboard_graph(config)
    fanout = coupling_fanouts(config)
    scale = max_local_interaction_energy(g)
    return Dict{String,Any}(
        "config" => config.name,
        "dynamics" => string(config.dynamics_mode),
        "inter_radius" => config.inter_radius,
        "internal_nn" => config.internal_nn,
        "temp_factor" => config.temp,
        "effective_temp" => effective_temp(g, config),
        "max_local_interaction_energy" => scale,
        "stepsize" => config.stepsize,
        "output_clamp_mode" => string(config.output_clamp_mode),
        "inter_weight_scale" => config.inter_weight_scale,
        "input_internal_scale" => config.input_internal_scale,
        "hidden_internal_scale" => config.hidden_internal_scale,
        "output_internal_scale" => config.output_internal_scale,
        "fanout_input_hidden" => fanout.input_hidden,
        "fanout_hidden_output" => fanout.hidden_output,
        "symmetry_error" => adjacency_symmetry_error(g),
        "best_mse" => result.best_mse,
        "best_accuracy" => result.best_acc,
        "graph" => result.graph_path,
    )
end

function field_scaled_main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_FIELD_LGV_DIR", joinpath(LANGEVIN_CHECKER_RUN_ROOT, "field_scaled_langevin_checkerboard_$timestamp"))
    mkpath(outdir)

    configs = field_scaled_configs()
    isempty(configs) && error("no field-scaled configs")
    all_rows = Dict{String,Any}[]
    summary_rows = Dict{String,Any}[]
    for (idx, config) in enumerate(configs)
        println("Field-scaled config $idx/$(length(configs)): $(config.name)")
        println("  dynamics=$(config.dynamics_mode), T=$(config.temp) * Emax, effective temp will be set after graph build")
        println("  inter scale=$(config.inter_weight_scale), internal scales=$(config.input_internal_scale),$(config.hidden_internal_scale),$(config.output_internal_scale)")
        result = run_config(config, joinpath(outdir, config.name))
        append!(all_rows, result.rows)
        push!(summary_rows, field_scaled_summary_row(config, result))
    end

    metrics_csv = write_csv(joinpath(outdir, "field_scaled_langevin_checkerboard_metrics.csv"), all_rows)
    summary_csv = write_csv(joinpath(outdir, "field_scaled_langevin_checkerboard_summary.csv"), summary_rows)
    progress_png = plot_langevin_summary(joinpath(outdir, "field_scaled_langevin_checkerboard_progress.png"), all_rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Field-Scaled Langevin Checkerboard XOR")
        println(io)
        println(io, "This run normalizes random edge scales by measured fanout, sets `temp = factor * max_local_interaction_energy(graph)`, and checks adjacency symmetry for every saved graph.")
        println(io)
        println(io, "## Results")
        for row in sort(summary_rows, by = row -> row["best_mse"])
            println(io, "- `$(row["config"])`: best mse=$(round(row["best_mse"], digits=6)), acc=$(round(row["best_accuracy"], digits=3)), Teff=$(round(row["effective_temp"], sigdigits=4)), symmetry=$(row["symmetry_error"])")
        end
        println(io)
        println(io, "## Files")
        println(io, "- Metrics: `$(basename(metrics_csv))`")
        println(io, "- Summary: `$(basename(summary_csv))`")
        println(io, "- Plot: `$(basename(progress_png))`")
    end
    println("Saved field-scaled run: $outdir")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    field_scaled_main()
end
