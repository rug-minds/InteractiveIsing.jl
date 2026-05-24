using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random
using Serialization
using Statistics

const II = IsingLearning.InteractiveIsing
const MNIST_NN_FT = Float32
const MNIST_NN_HIDDEN1_SCALE_REF = Ref(MNIST_NN_FT(0.01))
const MNIST_NN_HIDDEN2_SCALE_REF = Ref(MNIST_NN_FT(0.01))
const MNIST_NN_OUTPUT_SCALE_REF = Ref(MNIST_NN_FT(0.01))
const MNIST_NN_HIDDEN1_RNG_REF = Ref(Random.MersenneTwister(1))
const MNIST_NN_HIDDEN2_RNG_REF = Ref(Random.MersenneTwister(2))
const MNIST_NN_OUTPUT_RNG_REF = Ref(Random.MersenneTwister(3))

Base.@kwdef struct MNISTLocalNNConfig
    name::String = "r5_open"
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_EPOCHS", "5"))
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_WORKERS", "32"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_BATCHSIZE", "256"))
    train_limit_text::String = get(ENV, "ISING_MNIST_LOCAL_NN_TRAIN_LIMIT", "1024")
    validation_limit_text::String = get(ENV, "ISING_MNIST_LOCAL_NN_VALIDATION_LIMIT", "256")
    train_eval_limit::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_TRAIN_EVAL_LIMIT", "256"))
    input_side::Int = D_MNIST
    hidden1_side::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_H1_SIDE", "28"))
    hidden2_side::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_H2_SIDE", "11"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_OUTPUT_REPLICAS", "4"))
    local_radius::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_RADIUS", "5"))
    hidden_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_INTERNAL_RADIUS", "1"))
    output_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_OUTPUT_INTERNAL_RADIUS", "1"))
    hidden_periodic::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_LOCAL_NN_HIDDEN_PERIODIC", "false")))
    sweeps::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_SWEEPS", "50"))
    β::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_BETA", "0.1"))
    lr::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_LR", "0.003"))
    temp::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_TEMP", "0.001"))
    stepsize::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_STEPSIZE", "0.5"))
    max_drift_fraction::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_MAX_DRIFT", "0.15"))
    inter_weight_scale::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_INTER_SCALE", "0.005"))
    hidden_internal_scale::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_HIDDEN_INTERNAL_SCALE", "0.01"))
    output_internal_scale::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_OUTPUT_INTERNAL_SCALE", "0.01"))
    output_weight_scale::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_OUTPUT_SCALE", "0.005"))
    bias_scale::MNIST_NN_FT = parse(MNIST_NN_FT, get(ENV, "ISING_MNIST_LOCAL_NN_BIAS_SCALE", "0.0"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_SEED", "5301"))
    show_progress::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_LOCAL_NN_PROGRESS", "true")))
    outdir::String = get(
        ENV,
        "ISING_MNIST_LOCAL_NN_OUTDIR",
        joinpath(@__DIR__, "runs", "mnist_local_nn_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

"""Parse a comma-separated integer list, returning `default` for an empty string."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Parse a comma-separated boolean list, returning `default` for an empty string."""
function parse_bool_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Bool}}
    isempty(strip(value)) && return default
    return [parse(Bool, lowercase(strip(part))) for part in split(value, ",") if !isempty(strip(part))]
end

"""Parse an optional positive integer; empty strings and non-positive values mean `nothing`."""
function optional_positive_int(value::S) where {S<:AbstractString}
    isempty(strip(value)) && return nothing
    parsed = parse(Int, strip(value))
    return parsed > 0 ? parsed : nothing
end

"""Return a copy of `config` with selected keyword fields replaced."""
function copy_config(config::C; kwargs...) where {C<:MNISTLocalNNConfig}
    fields = Dict{Symbol,Any}(field => getfield(config, field) for field in fieldnames(C))
    for (field, value) in kwargs
        fields[field] = value
    end
    return MNISTLocalNNConfig(; fields...)
end

"""Append one named-tuple row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return a compact 2D display shape for a flat output layer."""
function factor_shape(units::I) where {I<:Integer}
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Topology that maps differently sized image-like layers onto the MNIST pixel frame."""
function mnist_image_topology(side::I, reference_side::J; periodic::Bool) where {I<:Integer,J<:Integer}
    scale = side <= 1 ? 1.0 : Float64(reference_side - 1) / Float64(side - 1)
    origin = ntuple(_ -> 1.0 - scale, 2)
    return II.SquareTopology((Int(side), Int(side)); origin, periodic), II.LatticeConstants(scale, scale)
end

"""Return true when two world coordinates fall inside one square local fanout."""
function inside_local_window(c1::C1, c2::C2, radius::I) where {C1,C2,I<:Integer}
    r = Float64(radius) + 1e-6
    return abs(c1[1] - c2[1]) <= r && abs(c1[2] - c2[2]) <= r
end

"""Generate sparse inter-layer couplings by filtering all pairs with a local square window."""
function inter_layer_generator(radius::I, scale::T, rng::R) where {I<:Integer,T<:Real,R<:Random.AbstractRNG}
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> inside_local_window(c1, c2, radius) ? MNIST_NN_FT(scale) * randn(rng, MNIST_NN_FT) : zero(MNIST_NN_FT),
        rng,
    )
end

"""Generate dense hidden-to-output readout couplings."""
function output_readout_generator(scale::T, rng::R) where {T<:Real,R<:Random.AbstractRNG}
    return II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> MNIST_NN_FT(scale) * randn(rng, MNIST_NN_FT), rng)
end

"""Named hidden-1 local recurrent callback used by `WeightGenerator`."""
function hidden1_internal_weight(; dr::DR, c1::C1, c2::C2, dc::DC) where {DR,C1,C2,DC}
    return MNIST_NN_HIDDEN1_SCALE_REF[] * randn(MNIST_NN_HIDDEN1_RNG_REF[], MNIST_NN_FT)
end

"""Named hidden-2 local recurrent callback used by `WeightGenerator`."""
function hidden2_internal_weight(; dr::DR, c1::C1, c2::C2, dc::DC) where {DR,C1,C2,DC}
    return MNIST_NN_HIDDEN2_SCALE_REF[] * randn(MNIST_NN_HIDDEN2_RNG_REF[], MNIST_NN_FT)
end

"""Named output local recurrent callback used by `WeightGenerator`."""
function output_internal_weight(; dr::DR, c1::C1, c2::C2, dc::DC) where {DR,C1,C2,DC}
    return MNIST_NN_OUTPUT_SCALE_REF[] * randn(MNIST_NN_OUTPUT_RNG_REF[], MNIST_NN_FT)
end

"""Generate trainable local recurrent hidden-1 couplings."""
function hidden1_internal_generator(radius::I, scale::T, rng::R) where {I<:Integer,T<:Real,R<:Random.AbstractRNG}
    MNIST_NN_HIDDEN1_SCALE_REF[] = MNIST_NN_FT(scale)
    MNIST_NN_HIDDEN1_RNG_REF[] = rng
    return II.WeightGenerator(hidden1_internal_weight, Int(radius), rng; symmetric = true)
end

"""Generate trainable local recurrent hidden-2 couplings."""
function hidden2_internal_generator(radius::I, scale::T, rng::R) where {I<:Integer,T<:Real,R<:Random.AbstractRNG}
    MNIST_NN_HIDDEN2_SCALE_REF[] = MNIST_NN_FT(scale)
    MNIST_NN_HIDDEN2_RNG_REF[] = rng
    return II.WeightGenerator(hidden2_internal_weight, Int(radius), rng; symmetric = true)
end

"""Generate trainable local recurrent output couplings."""
function output_internal_generator(radius::I, scale::T, rng::R) where {I<:Integer,T<:Real,R<:Random.AbstractRNG}
    MNIST_NN_OUTPUT_SCALE_REF[] = MNIST_NN_FT(scale)
    MNIST_NN_OUTPUT_RNG_REF[] = rng
    return II.WeightGenerator(output_internal_weight, Int(radius), rng; symmetric = true)
end

"""Build a local MNIST graph with local image fanout and dense replicated readout."""
function mnist_local_nn_graph(config::C) where {C<:MNISTLocalNNConfig}
    rng_inter_01 = Random.MersenneTwister(config.seed)
    rng_inter_12 = Random.MersenneTwister(config.seed + 1)
    rng_output = Random.MersenneTwister(config.seed + 2)
    rng_hidden_1 = Random.MersenneTwister(config.seed + 3)
    rng_hidden_2 = Random.MersenneTwister(config.seed + 4)
    rng_output_internal = Random.MersenneTwister(config.seed + 5)
    rng_b = Random.MersenneTwister(config.seed + 6)

    input_topology, input_lattice = mnist_image_topology(config.input_side, config.input_side; periodic = false)
    hidden1_topology, hidden1_lattice = mnist_image_topology(config.hidden1_side, config.input_side; periodic = config.hidden_periodic)
    hidden2_topology, hidden2_lattice = mnist_image_topology(config.hidden2_side, config.input_side; periodic = config.hidden_periodic)
    output_rows, output_cols = factor_shape(MNIST_NCLASSES * config.output_replicas)

    input = II.Layer(
        config.input_side,
        config.input_side,
        II.StateSet(-one(MNIST_NN_FT), one(MNIST_NN_FT)),
        II.Continuous(),
        input_topology,
        input_lattice,
        II.Coords(0, 0, 0);
        periodic = false,
    )
    hidden1 = II.Layer(
        config.hidden1_side,
        config.hidden1_side,
        II.StateSet(-one(MNIST_NN_FT), one(MNIST_NN_FT)),
        hidden1_internal_generator(config.hidden_internal_radius, config.hidden_internal_scale, rng_hidden_1),
        II.Continuous(),
        hidden1_topology,
        hidden1_lattice,
        II.Coords(0, config.input_side + 8, 0);
        periodic = config.hidden_periodic,
    )
    hidden2 = II.Layer(
        config.hidden2_side,
        config.hidden2_side,
        II.StateSet(-one(MNIST_NN_FT), one(MNIST_NN_FT)),
        hidden2_internal_generator(config.hidden_internal_radius, config.hidden_internal_scale, rng_hidden_2),
        II.Continuous(),
        hidden2_topology,
        hidden2_lattice,
        II.Coords(0, config.input_side + config.hidden1_side + 16, 0);
        periodic = config.hidden_periodic,
    )
    output = II.Layer(
        output_rows,
        output_cols,
        II.StateSet(-one(MNIST_NN_FT), one(MNIST_NN_FT)),
        output_internal_generator(config.output_internal_radius, config.output_internal_scale, rng_output_internal),
        II.Continuous(),
        II.Coords(0, config.input_side + config.hidden1_side + config.hidden2_side + 24, 0);
        periodic = false,
    )

    bias = g -> config.bias_scale .* randn(rng_b, MNIST_NN_FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(MNIST_NN_FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(MNIST_NN_FT), II.statelen(g))
    graph = II.IsingGraph(
        input,
        inter_layer_generator(config.local_radius, config.inter_weight_scale, rng_inter_01),
        hidden1,
        inter_layer_generator(config.local_radius, config.inter_weight_scale, rng_inter_12),
        hidden2,
        output_readout_generator(config.output_weight_scale, rng_output),
        output,
        II.Bilinear() + II.MagField(b = bias) + II.Clamping(β = II.UniformArray(zero(MNIST_NN_FT)), y = target, mask = mask);
        precision = MNIST_NN_FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Return the number of non-input spins stepped when MNIST is supplied as a field."""
function active_units(graph::G) where {G}
    return sum(length(II.layerrange(graph[layer_idx])) for layer_idx in 2:length(graph))
end

"""Wrap one local MNIST graph in the standard manager-compatible Learning layer."""
function mnist_local_nn_layer(graph::G, config::C) where {G,C<:MNISTLocalNNConfig}
    relaxation_steps = max(1, round(Int, config.sweeps * active_units(graph)))
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        adjusted = false,
        order = :cyclic,
    )
    return LayeredIsingGraphLayer(
        graph;
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps,
        free_relaxation_steps = relaxation_steps,
        nudged_relaxation_steps = relaxation_steps,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Return the grid of local-radii and hidden-periodicity configurations."""
function experiment_configs(base::C) where {C<:MNISTLocalNNConfig}
    radii = parse_int_list(get(ENV, "ISING_MNIST_LOCAL_NN_RADII", "1,2,3,5,7,10"), [1, 2, 3, 5, 7, 10])
    periodic_values = parse_bool_list(get(ENV, "ISING_MNIST_LOCAL_NN_PERIODIC", "false"), [false])
    limit = parse(Int, get(ENV, "ISING_MNIST_LOCAL_NN_LIMIT", "0"))

    configs = MNISTLocalNNConfig[]
    for periodic in periodic_values, radius in radii
        name = "r$(radius)_$(periodic ? "periodic" : "open")"
        seed = base.seed + 97 * radius + (periodic ? 10_000 : 0)
        push!(configs, copy_config(base; name, local_radius = radius, hidden_periodic = periodic, seed))
    end
    if limit > 0
        resize!(configs, min(limit, length(configs)))
    end
    return configs
end

"""Read one metric value from a possibly missing train or validation result."""
function metric_or_missing(stat::S, split::Symbol, metric::Symbol) where {S<:NamedTuple}
    result = getproperty(stat, split)
    isnothing(result) && return missing
    return getproperty(result, metric)
end

"""Convert one `fit_mnist_threaded!` stats vector into aggregate CSV rows."""
function stats_rows(config::C, stats::S, elapsed::Real, run_dir::P) where {C<:MNISTLocalNNConfig,S<:AbstractVector,P<:AbstractString}
    rows = NamedTuple[]
    for stat in stats
        push!(
            rows,
            (;
                config = config.name,
                radius = config.local_radius,
                periodic = config.hidden_periodic,
                epoch = stat.epoch,
                nbatches = stat.nbatches,
                train_accuracy = metric_or_missing(stat, :train, :accuracy),
                train_error = metric_or_missing(stat, :train, :classification_error),
                train_mse = metric_or_missing(stat, :train, :mean_squared_error),
                validation_accuracy = metric_or_missing(stat, :validation, :accuracy),
                validation_error = metric_or_missing(stat, :validation, :classification_error),
                validation_mse = metric_or_missing(stat, :validation, :mean_squared_error),
                elapsed_seconds = elapsed,
                run_dir,
            ),
        )
    end
    return rows
end

"""Choose the best epoch using validation accuracy when available, otherwise train accuracy."""
function summary_row(config::C, rows::R, checkpoint::P) where {C<:MNISTLocalNNConfig,R<:AbstractVector,P<:AbstractString}
    best_epoch = 0
    best_accuracy = -Inf
    best_mse = Inf
    for row in rows
        accuracy = ismissing(row.validation_accuracy) ? row.train_accuracy : row.validation_accuracy
        mse = ismissing(row.validation_mse) ? row.train_mse : row.validation_mse
        ismissing(accuracy) && continue
        if accuracy > best_accuracy || (accuracy == best_accuracy && !ismissing(mse) && mse < best_mse)
            best_epoch = row.epoch
            best_accuracy = Float64(accuracy)
            best_mse = ismissing(mse) ? Inf : Float64(mse)
        end
    end
    return (;
        config = config.name,
        radius = config.local_radius,
        periodic = config.hidden_periodic,
        best_epoch,
        best_accuracy,
        best_mse,
        checkpoint,
    )
end

"""Serialize the final graph parameters and run metadata for one configuration."""
function save_checkpoint(path::P, trainer::T, config::C, stats::S, relaxation_steps::I) where {P<:AbstractString,T,C<:MNISTLocalNNConfig,S,I<:Integer}
    open(path, "w") do io
        serialize(io, (;
            architecture = "$(config.input_side)^2 -> $(config.hidden1_side)^2 -> $(config.hidden2_side)^2 -> $(MNIST_NCLASSES * config.output_replicas)",
            params = trainer.params,
            stats,
            config,
            relaxation_steps = Int(relaxation_steps),
        ))
    end
    return path
end

"""Run one configured MNIST local-radius experiment."""
function run_config!(config::C, root_outdir::P) where {C<:MNISTLocalNNConfig,P<:AbstractString}
    run_dir = joinpath(root_outdir, config.name)
    mkpath(run_dir)

    graph = mnist_local_nn_graph(config)
    layer = mnist_local_nn_layer(graph, config)
    relaxation_steps = layer.free_relaxation_steps
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = config.workers,
        optimiser = Optimisers.Adam(config.lr),
        share_static_model_data = true,
        input_mode = :field,
    )

    train_limit = optional_positive_int(config.train_limit_text)
    validation_limit = optional_positive_int(config.validation_limit_text)
    stats = nothing
    elapsed = 0.0
    try
        elapsed = @elapsed begin
            _, stats = fit_mnist_threaded!(
                trainer;
                epochs = config.epochs,
                batchsize = config.batchsize,
                split = :train,
                validation_split = isnothing(validation_limit) ? nothing : :test,
                shuffle = true,
                rng = Random.MersenneTwister(config.seed + 20),
                limit = train_limit,
                validation_limit = validation_limit,
                show_progress = config.show_progress,
                show_validation_progress = false,
                train_eval_limit = config.train_eval_limit,
            )
        end
        checkpoint = save_checkpoint(joinpath(run_dir, "final_params.bin"), trainer, config, stats, relaxation_steps)
        rows = stats_rows(config, stats, elapsed, run_dir)
        for row in rows
            append_row!(joinpath(run_dir, "metrics.csv"), row)
        end
        return (; rows, summary = summary_row(config, rows, checkpoint))
    finally
        close_trainer!(trainer)
    end
end

"""Plot accuracy, MSE, and best validation result across local radii."""
function plot_results(outdir::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    isempty(rows) && return nothing

    fig = Figure(size = (1350, 850))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "MNIST accuracy")
    ax_mse = Axis(fig[2, 1], xlabel = "epoch", ylabel = "MSE", title = "MNIST output MSE")
    ax_best = Axis(fig[1:2, 2], xlabel = "configuration", ylabel = "best accuracy", title = "Best validation accuracy")

    configs = unique(row.config for row in rows)
    palette = Makie.wong_colors()
    for (idx, config_name) in enumerate(configs)
        subset = [row for row in rows if row.config == config_name]
        color = palette[mod1(idx, length(palette))]
        accuracy = [ismissing(row.validation_accuracy) ? row.train_accuracy : row.validation_accuracy for row in subset]
        mse = [ismissing(row.validation_mse) ? row.train_mse : row.validation_mse for row in subset]
        valid_acc = [!ismissing(value) for value in accuracy]
        valid_mse = [!ismissing(value) for value in mse]
        any(valid_acc) && lines!(ax_acc, [row.epoch for row in subset][valid_acc], Float64.(accuracy[valid_acc]), color = color, label = config_name)
        any(valid_mse) && lines!(ax_mse, [row.epoch for row in subset][valid_mse], Float64.(mse[valid_mse]), color = color)
    end

    sorted = sort(summary_rows; by = row -> row.best_accuracy, rev = true)
    xvals = 1:length(sorted)
    barplot!(ax_best, xvals, [row.best_accuracy for row in sorted], color = :steelblue)
    ax_best.xticks = (xvals, [row.config for row in sorted])
    ax_best.xticklabelrotation = pi / 3
    axislegend(ax_acc, position = :rb, nbanks = 2)

    path = joinpath(outdir, "mnist_local_nn_summary.png")
    save(path, fig)
    return path
end

"""Write a compact markdown note describing the sweep and its outputs."""
function write_summary_note!(path::P, base::C, summary_rows::S, plot_path) where {P<:AbstractString,C<:MNISTLocalNNConfig,S<:AbstractVector}
    open(path, "w") do io
        println(io, "# MNIST Local NN Grid")
        println(io)
        println(io, "Manager-backed MNIST comparison for local square fanout radii.")
        println(io)
        println(io, "- architecture: `$(base.input_side)x$(base.input_side) -> $(base.hidden1_side)x$(base.hidden1_side) -> $(base.hidden2_side)x$(base.hidden2_side) -> $(MNIST_NCLASSES * base.output_replicas)`")
        println(io, "- inter-layer local radii: `$(get(ENV, "ISING_MNIST_LOCAL_NN_RADII", "1,2,3,5,7,10"))`")
        println(io, "- hidden periodic values: `$(get(ENV, "ISING_MNIST_LOCAL_NN_PERIODIC", "false"))`")
        println(io, "- hidden internal radius: `$(base.hidden_internal_radius)`")
        println(io, "- output internal radius: `$(base.output_internal_radius)`")
        println(io, "- workers / batchsize / epochs: `$(base.workers)` / `$(base.batchsize)` / `$(base.epochs)`")
        println(io, "- train/validation limits: `$(base.train_limit_text)` / `$(base.validation_limit_text)`")
        println(io, "- sweeps: `$(base.sweeps)` active-layer sweeps per free/nudged phase")
        println(io, "- input mode: `:field`, with shared static model data")
        println(io)
        println(io, "## Best Results")
        println(io)
        println(io, "| Rank | Config | Radius | Periodic | Best Accuracy | Best MSE | Best Epoch |")
        println(io, "|---:|---|---:|---|---:|---:|---:|")
        for (rank, row) in enumerate(sort(summary_rows; by = row -> row.best_accuracy, rev = true))
            println(io, "| $rank | `$(row.config)` | $(row.radius) | $(row.periodic) | $(round(row.best_accuracy; digits = 4)) | $(round(row.best_mse; digits = 6)) | $(row.best_epoch) |")
        end
        println(io)
        !isnothing(plot_path) && println(io, "Plot: `$(basename(plot_path))`")
        println(io, "Metrics: `metrics.csv`")
        println(io, "Summary: `summary.csv`")
    end
    return path
end

"""Run the full MNIST local-radius comparison."""
function main()
    base = MNISTLocalNNConfig()
    base.workers > 0 || throw(ArgumentError("ISING_MNIST_LOCAL_NN_WORKERS must be positive"))
    base.batchsize > 0 || throw(ArgumentError("ISING_MNIST_LOCAL_NN_BATCHSIZE must be positive"))
    base.output_replicas > 0 || throw(ArgumentError("ISING_MNIST_LOCAL_NN_OUTPUT_REPLICAS must be positive"))
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers

    configs = experiment_configs(base)
    isempty(configs) && throw(ArgumentError("no MNIST local NN configs selected"))
    mkpath(base.outdir)
    for filename in ("metrics.csv", "summary.csv")
        path = joinpath(base.outdir, filename)
        isfile(path) && rm(path)
    end

    all_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    println("Running $(length(configs)) MNIST local NN config(s)")
    println("threads = $(Threads.nthreads()), workers = $(base.workers), batchsize = $(base.batchsize), epochs = $(base.epochs)")
    for (idx, config) in enumerate(configs)
        println("[$idx/$(length(configs))] $(config.name)")
        result = run_config!(config, base.outdir)
        append!(all_rows, result.rows)
        push!(summary_rows, result.summary)
        for row in result.rows
            append_row!(joinpath(base.outdir, "metrics.csv"), row)
        end
        append_row!(joinpath(base.outdir, "summary.csv"), result.summary)
    end

    plot_path = plot_results(base.outdir, all_rows, summary_rows)
    note_path = write_summary_note!(joinpath(base.outdir, "README.md"), base, summary_rows, plot_path)
    println("saved metrics ", joinpath(base.outdir, "metrics.csv"))
    println("saved summary ", joinpath(base.outdir, "summary.csv"))
    !isnothing(plot_path) && println("saved plot ", plot_path)
    println("saved note ", note_path)
    return (; outdir = base.outdir, rows = all_rows, summary_rows, plot_path, note_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
