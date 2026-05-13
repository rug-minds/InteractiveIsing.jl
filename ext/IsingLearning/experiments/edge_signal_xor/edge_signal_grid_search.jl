using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include("edge_signal_xor.jl")

"""
    search_configs()

Return a targeted list of edge-signal XOR configurations.

The first `NN=1` and `NN=5` runs did not solve XOR. The `NN=5` run started with
some separation but lost it, so this grid tests three concrete hypotheses:

- stronger input/output edge couplings should transmit the boundary signal
  across the hidden layer more clearly;
- lower hidden-local scale for larger `NN` should avoid making the hidden layer
  act like one over-correlated block;
- temperature should be tied to interaction scale, but the fraction may need to
  move above or below the first value `0.025`.
"""
function search_configs()
    base = EdgeSignalXORConfig(;
        epochs = parse(Int, get(ENV, "EDGE_GRID_EPOCHS", "800")),
        log_every = parse(Int, get(ENV, "EDGE_GRID_LOG_EVERY", "100")),
        minit = parse(Int, get(ENV, "EDGE_GRID_MINIT", "4")),
        eval_repeats = parse(Int, get(ENV, "EDGE_GRID_EVAL_REPEATS", "8")),
        free_relaxation = parse(Int, get(ENV, "EDGE_GRID_FREE", "500")),
        nudged_relaxation = parse(Int, get(ENV, "EDGE_GRID_NUDGED", "500")),
        validation_relaxation = parse(Int, get(ENV, "EDGE_GRID_VALIDATION", "1500")),
        β = parse(FT, get(ENV, "EDGE_GRID_BETA", "2.0")),
        lr = parse(FT, get(ENV, "EDGE_GRID_LR", "0.002")),
        stepsize = parse(FT, get(ENV, "EDGE_GRID_STEPSIZE", "0.4")),
        skip_response = true,
    )

    specs = [
        (; hidden_nn = 1, temp_fraction = 0.015, input_hidden_scale = 0.12, hidden_output_scale = 0.12, hidden_local_scale = 0.04),
        (; hidden_nn = 1, temp_fraction = 0.025, input_hidden_scale = 0.16, hidden_output_scale = 0.16, hidden_local_scale = 0.04),
        (; hidden_nn = 1, temp_fraction = 0.05, input_hidden_scale = 0.16, hidden_output_scale = 0.16, hidden_local_scale = 0.04),
        (; hidden_nn = 2, temp_fraction = 0.015, input_hidden_scale = 0.16, hidden_output_scale = 0.16, hidden_local_scale = 0.025),
        (; hidden_nn = 2, temp_fraction = 0.025, input_hidden_scale = 0.16, hidden_output_scale = 0.16, hidden_local_scale = 0.025),
        (; hidden_nn = 2, temp_fraction = 0.05, input_hidden_scale = 0.20, hidden_output_scale = 0.20, hidden_local_scale = 0.025),
        (; hidden_nn = 3, temp_fraction = 0.025, input_hidden_scale = 0.20, hidden_output_scale = 0.20, hidden_local_scale = 0.015),
        (; hidden_nn = 3, temp_fraction = 0.05, input_hidden_scale = 0.20, hidden_output_scale = 0.20, hidden_local_scale = 0.015),
        (; hidden_nn = 5, temp_fraction = 0.025, input_hidden_scale = 0.20, hidden_output_scale = 0.20, hidden_local_scale = 0.008),
        (; hidden_nn = 5, temp_fraction = 0.05, input_hidden_scale = 0.20, hidden_output_scale = 0.20, hidden_local_scale = 0.008),
        (; hidden_nn = 5, temp_fraction = 0.025, input_hidden_scale = 0.25, hidden_output_scale = 0.25, hidden_local_scale = 0.005),
        (; hidden_nn = 5, temp_fraction = 0.08, input_hidden_scale = 0.25, hidden_output_scale = 0.25, hidden_local_scale = 0.005),
    ]

    return [EdgeSignalXORConfig(; pairs(merge(config_namedtuple(base), spec))...) for spec in specs]
end

"""Convert `EdgeSignalXORConfig` to a NamedTuple so individual fields can be overridden."""
function config_namedtuple(config::EdgeSignalXORConfig)
    names = fieldnames(EdgeSignalXORConfig)
    values = ntuple(i -> getfield(config, names[i]), length(names))
    return NamedTuple{names}(values)
end

"""Return a compact string identifying one grid configuration."""
function config_label(config::EdgeSignalXORConfig)
    return "NN$(config.hidden_nn)_T$(config.temp_fraction)_io$(config.input_hidden_scale)_h$(config.hidden_local_scale)_lr$(config.lr)"
end

"""Write a grid-level README with the ideas being tested."""
function write_grid_readme(path, configs)
    open(path, "w") do io
        println(io, "# Edge Signal XOR Grid")
        println(io)
        println(io, "This grid follows the first `NN=1` and `NN=5` runs.")
        println(io)
        println(io, "What is being tested:")
        println(io, "- stronger input/output edge couplings: `0.12` to `0.25` instead of `0.08`")
        println(io, "- smaller hidden-local coupling when `NN` is larger")
        println(io, "- temperature fractions `0.015`, `0.025`, `0.05`, and `0.08` of max column interaction")
        println(io, "- non-periodic hidden boundaries")
        println(io)
        println(io, "The main failure mode to watch is output means collapsing back toward zero or all cases sharing the same sign.")
        println(io)
        println(io, "Number of configs: `$(length(configs))`")
    end
    return path
end

"""Run the targeted grid and save one row per configuration."""
function main()
    configs = search_configs()
    rootdir = get(
        ENV,
        "EDGE_GRID_DIR",
        joinpath(@__DIR__, "runs", "edge_signal_grid_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(rootdir)
    write_grid_readme(joinpath(rootdir, "README.md"), configs)

    rows = Dict{String,Any}[]
    for (idx, config) in enumerate(configs)
        label = config_label(config)
        outdir = joinpath(rootdir, lpad(idx, 2, "0") * "_" * replace(label, "." => "p"))
        println("grid ", idx, "/", length(configs), ": ", label)
        result = train_edge_xor(config; outdir)
        push!(rows, Dict{String,Any}(
            "idx" => idx,
            "label" => label,
            "hidden_nn" => config.hidden_nn,
            "temp_fraction" => config.temp_fraction,
            "input_hidden_scale" => config.input_hidden_scale,
            "hidden_output_scale" => config.hidden_output_scale,
            "hidden_local_scale" => config.hidden_local_scale,
            "free_relaxation" => config.free_relaxation,
            "nudged_relaxation" => config.nudged_relaxation,
            "validation_relaxation" => config.validation_relaxation,
            "lr" => config.lr,
            "best_epoch" => result.best.epoch,
            "best_mse" => result.best.mse,
            "best_accuracy" => result.best.acc,
            "outdir" => outdir,
        ))
        write_csv(joinpath(rootdir, "grid_summary_partial.csv"), rows)
    end
    write_csv(joinpath(rootdir, "grid_summary.csv"), rows)
    println("Saved edge signal grid: ", rootdir)
    return (; rootdir, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
