using Dates
using Printf
using CairoMakie

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const EXT_ROOT = normpath(joinpath(@__DIR__, ".."))
const TRAIN_SCRIPT = joinpath(@__DIR__, "xor_statistical_ep_2input.jl")

const OUTDIR = get(
    ENV,
    "ISING_XOR_LANGEVIN_SEARCH_DIR",
    joinpath(EXT_ROOT, "runs", "xor_langevin_2input_search_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
)

const PRESET = Symbol(get(ENV, "ISING_XOR_LANGEVIN_SEARCH_PRESET", "focused"))
const EPOCHS = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_EPOCHS", "1000"))
const LOG_EVERY = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_LOG_EVERY", "250"))
const MINIT = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_MINIT", "4"))
const EVAL_REPEATS = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_EVAL_REPEATS", "32"))
const FREE_RELAXATION = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_FREE_RELAXATION", "1000"))
const NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_NUDGED_RELAXATION", "1000"))
const MAX_CONFIGS = parse(Int, get(ENV, "ISING_XOR_LANGEVIN_SEARCH_MAX_CONFIGS", "0"))

const COMMON_ENV = Dict(
    "ISING_XOR_2IN_STATE" => "continuous",
    "ISING_XOR_2IN_DYNAMICS" => "langevin",
    "ISING_XOR_2IN_EPOCHS" => string(EPOCHS),
    "ISING_XOR_2IN_LOG_EVERY" => string(LOG_EVERY),
    "ISING_XOR_2IN_MINIT" => string(MINIT),
    "ISING_XOR_2IN_EVAL_REPEATS" => string(EVAL_REPEATS),
    "ISING_XOR_2IN_FREE_RELAXATION" => string(FREE_RELAXATION),
    "ISING_XOR_2IN_NUDGED_RELAXATION" => string(NUDGED_RELAXATION),
    "ISING_XOR_2IN_WEIGHT_SCALE" => "0.2",
    "ISING_XOR_2IN_BIAS_SCALE" => "0.05",
    "ISING_XOR_2IN_BETA" => "1.0",
    "ISING_XOR_2IN_LR" => "0.0015",
    "ISING_XOR_2IN_WEIGHT_DECAY" => "1e-4",
    "ISING_XOR_2IN_BLOCK_SIZE" => "8",
    "ISING_XOR_2IN_WEIGHT_SEED" => "13",
    "ISING_XOR_2IN_BIAS_SEED" => "23",
    "ISING_XOR_2IN_RULE" => "ep",
    "ISING_XOR_2IN_INIT_MODE" => "random",
)

function config_list()
    if PRESET === :smoke
        return [
            (name = "T005_s005", temp = 0.05, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85000),
        ]
    elseif PRESET === :focused
        return [
            (name = "T050_s005", temp = 0.50, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85000),
            (name = "T020_s005", temp = 0.20, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85100),
            (name = "T010_s005", temp = 0.10, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85200),
            (name = "T005_s005", temp = 0.05, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85300),
            (name = "T002_s005", temp = 0.02, stepsize = 0.05, lr = 0.0015, beta = 1.0, base_seed = 85400),
            (name = "T010_s010", temp = 0.10, stepsize = 0.10, lr = 0.0015, beta = 1.0, base_seed = 85500),
            (name = "T005_s010", temp = 0.05, stepsize = 0.10, lr = 0.0015, beta = 1.0, base_seed = 85600),
            (name = "T002_s010", temp = 0.02, stepsize = 0.10, lr = 0.0015, beta = 1.0, base_seed = 85700),
        ]
    elseif PRESET === :wide
        configs = NamedTuple[]
        idx = 0
        for temp in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
            for stepsize in (0.02, 0.05, 0.1, 0.2)
                idx += 1
                name = "T" * replace(@sprintf("%.3f", temp), "." => "p") *
                    "_s" * replace(@sprintf("%.3f", stepsize), "." => "p")
                push!(configs, (;
                    name,
                    temp,
                    stepsize,
                    lr = 0.0015,
                    beta = 1.0,
                    base_seed = 86000 + 100idx,
                ))
            end
        end
        return configs
    else
        throw(ArgumentError("ISING_XOR_LANGEVIN_SEARCH_PRESET must be smoke, focused, or wide, got $(PRESET)."))
    end
end

function parse_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = split(first(lines), ",")
    rows = Vector{Dict{String,Float64}}()
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ",")
        length(fields) == length(header) || error("bad CSV row in $path: $line")
        row = Dict{String,Float64}()
        for (key, value) in zip(header, fields)
            row[key] = parse(Float64, value)
        end
        push!(rows, row)
    end
    return rows
end

function run_config(config)
    rundir = joinpath(OUTDIR, config.name)
    mkpath(rundir)
    logpath = joinpath(rundir, "run.log")
    env = copy(COMMON_ENV)
    env["ISING_XOR_2IN_DIR"] = rundir
    env["ISING_XOR_2IN_TEMP"] = string(config.temp)
    env["ISING_XOR_2IN_STEPSIZE"] = string(config.stepsize)
    env["ISING_XOR_2IN_LR"] = string(config.lr)
    env["ISING_XOR_2IN_BETA"] = string(config.beta)
    env["ISING_XOR_2IN_BASE_SEED"] = string(config.base_seed)

    cmd = `$(Base.julia_cmd()) --project=$(EXT_ROOT) $(TRAIN_SCRIPT)`
    println("running $(config.name): T=$(config.temp), stepsize=$(config.stepsize), lr=$(config.lr), beta=$(config.beta)")
    open(logpath, "w") do io
        withenv(collect(pairs(env))...) do
            run(pipeline(cmd; stdout = io, stderr = io))
        end
    end

    csvpath = joinpath(rundir, "xor_statistical_ep_2input.csv")
    rows = parse_csv(csvpath)
    best = rows[argmin([row["mse"] for row in rows])]
    final = last(rows)
    return (;
        config...,
        dir = rundir,
        csv = csvpath,
        log = logpath,
        best_epoch = best["epoch"],
        best_mse = best["mse"],
        best_accuracy = best["accuracy"],
        final_epoch = final["epoch"],
        final_mse = final["mse"],
        final_accuracy = final["accuracy"],
    )
end

function write_summary(path, results)
    keys = [
        :name, :temp, :stepsize, :lr, :beta, :base_seed,
        :best_epoch, :best_mse, :best_accuracy,
        :final_epoch, :final_mse, :final_accuracy,
        :dir,
    ]
    open(path, "w") do io
        println(io, join(string.(keys), ","))
        for result in results
            println(io, join((string(getproperty(result, key)) for key in keys), ","))
        end
    end
    return path
end

function plot_summary(path, results)
    sorted = sort(results; by = r -> r.best_mse)
    labels = [r.name for r in sorted]
    best_mse = [r.best_mse for r in sorted]
    final_mse = [r.final_mse for r in sorted]
    acc = [r.best_accuracy for r in sorted]

    fig = Figure(size = (1200, 760))
    ax1 = Axis(fig[1, 1], title = "2-input XOR Langevin search", xlabel = "config", ylabel = "MSE", xticks = (1:length(labels), labels))
    barplot!(ax1, 1:length(labels), best_mse, color = :dodgerblue, label = "best")
    scatter!(ax1, 1:length(labels), final_mse, color = :firebrick, markersize = 12, label = "final")
    axislegend(ax1, position = :rt)
    ax1.xticklabelrotation = pi / 4

    ax2 = Axis(fig[2, 1], xlabel = "config", ylabel = "best accuracy", xticks = (1:length(labels), labels), limits = (nothing, (-0.05, 1.05)))
    barplot!(ax2, 1:length(labels), acc, color = :seagreen)
    ax2.xticklabelrotation = pi / 4

    save(path, fig)
    return path
end

function main()
    mkpath(OUTDIR)
    configs = config_list()
    if MAX_CONFIGS > 0
        configs = configs[1:min(MAX_CONFIGS, length(configs))]
    end
    println("Langevin XOR search output: $OUTDIR")
    println("preset=$(PRESET), configs=$(length(configs)), epochs=$(EPOCHS), minit=$(MINIT), eval_repeats=$(EVAL_REPEATS)")

    results = NamedTuple[]
    for config in configs
        result = run_config(config)
        push!(results, result)
        println(
            "  $(result.name): best mse=$(round(result.best_mse; digits=6)) ",
            "acc=$(result.best_accuracy) at epoch=$(Int(result.best_epoch)); ",
            "final mse=$(round(result.final_mse; digits=6)) acc=$(result.final_accuracy)",
        )
    end

    summary_path = write_summary(joinpath(OUTDIR, "langevin_search_summary.csv"), results)
    plot_path = plot_summary(joinpath(OUTDIR, "langevin_search_summary.png"), results)
    best = results[argmin([r.best_mse for r in results])]
    println("best config: $(best.name), mse=$(best.best_mse), acc=$(best.best_accuracy), dir=$(best.dir)")
    println("summary: $summary_path")
    println("plot: $plot_path")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
