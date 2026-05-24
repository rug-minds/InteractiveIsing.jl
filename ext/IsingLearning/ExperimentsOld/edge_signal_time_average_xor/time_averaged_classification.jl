using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "edge_signal_xor", "edge_signal_xor.jl"))

"""
    OutputAverageTrace

Mutable storage for one time-averaged output measurement.
"""
mutable struct OutputAverageTrace
    sum::FT
    sumsq::FT
    count::Int
    samples::Vector{FT}
end

"""Create an empty output-average trace."""
OutputAverageTrace() = OutputAverageTrace(zero(FT), zero(FT), 0, FT[])

"""Reset an output-average trace before reusing it for another input/repeat."""
function reset!(trace::OutputAverageTrace)
    trace.sum = zero(FT)
    trace.sumsq = zero(FT)
    trace.count = 0
    empty!(trace.samples)
    return trace
end

"""Return the mean of the recorded output samples."""
output_mean(trace::OutputAverageTrace) = trace.count == 0 ? FT(NaN) : trace.sum / trace.count

"""Return the sample standard deviation of the recorded output samples."""
function output_std(trace::OutputAverageTrace)
    trace.count <= 1 && return zero(FT)
    μ = output_mean(trace)
    return sqrt(max(zero(FT), trace.sumsq / trace.count - μ^2))
end

"""
    OutputAverager(trace, output_idx)

ProcessAlgorithm that records the current output spin whenever it is scheduled.
Schedule it after a relaxation/burn-in period to average the fluctuating output
observable instead of reading one final state.
"""
struct OutputAverager{Trace} <: Processes.ProcessAlgorithm
    trace::Trace
    output_idx::Int
end

Processes.init(::OutputAverager, context) = (;)

function Processes.step!(averager::OutputAverager, context)
    value = FT(II.state(context.model)[averager.output_idx])
    averager.trace.sum += value
    averager.trace.sumsq += value^2
    averager.trace.count += 1
    push!(averager.trace.samples, value)
    return (; output_average = output_mean(averager.trace))
end

"""
    TimeAverageConfig(; kwargs...)

Configuration for the fixed-graph time-averaged XOR classification diagnostic.
"""
Base.@kwdef struct TimeAverageConfig
    graph_path::String = get(
        ENV,
        "EDGE_TIMEAVG_GRAPH",
        joinpath(@__DIR__, "..", "edge_signal_xor", "runs", "edge_signal_grid_20260512_154608", "05_NN2_T0p025_io0p16_h0p025_lr0p002", "best_graph.jld2"),
    )
    burnin_sweeps::Vector{Int} = parse_int_list(get(ENV, "EDGE_TIMEAVG_BURNINS", "25,50,100,250,500"))
    average_sweeps::Vector{Int} = parse_int_list(get(ENV, "EDGE_TIMEAVG_AVGS", "50"))
    sample_every_sweeps::Int = parse(Int, get(ENV, "EDGE_TIMEAVG_EVERY", "1"))
    repeats::Int = parse(Int, get(ENV, "EDGE_TIMEAVG_REPEATS", "16"))
    stepsize::FT = parse(FT, get(ENV, "EDGE_TIMEAVG_STEPSIZE", "0.4"))
    temperature_factors::Vector{FT} = parse_float_list(get(ENV, "EDGE_TIMEAVG_TEMP_FACTORS", "1.0,0.5,0.25,0.1"))
    base_seed::Int = parse(Int, get(ENV, "EDGE_TIMEAVG_SEED", "515000"))
    outdir::String = get(
        ENV,
        "EDGE_TIMEAVG_DIR",
        joinpath(@__DIR__, "runs", "time_average_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

"""Parse a comma-separated integer list from an environment variable."""
parse_int_list(s::AbstractString) = [parse(Int, strip(part)) for part in split(s, ",") if !isempty(strip(part))]

"""Parse a comma-separated floating-point list from an environment variable."""
parse_float_list(s::AbstractString) = [parse(FT, strip(part)) for part in split(s, ",") if !isempty(strip(part))]

"""Return a LocalLangevin sampler for the time-average diagnostic."""
function timeavg_sampler(config::TimeAverageConfig)
    return II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
end

"""
    run_burnin_and_average!(graph, x, config, burnin_sweeps, average_sweeps; seed)

Randomize the graph, apply one XOR input, run burn-in dynamics, then average the
output spin for `average_sweeps` full sweeps. The averaging process is scheduled
once per `config.sample_every_sweeps` full sweeps.
"""
function run_burnin_and_average!(graph, x, config::TimeAverageConfig, burnin_sweeps::Int, average_sweeps::Int; seed::Int)
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, x)

    sweep_steps = length(II.sampling_indices(graph.index_set))
    sampler = timeavg_sampler(config)

    burnin_steps = burnin_sweeps * sweep_steps
    burnin_steps > 0 && run_dynamics_steps!(graph, sampler, burnin_steps; seed = seed + 1)

    trace = OutputAverageTrace()
    output_idx = only(II.layerrange(graph[end]))
    averager = OutputAverager(trace, output_idx)
    dynamics = deepcopy(sampler)
    sample_interval = config.sample_every_sweeps * sweep_steps
    total_steps = average_sweeps * sweep_steps
    routine = Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @every sample_interval averager(model = dynamics.model)
    end
    wrapped = Processes.@Routine begin
        @repeat total_steps routine()
    end
    inputs = (Processes.Init(dynamics, model = graph, rng = Random.MersenneTwister(seed + 2)),)
    process = Processes.Process(Processes.resolve(wrapped), inputs...; repeats = 1)
    run(process)
    wait(process)
    close(process)

    return (mean = output_mean(trace), std = output_std(trace), nsamples = trace.count)
end

"""Evaluate all four XOR cases for one burn-in/averaging configuration."""
function evaluate_timeavg_setting(config::TimeAverageConfig, base_graph, x, y, burnin_sweeps::Int, average_sweeps::Int, temp_factor::FT)
    graph = deepcopy(base_graph)
    II.temp!(graph, FT(base_graph.temp * temp_factor))

    means = zeros(FT, size(x, 2))
    within_stds = zeros(FT, size(x, 2))
    repeat_stds = zeros(FT, size(x, 2))
    nsamples = 0
    for sample_idx in axes(x, 2)
        repeat_means = zeros(FT, config.repeats)
        repeat_within = zeros(FT, config.repeats)
        for repeat_idx in 1:config.repeats
            result = run_burnin_and_average!(
                graph,
                view(x, :, sample_idx),
                config,
                burnin_sweeps,
                average_sweeps;
                seed = config.base_seed + 1_000_000 * sample_idx + 10_000 * repeat_idx + 100 * burnin_sweeps + average_sweeps,
            )
            repeat_means[repeat_idx] = result.mean
            repeat_within[repeat_idx] = result.std
            nsamples = result.nsamples
        end
        means[sample_idx] = mean(repeat_means)
        within_stds[sample_idx] = mean(repeat_within)
        repeat_stds[sample_idx] = std(repeat_means)
    end

    targets = vec(y)
    return (;
        temp = II.temp(graph),
        temp_factor,
        burnin_sweeps,
        average_sweeps,
        sample_every_sweeps = config.sample_every_sweeps,
        repeats = config.repeats,
        nsamples,
        mse = mean(abs2, means .- targets),
        accuracy = mean(sign.(means) .== sign.(targets)),
        margin = minimum(abs.(means)),
        means,
        within_stds,
        repeat_stds,
    )
end

"""Append one summary row to the result table."""
function push_timeavg_row!(rows, metrics)
    row = Dict{String,Any}(
        "temp" => metrics.temp,
        "temp_factor" => metrics.temp_factor,
        "burnin_sweeps" => metrics.burnin_sweeps,
        "average_sweeps" => metrics.average_sweeps,
        "sample_every_sweeps" => metrics.sample_every_sweeps,
        "repeats" => metrics.repeats,
        "nsamples" => metrics.nsamples,
        "mse" => metrics.mse,
        "accuracy" => metrics.accuracy,
        "margin" => metrics.margin,
    )
    for i in eachindex(metrics.means)
        row["mean_$i"] = metrics.means[i]
        row["within_std_$i"] = metrics.within_stds[i]
        row["repeat_std_$i"] = metrics.repeat_stds[i]
    end
    push!(rows, row)
    return rows
end

"""Plot MSE versus burn-in for each average length and temperature factor."""
function plot_timeavg_results(path, rows)
    fig = Figure(size = (1200, 760))
    ax1 = Axis(fig[1, 1], title = "time-averaged output MSE", xlabel = "burn-in full sweeps", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "burn-in full sweeps", ylabel = "accuracy")
    ax3 = Axis(fig[2, 1], title = "minimum output margin", xlabel = "burn-in full sweeps", ylabel = "min |mean output|")
    ax4 = Axis(fig[2, 2], title = "repeat-to-repeat std", xlabel = "burn-in full sweeps", ylabel = "mean std(mean output)")

    groups = Dict{Tuple{FT,Int},Vector{Dict{String,Any}}}()
    for row in rows
        key = (FT(row["temp_factor"]), Int(row["average_sweeps"]))
        push!(get!(groups, key, Dict{String,Any}[]), row)
    end
    for key in sort!(collect(keys(groups)); by = x -> (x[1], x[2]))
        group_rows = sort!(groups[key]; by = row -> row["burnin_sweeps"])
        label = "T×$(key[1]), avg=$(key[2])"
        xs = [row["burnin_sweeps"] for row in group_rows]
        lines!(ax1, xs, [row["mse"] for row in group_rows], label = label)
        lines!(ax2, xs, [row["accuracy"] for row in group_rows], label = label)
        lines!(ax3, xs, [row["margin"] for row in group_rows], label = label)
        lines!(ax4, xs, [mean([row["repeat_std_$i"] for i in 1:4]) for row in group_rows], label = label)
    end
    axislegend(ax1, position = :rt)
    save(path, fig)
    return path
end

"""Write a short Markdown summary for the run."""
function write_summary(path, config::TimeAverageConfig, rows)
    best = first(sort(rows; by = row -> (row["mse"], -row["accuracy"])))
    open(path, "w") do io
        println(io, "# Time-Averaged Edge XOR Classification")
        println(io)
        println(io, "This run loads a fixed learned graph and classifies each XOR input by averaging")
        println(io, "the output spin after a burn-in period. The output averager is a")
        println(io, "`ProcessAlgorithm` scheduled together with the dynamics.")
        println(io)
        println(io, "Graph: `$(config.graph_path)`")
        println(io)
        println(io, "Best tested setting:")
        println(io)
        println(io, "| field | value |")
        println(io, "|---|---:|")
        for field in ("mse", "accuracy", "margin", "temp_factor", "temp", "burnin_sweeps", "average_sweeps", "nsamples", "repeats")
            println(io, "| `$field` | `$(best[field])` |")
        end
        println(io)
        println(io, "Averaging reduces stochastic readout noise, but it does not by itself change")
        println(io, "the learned energy landscape. If the averaged output remains near zero, the")
        println(io, "graph still has weak class separation; if the average is stable but wrong, the")
        println(io, "attractor for that input is wrong.")
    end
    return path
end

"""Run the full fixed-graph time-average grid and save CSV/PNG/README outputs."""
function main()
    config = TimeAverageConfig()
    mkpath(config.outdir)
    base_graph = II.load_isinggraph(config.graph_path)
    x, y = xor_dataset(EdgeSignalXORConfig())
    rows = Dict{String,Any}[]

    for temp_factor in config.temperature_factors
        for burnin_sweeps in config.burnin_sweeps
            for average_sweeps in config.average_sweeps
                metrics = evaluate_timeavg_setting(config, base_graph, x, y, burnin_sweeps, average_sweeps, temp_factor)
                push_timeavg_row!(rows, metrics)
                println(
                    "T×", temp_factor,
                    " burn=", burnin_sweeps,
                    " avg=", average_sweeps,
                    " mse=", round(metrics.mse, digits = 6),
                    " acc=", metrics.accuracy,
                    " margin=", round(metrics.margin, digits = 4),
                    " means=", round.(metrics.means, digits = 3),
                )
            end
        end
    end

    csv_path = write_csv(joinpath(config.outdir, "time_average_results.csv"), rows)
    png_path = plot_timeavg_results(joinpath(config.outdir, "time_average_results.png"), rows)
    md_path = write_summary(joinpath(config.outdir, "README.md"), config, rows)
    println("Saved time-average diagnostic: ", config.outdir)
    return (; config, csv_path, png_path, md_path, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
