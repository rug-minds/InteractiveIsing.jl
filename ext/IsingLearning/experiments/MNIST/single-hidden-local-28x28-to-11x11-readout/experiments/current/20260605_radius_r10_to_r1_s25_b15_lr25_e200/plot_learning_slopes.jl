using Dates
using Pkg
using Statistics

Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", "..", ".."))

using CairoMakie

"""
    csv_rows(path)

Read a simple metrics CSV into named tuples. The experiment metrics do not use
quoted comma-containing cells, so a small parser is enough here.
"""
function csv_rows(path::P) where {P<:AbstractString}
    lines = filter(!isempty, readlines(path))
    isempty(lines) && return NamedTuple[]
    header = Symbol.(split(lines[1], ','))
    rows = NamedTuple[]
    for line in lines[2:end]
        cells = split(line, ',')
        length(cells) == length(header) || continue
        push!(rows, NamedTuple{Tuple(header)}(Tuple(cells)))
    end
    return rows
end

"""
    parse_metric(value)

Parse numeric metrics while preserving missing-like cells as `missing`.
"""
function parse_metric(value)
    s = strip(String(value))
    (isempty(s) || s == "missing") && return missing
    return tryparse(Float64, s)
end

"""
    run_label(root, metrics_path)

Return a stable run label relative to the experiment series root.
"""
function run_label(root::R, metrics_path::P) where {R<:AbstractString,P<:AbstractString}
    dir = dirname(metrics_path)
    rel = relpath(dir, root)
    return replace(rel, '\\' => '/')
end

"""
    run_group(label)

Classify runs so plots can separate the main radius grid from tuning and
diagnostic branches.
"""
function run_group(label::S) where {S<:AbstractString}
    occursin("/diagnostics/", "/$label/") && return "diagnostics"
    startswith(label, "diagnostics/") && return "diagnostics"
    startswith(label, "fine_tuning/") && return "fine_tuning"
    occursin(r"^r\d+$", label) && return "main_radius_grid"
    return "other"
end

"""
    metric_series(rows, metric)

Extract `(epoch, value)` pairs for one numeric metric from parsed CSV rows.
"""
function metric_series(rows::R, metric::Symbol) where {R<:AbstractVector}
    pairs = Tuple{Int,Float64}[]
    for row in rows
        hasproperty(row, :epoch) || continue
        hasproperty(row, metric) || continue
        epoch_value = parse_metric(getproperty(row, :epoch))
        metric_value = parse_metric(getproperty(row, metric))
        (epoch_value === missing || metric_value === missing || metric_value === nothing) && continue
        push!(pairs, (Int(round(epoch_value)), Float64(metric_value)))
    end
    sort!(pairs; by = first)
    return pairs
end

"""
    slope_rows(root)

Compute per-epoch accuracy deltas for every metrics file under `root`.
"""
function slope_rows(root::R) where {R<:AbstractString}
    result = NamedTuple[]
    for (dir, _, files) in walkdir(root)
        "metrics.csv" in files || continue
        metrics_path = joinpath(dir, "metrics.csv")
        label = run_label(root, metrics_path)
        rows = csv_rows(metrics_path)
        for metric in (:test_accuracy, :train_accuracy)
            series = metric_series(rows, metric)
            length(series) < 2 && continue
            for idx in 2:length(series)
                prev_epoch, prev_value = series[idx - 1]
                epoch, value = series[idx]
                epoch_delta = epoch - prev_epoch
                epoch_delta <= 0 && continue
                push!(result, (;
                    run = label,
                    group = run_group(label),
                    metric = String(metric),
                    epoch,
                    previous_epoch = prev_epoch,
                    value,
                    previous_value = prev_value,
                    slope = (value - prev_value) / epoch_delta,
                ))
            end
        end
    end
    sort!(result; by = row -> (row.group, row.run, row.metric, row.epoch))
    return result
end

"""
    summary_rows(slopes)

Aggregate learning-slope diagnostics per run and metric.
"""
function summary_rows(slopes::R) where {R<:AbstractVector}
    keys = unique((row.run, row.group, row.metric) for row in slopes)
    result = NamedTuple[]
    for (run, group, metric) in keys
        selected = [row for row in slopes if row.run == run && row.group == group && row.metric == metric]
        isempty(selected) && continue
        values = [row.value for row in selected]
        slope_values = [row.slope for row in selected]
        tail = slope_values[max(1, length(slope_values) - 9):end]
        push!(result, (;
            run,
            group,
            metric,
            epochs = length(selected),
            best_value = maximum(values),
            final_value = values[end],
            mean_slope = mean(slope_values),
            median_slope = median(slope_values),
            mean_last10_slope = mean(tail),
            positive_slope_fraction = mean(slope_values .> 0),
        ))
    end
    sort!(result; by = row -> (row.group, row.run, row.metric))
    return result
end

"""
    write_csv(path, rows)

Write a vector of named tuples to CSV.
"""
function write_csv(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    mkpath(dirname(path))
    open(path, "w") do io
        isempty(rows) && return nothing
        names = propertynames(first(rows))
        println(io, join(String.(names), ','))
        for row in rows
            println(io, join((string(getproperty(row, name)) for name in names), ','))
        end
    end
    return path
end

"""
    moving_average(values, window)

Return a trailing moving average with the same length as `values`.
"""
function moving_average(values::V, window::I) where {V<:AbstractVector,I<:Integer}
    result = similar(values, Float64)
    for idx in eachindex(values)
        lo = max(firstindex(values), idx - Int(window) + 1)
        result[idx] = mean(values[lo:idx])
    end
    return result
end

"""
    plot_main_radius(root, outdir, slopes)

Plot accuracy and smoothed test-accuracy slopes for the main r1-r10 grid.
"""
function plot_main_radius(root::R, outdir::O, slopes::S) where {R<:AbstractString,O<:AbstractString,S<:AbstractVector}
    main_runs = filter(row -> row.group == "main_radius_grid" && row.metric == "test_accuracy", slopes)
    isempty(main_runs) && return nothing

    labels = sort(unique(row.run for row in main_runs); by = label -> parse(Int, replace(label, "r" => "")))
    colors = CairoMakie.cgrad(:tab10, length(labels), categorical = true)
    fig = CairoMakie.Figure(size = (1400, 900))
    ax_acc = CairoMakie.Axis(fig[1, 1], xlabel = "epoch", ylabel = "test accuracy", title = "Main radius grid: test accuracy")
    ax_slope = CairoMakie.Axis(fig[2, 1], xlabel = "epoch", ylabel = "extra accuracy / epoch", title = "Main radius grid: smoothed learning slope")
    CairoMakie.hlines!(ax_slope, [0.0], color = (:black, 0.35), linestyle = :dash)
    for (idx, label) in enumerate(labels)
        path = joinpath(root, label, "metrics.csv")
        series = metric_series(csv_rows(path), :test_accuracy)
        epochs = [first(pair) for pair in series]
        values = [last(pair) for pair in series]
        CairoMakie.lines!(ax_acc, epochs, values, label = label, color = colors[idx])
        selected = [row for row in main_runs if row.run == label]
        slope_epochs = [row.epoch for row in selected]
        smooth = moving_average([row.slope for row in selected], 7)
        CairoMakie.lines!(ax_slope, slope_epochs, smooth, label = label, color = colors[idx])
    end
    CairoMakie.axislegend(ax_acc, position = :rb, nbanks = 2)
    CairoMakie.save(joinpath(outdir, "main_radius_accuracy_and_smoothed_slope.png"), fig)
    return nothing
end

"""
    plot_summary(outdir, summaries)

Plot mean and recent learning slopes for all test-accuracy runs.
"""
function plot_summary(outdir::O, summaries::S) where {O<:AbstractString,S<:AbstractVector}
    rows = [row for row in summaries if row.metric == "test_accuracy"]
    isempty(rows) && return nothing
    sort!(rows; by = row -> (row.group, row.run))
    xs = collect(eachindex(rows))
    labels = [row.run for row in rows]

    fig = CairoMakie.Figure(size = (max(1400, 45 * length(rows)), 900))
    ax_mean = CairoMakie.Axis(fig[1, 1], xlabel = "run", ylabel = "accuracy / epoch", title = "Mean test-accuracy slope")
    ax_tail = CairoMakie.Axis(fig[2, 1], xlabel = "run", ylabel = "accuracy / epoch", title = "Mean slope over last 10 recorded epochs")
    CairoMakie.barplot!(ax_mean, xs, [row.mean_slope for row in rows], color = :steelblue)
    CairoMakie.barplot!(ax_tail, xs, [row.mean_last10_slope for row in rows], color = :darkorange)
    for ax in (ax_mean, ax_tail)
        CairoMakie.hlines!(ax, [0.0], color = (:black, 0.4), linestyle = :dash)
        ax.xticks = (xs, labels)
        ax.xticklabelrotation = pi / 3
    end
    CairoMakie.save(joinpath(outdir, "all_runs_test_slope_summary.png"), fig)
    return nothing
end

"""
    main()

Generate slope CSVs and plots for this experiment series.
"""
function main()
    root = @__DIR__
    outdir = joinpath(root, "analysis", "learning_slopes")
    mkpath(outdir)
    slopes = slope_rows(root)
    summaries = summary_rows(slopes)
    write_csv(joinpath(outdir, "learning_slopes_all.csv"), slopes)
    write_csv(joinpath(outdir, "learning_slope_summary.csv"), summaries)
    plot_main_radius(root, outdir, slopes)
    plot_summary(outdir, summaries)
    println("wrote learning slope analysis to ", outdir)
    return (; outdir, slopes = length(slopes), summaries = length(summaries))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
