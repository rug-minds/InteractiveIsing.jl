using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates

"""Read a simple comma-separated file into string-keyed rows."""
function read_csv_rows(path::P) where {P<:AbstractString}
    lines = readlines(path)
    isempty(lines) && return String[], Vector{Dict{String,String}}()
    header = split(first(lines), ',')
    rows = Dict{String,String}[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        cells = split(line, ',')
        row = Dict{String,String}()
        for (idx, name) in enumerate(header)
            row[name] = idx <= length(cells) ? cells[idx] : ""
        end
        push!(rows, row)
    end
    return header, rows
end

"""Parse one numeric cell, returning `missing` for empty, missing, or NaN cells."""
function parse_cell(value::S) where {S<:AbstractString}
    stripped = strip(value)
    (isempty(stripped) || stripped == "missing" || stripped == "NaN") && return missing
    return try
        parse(Float64, stripped)
    catch
        missing
    end
end

"""Return a column as numeric values with missing entries preserved."""
function numeric_column(rows::R, name::S) where {R<:AbstractVector,S<:AbstractString}
    return [parse_cell(get(row, name, "")) for row in rows]
end

"""Filter paired values to finite numeric vectors."""
function finite_xy(xs::X, ys::Y) where {X<:AbstractVector,Y<:AbstractVector}
    x_out = Float64[]
    y_out = Float64[]
    for (x, y) in zip(xs, ys)
        (ismissing(x) || ismissing(y)) && continue
        push!(x_out, Float64(x))
        push!(y_out, Float64(y))
    end
    return x_out, y_out
end

"""Plot one XOR metrics CSV with MSE, accuracy, margins, and optimizer traces."""
function plot_metrics_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    ("epoch" in header && "mse" in header) || return nothing

    epoch = numeric_column(rows, "epoch")
    mse = numeric_column(rows, "mse")
    output_mse = "output_mse" in header ? numeric_column(rows, "output_mse") : fill(missing, length(rows))
    accuracy = "accuracy" in header ? numeric_column(rows, "accuracy") : fill(missing, length(rows))
    min_margin = "min_margin" in header ? numeric_column(rows, "min_margin") : fill(missing, length(rows))
    mean_margin = "mean_margin" in header ? numeric_column(rows, "mean_margin") : fill(missing, length(rows))
    lr = "learning_rate" in header ? numeric_column(rows, "learning_rate") : fill(missing, length(rows))
    elapsed = "elapsed_seconds" in header ? numeric_column(rows, "elapsed_seconds") : fill(missing, length(rows))

    fig = Figure(size = (1300, 900))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", title = "MSE")
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_margin = Axis(fig[1, 2], xlabel = "epoch", ylabel = "margin", title = "Margins")
    ax_opt = Axis(fig[2, 2], xlabel = "epoch", ylabel = "value", title = "Learning rate / elapsed")

    x, y = finite_xy(epoch, mse)
    !isempty(x) && lines!(ax_mse, x, y, color = :steelblue, label = "mse")
    x, y = finite_xy(epoch, output_mse)
    !isempty(x) && lines!(ax_mse, x, y, color = :orange, label = "output mse")
    x, y = finite_xy(epoch, accuracy)
    !isempty(x) && lines!(ax_acc, x, y, color = :seagreen)
    x, y = finite_xy(epoch, min_margin)
    !isempty(x) && lines!(ax_margin, x, y, color = :tomato, label = "min")
    x, y = finite_xy(epoch, mean_margin)
    !isempty(x) && lines!(ax_margin, x, y, color = :purple, label = "mean")
    x, y = finite_xy(epoch, lr)
    !isempty(x) && lines!(ax_opt, x, y, color = :black, label = "lr")
    x, y = finite_xy(epoch, elapsed)
    !isempty(x) && lines!(ax_opt, x, y, color = :gray50, label = "elapsed")
    try
        axislegend(ax_mse, position = :rt)
    catch
    end
    try
        axislegend(ax_margin, position = :rb)
    catch
    end
    try
        axislegend(ax_opt, position = :rt)
    catch
    end

    out = joinpath(dirname(path), "metrics_plot.png")
    save(out, fig)
    return out
end

"""Plot one XOR summary CSV as best/final metrics across configurations."""
function plot_summary_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    ("config" in header && ("best_mse" in header || "best_accuracy" in header)) || return nothing

    labels = [get(row, "config", string(idx)) for (idx, row) in enumerate(rows)]
    best_mse = "best_mse" in header ? [parse_cell(row["best_mse"]) for row in rows] : fill(missing, length(rows))
    best_acc = "best_accuracy" in header ? [parse_cell(row["best_accuracy"]) for row in rows] : fill(missing, length(rows))
    best_margin = "best_min_margin" in header ? [parse_cell(row["best_min_margin"]) for row in rows] :
        "best_margin" in header ? [parse_cell(row["best_margin"]) for row in rows] : fill(missing, length(rows))

    fig = Figure(size = (1500, 820))
    ax_mse = Axis(fig[1, 1], xlabel = "configuration", ylabel = "MSE", title = "Best MSE", xticks = (1:length(labels), labels))
    ax_acc = Axis(fig[2, 1], xlabel = "configuration", ylabel = "accuracy", title = "Best accuracy", xticks = (1:length(labels), labels))
    ax_margin = Axis(fig[:, 2], xlabel = "configuration", ylabel = "margin", title = "Best minimum margin", xticks = (1:length(labels), labels))
    x = collect(1:length(labels))
    y = [ismissing(v) ? NaN : Float64(v) for v in best_mse]
    any(isfinite, y) && barplot!(ax_mse, x, y, color = :steelblue)
    y = [ismissing(v) ? NaN : Float64(v) for v in best_acc]
    any(isfinite, y) && barplot!(ax_acc, x, y, color = :seagreen)
    y = [ismissing(v) ? NaN : Float64(v) for v in best_margin]
    any(isfinite, y) && barplot!(ax_margin, x, y, color = :purple)
    ax_mse.xticklabelrotation = pi / 7
    ax_acc.xticklabelrotation = pi / 7
    ax_margin.xticklabelrotation = pi / 7

    out = joinpath(dirname(path), "summary_plot.png")
    save(out, fig)
    return out
end

"""Generate plots for every recognized XOR result CSV under `runs`."""
function main()
    root = joinpath(@__DIR__, "runs")
    outputs = String[]
    for (dir, _, files) in walkdir(root)
        for file in files
            endswith(file, ".csv") || continue
            path = joinpath(dir, file)
            out = file == "summary.csv" ? plot_summary_csv(path) :
                file == "metrics.csv" ? plot_metrics_csv(path) : nothing
            isnothing(out) || push!(outputs, out)
        end
    end
    println("Generated $(length(outputs)) XOR plots at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    foreach(path -> println("  ", path), outputs)
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
