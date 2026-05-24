using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates
using Statistics

"""Read a simple comma-separated file into rows of string dictionaries."""
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

"""Parse one numeric CSV cell, returning `missing` for empty or missing cells."""
function parse_cell(value::S) where {S<:AbstractString}
    stripped = strip(value)
    (isempty(stripped) || stripped == "missing") && return missing
    return try
        parse(Float64, stripped)
    catch
        missing
    end
end

"""Return numeric values for a CSV column, preserving missing entries."""
function numeric_column(rows::R, name::S) where {R<:AbstractVector,S<:AbstractString}
    return [parse_cell(get(row, name, "")) for row in rows]
end

"""Filter paired x/y arrays to finite numeric points for plotting."""
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

"""Parse final prediction-count strings written as either `;` or `-` separated."""
function parse_counts(value::S) where {S<:AbstractString}
    stripped = strip(value)
    isempty(stripped) && return Int[]
    delim = occursin(';', stripped) ? ';' : '-'
    return [parse(Int, part) for part in split(stripped, delim) if !isempty(strip(part))]
end

"""Plot one train/test metrics file with accuracy, loss, timing, and predictions."""
function plot_learning_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    has_epoch = "epoch" in header
    has_epoch || return nothing

    epochs = numeric_column(rows, "epoch")
    train_acc = numeric_column(rows, "train_accuracy")
    test_acc = numeric_column(rows, "test_accuracy")
    train_loss = numeric_column(rows, "train_loss")
    test_loss = numeric_column(rows, "test_loss")
    seconds = "epoch_time_s" in header ? numeric_column(rows, "epoch_time_s") :
        "seconds" in header ? numeric_column(rows, "seconds") : fill(missing, length(rows))
    pred_col = "test_pred_counts" in header ? "test_pred_counts" : "pred_counts"
    final_counts = parse_counts(get(last(rows), pred_col, ""))

    fig = Figure(size = (1300, 880))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Loss")
    ax_time = Axis(fig[1, 2], xlabel = "epoch", ylabel = "seconds", title = "Epoch time")
    ax_pred = Axis(fig[2, 2], xlabel = "digit", ylabel = "count", title = "Final prediction counts")

    x, y = finite_xy(epochs, train_acc)
    !isempty(x) && lines!(ax_acc, x, y, label = "train", color = :steelblue)
    x, y = finite_xy(epochs, test_acc)
    !isempty(x) && lines!(ax_acc, x, y, label = "test", color = :orange)

    x, y = finite_xy(epochs, train_loss)
    !isempty(x) && lines!(ax_loss, x, y, label = "train", color = :steelblue)
    x, y = finite_xy(epochs, test_loss)
    !isempty(x) && lines!(ax_loss, x, y, label = "test", color = :orange)

    x, y = finite_xy(epochs, seconds)
    !isempty(x) && lines!(ax_time, x, y, color = :black)
    !isempty(final_counts) && barplot!(ax_pred, 0:(length(final_counts) - 1), final_counts, color = :gray55)
    axislegend(ax_acc, position = :rb)
    axislegend(ax_loss, position = :rt)

    out = joinpath(dirname(path), "learning_curves.png")
    save(out, fig)
    return out
end

"""Plot one checkpoint/posthoc evaluation CSV as accuracy/loss and predictions."""
function plot_eval_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    ("accuracy" in header && "loss" in header) || return nothing

    labels = String[]
    for row in rows
        if haskey(row, "free_reads") && haskey(row, "free_sweeps")
            push!(labels, "r$(row["free_reads"])/s$(row["free_sweeps"])")
        elseif haskey(row, "reads") && haskey(row, "sweeps")
            push!(labels, "r$(row["reads"])/s$(row["sweeps"])")
        else
            push!(labels, string(length(labels) + 1))
        end
    end
    accuracy = Float64[Float64(parse_cell(row["accuracy"])) for row in rows]
    loss = Float64[Float64(parse_cell(row["loss"])) for row in rows]
    counts = parse_counts(get(rows[argmax(accuracy)], "pred_counts", ""))

    fig = Figure(size = (1200, 760))
    ax_acc = Axis(fig[1, 1], xlabel = "evaluation", ylabel = "accuracy", title = "Checkpoint accuracy", xticks = (1:length(labels), labels))
    ax_loss = Axis(fig[2, 1], xlabel = "evaluation", ylabel = "loss", title = "Checkpoint loss", xticks = (1:length(labels), labels))
    ax_pred = Axis(fig[:, 2], xlabel = "digit", ylabel = "count", title = "Best-eval prediction counts")
    barplot!(ax_acc, 1:length(accuracy), accuracy, color = :seagreen)
    barplot!(ax_loss, 1:length(loss), loss, color = :tomato)
    !isempty(counts) && barplot!(ax_pred, 0:(length(counts) - 1), counts, color = :gray55)

    stem = splitext(basename(path))[1]
    out = joinpath(dirname(path), stem * ".png")
    save(out, fig)
    return out
end

"""Plot aggregate one-row or multi-row configuration summary CSVs."""
function plot_summary_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    "best_test_accuracy" in header || return nothing
    labels = [get(row, "config", string(idx)) for (idx, row) in enumerate(rows)]
    best_acc = Float64[Float64(parse_cell(row["best_test_accuracy"])) for row in rows]
    final_acc = "final_test_accuracy" in header ? Float64[Float64(parse_cell(row["final_test_accuracy"])) for row in rows] : best_acc

    fig = Figure(size = (1100, 560))
    ax = Axis(fig[1, 1], xlabel = "configuration", ylabel = "accuracy", title = "Configuration summary", xticks = (1:length(labels), labels))
    barplot!(ax, (1:length(best_acc)) .- 0.18, best_acc; width = 0.34, color = :steelblue, label = "best test")
    barplot!(ax, (1:length(final_acc)) .+ 0.18, final_acc; width = 0.34, color = :orange, label = "final test")
    axislegend(ax, position = :rb)
    ax.xticklabelrotation = pi / 7
    out = joinpath(dirname(path), splitext(basename(path))[1] * "_plot.png")
    save(out, fig)
    return out
end

"""Collect checkpoint eval rows under `root` and draw one current-run overview."""
function plot_checkpoint_overview(root::P) where {P<:AbstractString}
    paths = sort([path for path in readdir(root; join = true) if isdir(path)])
    eval_rows = NamedTuple[]
    for dir in paths
        csv = joinpath(dir, "checkpoint_eval.csv")
        isfile(csv) || continue
        _, rows = read_csv_rows(csv)
        isempty(rows) && continue
        row = first(rows)
        push!(eval_rows, (;
            label = basename(dir),
            accuracy = Float64(parse_cell(row["accuracy"])),
            loss = Float64(parse_cell(row["loss"])),
        ))
    end
    isempty(eval_rows) && return nothing

    labels = [row.label for row in eval_rows]
    fig = Figure(size = (1500, 720))
    ax_acc = Axis(fig[1, 1], xlabel = "checkpoint eval", ylabel = "accuracy", title = "Checkpoint eval accuracy", xticks = (1:length(labels), labels))
    ax_loss = Axis(fig[2, 1], xlabel = "checkpoint eval", ylabel = "loss", title = "Checkpoint eval loss", xticks = (1:length(labels), labels))
    barplot!(ax_acc, 1:length(eval_rows), [row.accuracy for row in eval_rows], color = :seagreen)
    barplot!(ax_loss, 1:length(eval_rows), [row.loss for row in eval_rows], color = :tomato)
    ax_acc.xticklabelrotation = pi / 7
    ax_loss.xticklabelrotation = pi / 7
    out = joinpath(root, "checkpoint_eval_overview.png")
    save(out, fig)
    return out
end

"""Generate plots for all recognized current MNIST result CSV files."""
function main()
    root = joinpath(@__DIR__, "runs", "current")
    outputs = String[]
    for (dir, _, files) in walkdir(root)
        for file in files
            endswith(file, ".csv") || continue
            path = joinpath(dir, file)
            if file in ("metrics.csv", "mnist_local_paper_like_ep.csv")
                out = plot_learning_csv(path)
            elseif file in ("checkpoint_eval.csv", "posthoc_eval.csv")
                out = plot_eval_csv(path)
            elseif occursin("summary", file)
                out = plot_summary_csv(path)
            else
                out = nothing
            end
            isnothing(out) || push!(outputs, out)
        end
    end
    overview = plot_checkpoint_overview(root)
    isnothing(overview) || push!(outputs, overview)
    println("Generated $(length(outputs)) plots at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    foreach(path -> println("  ", path), outputs)
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
