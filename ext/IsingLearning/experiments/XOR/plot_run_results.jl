using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates

const CONTRAST_COLORS = [
    :dodgerblue3,
    :darkorange,
    :seagreen,
    :purple4,
    :firebrick,
    :goldenrod3,
    :deepskyblue4,
    :magenta4,
    :sienna,
    :olivedrab,
]

const CHECKERBOARD_ROOT = joinpath(@__DIR__, "checkerboard-local-cnn-two-hidden-layers")
const EDGE_ROOT = joinpath(@__DIR__, "edge-driven-single-layer-readout")
const MAJORITY_BASELINE_ROOT = joinpath(@__DIR__, "two-input-2x2-hidden-majority-vote-baseline")
const INPUT_AVERAGED_ROOT = joinpath(@__DIR__, "all-to-all-input-averaged-readout")
const DIAGNOSTICS_ROOT = joinpath(@__DIR__, "diagnostics")

const RESULT_FAMILIES = (
    (name = "checkerboard-local-cnn-two-hidden-layers", root = CHECKERBOARD_ROOT),
    (name = "edge-driven-single-layer-readout", root = EDGE_ROOT),
    (name = "two-input-2x2-hidden-majority-vote-baseline", root = MAJORITY_BASELINE_ROOT),
    (name = "all-to-all-input-averaged-readout", root = INPUT_AVERAGED_ROOT),
    (name = "diagnostics", root = DIAGNOSTICS_ROOT),
)

"""Return the data tree to scan for a result family."""
function family_data_root(family)
    current_root = joinpath(family.root, "experiments", "current")
    isdir(current_root) && return current_root
    return joinpath(family.root, "runs")
end

"""Read a simple comma-separated metrics file into string-keyed rows."""
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

"""Parse a numeric CSV cell, using `missing` for empty, missing, or non-numeric fields."""
function parse_cell(value::S) where {S<:AbstractString}
    stripped = strip(value)
    (isempty(stripped) || stripped == "missing" || stripped == "NaN") && return missing
    return try
        parse(Float64, stripped)
    catch
        missing
    end
end

"""Extract one numeric CSV column while preserving missing values."""
function numeric_column(rows::R, name::S) where {R<:AbstractVector,S<:AbstractString}
    return [parse_cell(get(row, name, "")) for row in rows]
end

"""Return finite x/y plot vectors from possibly-missing CSV columns."""
function finite_xy(xs::X, ys::Y) where {X<:AbstractVector,Y<:AbstractVector}
    x_out = Float64[]
    y_out = Float64[]
    for (x, y) in zip(xs, ys)
        (ismissing(x) || ismissing(y)) && continue
        xf = Float64(x)
        yf = Float64(y)
        (isfinite(xf) && isfinite(yf)) || continue
        push!(x_out, xf)
        push!(y_out, yf)
    end
    return x_out, y_out
end

"""Parse the local radius or NN value from a run label."""
function nn_value(label::S) where {S<:AbstractString}
    for pattern in (r"(?:^|_)r(\d+)(?:_|$)", r"(?:^|_)nn(\d+)(?:_|$)")
        match_result = match(pattern, label)
        isnothing(match_result) || return parse(Int, match_result.captures[1])
    end
    return typemax(Int)
end

"""Choose a line pattern by NN group: low, mid, and high are visually distinct."""
function line_style_for_nn(nn::Integer)
    nn <= 3 && return :solid
    nn <= 7 && return :dash
    return :dot
end

"""Choose a line width by NN group so grouped results differ beyond color."""
function line_width_for_nn(nn::Integer)
    nn <= 3 && return 2.0
    nn <= 7 && return 2.8
    return 3.6
end

"""Choose a stable high-contrast color from an integer key."""
function contrast_color(key::Integer)
    return CONTRAST_COLORS[mod1(key, length(CONTRAST_COLORS))]
end

"""Return a short legend label for a metrics path."""
function config_label(path::P) where {P<:AbstractString}
    return basename(dirname(path))
end

"""Return a compact plot label that keeps only architecture-identifying parts."""
function plot_config_label(label::S) where {S<:AbstractString}
    nn_match = match(r"^nn(\d+)$", label)
    !isnothing(nn_match) && return "NN $(nn_match.captures[1])"

    hidden_radius_match = match(r"^h(\d+)_r(\d+)(?:_(open|periodic))?$", label)
    if !isnothing(hidden_radius_match)
        hidden, radius, boundary = hidden_radius_match.captures
        suffix = boundary == "periodic" ? ", periodic" : ""
        return "$(hidden)x$(hidden), r$(radius)" * suffix
    end

    return replace(label, r"_open\b" => "", "_" => " ")
end

"""Return true for a leaf configuration folder that should not become a plot subfolder."""
function is_leaf_config_label(label::S) where {S<:AbstractString}
    occursin(r"^nn\d+$", label) && return true
    occursin(r"^h\d+_r\d+(?:_(?:open|periodic))?$", label) && return true
    return false
end

"""Return a filename-safe configuration label without default boundary words."""
function plot_file_label(label::S) where {S<:AbstractString}
    return replace(label, r"_open\b" => "")
end

"""Save one plot as the requested PNG file."""
function save_plot(path::P, fig) where {P<:AbstractString}
    save(path, fig)
    return path
end

"""Return a readable folder segment for an XOR output encoding."""
function readable_output_segment(segment::S) where {S<:AbstractString}
    segment == "majority" && return "output_majority_vote"
    segment == "pattern" && return "output_spatial_pattern_target"
    segment == "two_class" && return "output_replicated_two_class"
    return segment
end

"""Return a readable mirrored path for one run-tree-relative path."""
function readable_relpath(path::P) where {P<:AbstractString}
    parts = splitpath(path)
    isempty(parts) && return path
    parts[1] = readable_output_segment(parts[1])
    return joinpath(parts...)
end

"""Return the run directory where a generated plot should be written."""
function mirrored_plot_dir(dir::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    mkpath(dir)
    return dir
end

"""Return the flattened in-run output path for a leaf configuration metrics plot."""
function mirrored_metrics_path(path::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    parts = splitpath(relpath(dirname(path), runs_root))
    label = isempty(parts) ? splitext(basename(path))[1] : last(parts)
    if is_leaf_config_label(label)
        outdir = dirname(dirname(path))
        mkpath(outdir)
        return joinpath(outdir, plot_file_label(label) * "_metrics.png")
    end
    return joinpath(mirrored_plot_dir(dirname(path), runs_root, individual_root), "metrics.png")
end

"""Collect leaf `metrics.csv` files below a run root for aggregate NN/radius comparisons."""
function collect_metric_paths(root::P) where {P<:AbstractString}
    paths = String[]
    isdir(root) || return paths
    for (dir, _, files) in walkdir(root)
        "metrics.csv" in files || continue
        nn_value(basename(dir)) == typemax(Int) && continue
        push!(paths, joinpath(dir, "metrics.csv"))
    end
    sort!(paths; by = path -> (nn_value(config_label(path)), config_label(path)))
    return paths
end

"""Draw the individual metrics plot for one XOR `metrics.csv` outside the run tree."""
function plot_individual_metrics(path::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    _, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    epoch = numeric_column(rows, "epoch")
    isempty(epoch) && return nothing

    fig = Figure(size = (1300, 900))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", title = "MSE")
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_margin = Axis(fig[1, 2], xlabel = "epoch", ylabel = "margin", title = "Margins")
    ax_opt = Axis(fig[2, 2], xlabel = "epoch", ylabel = "value", title = "Learning rate / elapsed")

    for (axis, column, color, label, style) in (
        (ax_mse, "mse", :black, "mse", :solid),
        (ax_mse, "output_mse", :orange, "output mse", :dash),
        (ax_acc, "accuracy", :seagreen, "accuracy", :solid),
        (ax_margin, "min_margin", :firebrick, "min margin", :solid),
        (ax_margin, "mean_margin", :purple4, "mean margin", :dash),
        (ax_opt, "learning_rate", :black, "learning rate", :solid),
        (ax_opt, "elapsed_seconds", :gray35, "elapsed", :dash),
    )
        y = numeric_column(rows, column)
        x_plot, y_plot = finite_xy(epoch, y)
        isempty(x_plot) || lines!(axis, x_plot, y_plot; color, label, linestyle = style, linewidth = 2.5)
    end
    for axis in (ax_mse, ax_margin, ax_opt)
        try
            axislegend(axis, position = :rt)
        catch
        end
    end

    out = mirrored_metrics_path(path, runs_root, individual_root)
    save_plot(out, fig)
    return out
end

"""Draw the individual summary plot for one XOR `summary.csv` outside the run tree."""
function plot_individual_summary(path::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    "config" in header || return nothing

    labels = [get(row, "config", string(idx)) for (idx, row) in enumerate(rows)]
    x = collect(1:length(labels))
    best_mse = "best_mse" in header ? [parse_cell(row["best_mse"]) for row in rows] : fill(missing, length(rows))
    best_acc = "best_accuracy" in header ? [parse_cell(row["best_accuracy"]) for row in rows] : fill(missing, length(rows))
    best_margin = "best_min_margin" in header ? [parse_cell(row["best_min_margin"]) for row in rows] : fill(missing, length(rows))

    fig = Figure(size = (1500, 820))
    ax_mse = Axis(fig[1, 1], xlabel = "configuration", ylabel = "MSE", title = "Best MSE", xticks = (x, labels))
    ax_acc = Axis(fig[2, 1], xlabel = "configuration", ylabel = "accuracy", title = "Best accuracy", xticks = (x, labels))
    ax_margin = Axis(fig[:, 2], xlabel = "configuration", ylabel = "margin", title = "Best minimum margin", xticks = (x, labels))
    barplot!(ax_mse, x, [ismissing(v) ? NaN : Float64(v) for v in best_mse]; color = :dodgerblue3)
    barplot!(ax_acc, x, [ismissing(v) ? NaN : Float64(v) for v in best_acc]; color = :seagreen)
    barplot!(ax_margin, x, [ismissing(v) ? NaN : Float64(v) for v in best_margin]; color = :purple4)
    for axis in (ax_mse, ax_acc, ax_margin)
        axis.xticklabelrotation = pi / 7
    end

    out = joinpath(mirrored_plot_dir(dirname(path), runs_root, individual_root), "summary.png")
    save_plot(out, fig)
    return out
end

"""Draw a validation CSV as margin, accuracy, and loss bars outside the run tree."""
function plot_individual_validation(path::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    "config" in header || return nothing

    labels = [get(row, "config", string(idx)) for (idx, row) in enumerate(rows)]
    x = collect(1:length(labels))
    accuracy = "accuracy" in header ? [parse_cell(row["accuracy"]) for row in rows] : fill(missing, length(rows))
    min_margin = "min_margin" in header ? [parse_cell(row["min_margin"]) for row in rows] : fill(missing, length(rows))
    mean_margin = "mean_margin" in header ? [parse_cell(row["mean_margin"]) for row in rows] : fill(missing, length(rows))
    mse = "mse" in header ? [parse_cell(row["mse"]) for row in rows] : fill(missing, length(rows))

    fig = Figure(size = (1500, 900))
    ax_margin = Axis(fig[1, 1], xlabel = "configuration", ylabel = "margin", title = "Validation margins", xticks = (x, labels))
    ax_acc = Axis(fig[2, 1], xlabel = "configuration", ylabel = "accuracy", title = "Validation accuracy", xticks = (x, labels))
    ax_mse = Axis(fig[:, 2], xlabel = "configuration", ylabel = "MSE", title = "Validation MSE", xticks = (x, labels))
    colors = [contrast_color(nn_value(label) == typemax(Int) ? idx : nn_value(label)) for (idx, label) in enumerate(labels)]
    barplot!(ax_margin, x .- 0.18, [ismissing(v) ? NaN : Float64(v) for v in min_margin]; width = 0.34, color = colors, label = "min")
    barplot!(ax_margin, x .+ 0.18, [ismissing(v) ? NaN : Float64(v) for v in mean_margin]; width = 0.34, color = (:gray55, 0.7), label = "mean")
    barplot!(ax_acc, x, [ismissing(v) ? NaN : Float64(v) for v in accuracy]; color = colors)
    barplot!(ax_mse, x, [ismissing(v) ? NaN : Float64(v) for v in mse]; color = colors)
    for axis in (ax_margin, ax_acc, ax_mse)
        axis.xticklabelrotation = pi / 6
    end
    axislegend(ax_margin, position = :rb)

    stem = splitext(basename(path))[1]
    out = joinpath(mirrored_plot_dir(dirname(path), runs_root, individual_root), stem * ".png")
    save_plot(out, fig)
    return out
end

"""Draw a generic numeric CSV plot for files without a custom plotting recipe."""
function plot_generic_csv(path::P, runs_root::Q, individual_root::R) where {P<:AbstractString,Q<:AbstractString,R<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    numeric_names = String[]
    for name in header
        values = numeric_column(rows, name)
        count(v -> !ismissing(v), values) > 0 && push!(numeric_names, name)
    end
    isempty(numeric_names) && return nothing

    x_values = "epoch" in numeric_names ? numeric_column(rows, "epoch") : collect(1:length(rows))
    plot_names = [name for name in numeric_names if name != "epoch"]
    isempty(plot_names) && return nothing
    fig = Figure(size = (1300, 780))
    axis = Axis(fig[1, 1], xlabel = "row", ylabel = "value", title = basename(path))
    for (idx, name) in enumerate(plot_names[1:min(end, 8)])
        x_plot, y_plot = finite_xy(x_values, numeric_column(rows, name))
        isempty(x_plot) || lines!(axis, x_plot, y_plot; color = contrast_color(idx), label = name, linewidth = 2.5)
    end
    axislegend(axis, position = :rt)
    stem = splitext(basename(path))[1]
    out = joinpath(mirrored_plot_dir(dirname(path), runs_root, individual_root), stem * ".png")
    save_plot(out, fig)
    return out
end

"""Generate individual plots directly in one family's run folders."""
function plot_family_individuals(family)
    runs_root = family_data_root(family)
    outputs = String[]
    isdir(runs_root) || return outputs
    for (dir, _, files) in walkdir(runs_root)
        for file in files
            endswith(file, ".csv") || continue
            path = joinpath(dir, file)
            out = if file == "metrics.csv"
                plot_individual_metrics(path, runs_root, runs_root)
            elseif file == "summary.csv"
                plot_individual_summary(path, runs_root, runs_root)
            elseif startswith(file, "validation_")
                plot_individual_validation(path, runs_root, runs_root)
            else
                plot_generic_csv(path, runs_root, runs_root)
            end
            isnothing(out) || push!(outputs, out)
        end
    end
    return outputs
end

"""Draw one aggregate comparison figure for MSE, accuracy, and margin."""
function plot_metric_comparison(paths::P, outpath::Q, title::S) where {P<:AbstractVector,Q<:AbstractString,S<:AbstractString}
    isempty(paths) && return nothing
    fig = Figure(size = (1800, 1100))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", title = "$title: MSE")
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "$title: accuracy")
    ax_margin = Axis(fig[1:2, 2], xlabel = "epoch", ylabel = "margin", title = "$title: minimum margin")
    handles = Any[]
    labels = String[]
    for (idx, path) in enumerate(paths)
        raw_label = config_label(path)
        nn = nn_value(raw_label)
        nn == typemax(Int) && continue
        label = plot_config_label(raw_label)
        color = contrast_color(nn)
        style = line_style_for_nn(nn)
        width = line_width_for_nn(nn)
        _, rows = read_csv_rows(path)
        isempty(rows) && continue
        epoch = numeric_column(rows, "epoch")
        first_handle = nothing
        for (axis, column) in ((ax_mse, "mse"), (ax_acc, "accuracy"), (ax_margin, "min_margin"))
            x_plot, y_plot = finite_xy(epoch, numeric_column(rows, column))
            isempty(x_plot) && continue
            handle = lines!(axis, x_plot, y_plot; color, linestyle = style, linewidth = width)
            isnothing(first_handle) && (first_handle = handle)
        end
        isnothing(first_handle) || (push!(handles, first_handle); push!(labels, label))
    end
    Legend(fig[:, 3], handles, labels, "configuration"; tellheight = false)
    save_plot(outpath, fig)
    return outpath
end

"""Draw one aggregate best-metric bar summary for a set of metrics files."""
function plot_best_summary(paths::P, outpath::Q, title::S) where {P<:AbstractVector,Q<:AbstractString,S<:AbstractString}
    isempty(paths) && return nothing
    labels = String[]
    best_mse = Float64[]
    best_accuracy = Float64[]
    best_margin = Float64[]
    colors = Any[]
    for path in paths
        raw_label = config_label(path)
        nn = nn_value(raw_label)
        nn == typemax(Int) && continue
        _, rows = read_csv_rows(path)
        isempty(rows) && continue
        mse = [v for v in numeric_column(rows, "mse") if !ismissing(v)]
        accuracy = [v for v in numeric_column(rows, "accuracy") if !ismissing(v)]
        margin = [v for v in numeric_column(rows, "min_margin") if !ismissing(v)]
        isempty(mse) && isempty(accuracy) && isempty(margin) && continue
        push!(labels, plot_config_label(raw_label))
        push!(best_mse, isempty(mse) ? NaN : minimum(Float64.(mse)))
        push!(best_accuracy, isempty(accuracy) ? NaN : maximum(Float64.(accuracy)))
        push!(best_margin, isempty(margin) ? NaN : maximum(Float64.(margin)))
        push!(colors, contrast_color(nn))
    end
    isempty(labels) && return nothing

    x = collect(1:length(labels))
    fig = Figure(size = (1800, 1050))
    ax_mse = Axis(fig[1, 1], xlabel = "configuration", ylabel = "best MSE", title = "$title: best MSE", xticks = (x, labels))
    ax_acc = Axis(fig[2, 1], xlabel = "configuration", ylabel = "best accuracy", title = "$title: best accuracy", xticks = (x, labels))
    ax_margin = Axis(fig[:, 2], xlabel = "configuration", ylabel = "best minimum margin", title = "$title: best margin", xticks = (x, labels))
    barplot!(ax_mse, x, best_mse; color = colors)
    barplot!(ax_acc, x, best_accuracy; color = colors)
    barplot!(ax_margin, x, best_margin; color = colors)
    for axis in (ax_mse, ax_acc, ax_margin)
        axis.xticklabelrotation = pi / 4
    end
    save_plot(outpath, fig)
    return outpath
end

"""Generate aggregate comparison plots in each result family folder."""
function plot_family_aggregates()
    outputs = String[]

    checker_out = joinpath(CHECKERBOARD_ROOT, "aggregate_plots")
    mkpath(checker_out)
    checker_specs = (
        (
            root = joinpath(CHECKERBOARD_ROOT, "experiments", "current", "local_checkerboard_twoclass_h8_r7to10_resume_bestmargin_e200_lr001_decay998"),
            stem = "output_replicated_two_class_h8_r7to10",
            title = "checkerboard replicated two-class output 8x8 radius 7-10",
        ),
        (
            root = joinpath(CHECKERBOARD_ROOT, "experiments", "current", "local_checkerboard_twoclass_h8_r9r10_resume2_bestmargin_e240_lr0005_decay999"),
            stem = "output_replicated_two_class_h8_r9r10",
            title = "checkerboard replicated two-class output 8x8 radius 9-10",
        ),
    )
    for spec in checker_specs
        checker_paths = collect_metric_paths(spec.root)
        for out in (
            plot_metric_comparison(checker_paths, joinpath(checker_out, spec.stem * "_curves.png"), spec.title),
            plot_best_summary(checker_paths, joinpath(checker_out, spec.stem * "_best_metrics.png"), spec.title),
        )
            isnothing(out) || push!(outputs, out)
        end
    end
    open(joinpath(checker_out, "README.md"), "w") do io
        println(io, "# Checkerboard Aggregate Plots")
        println(io)
        println(io, "Use of this folder: aggregate checkerboard comparison PNGs only. Solid/thinner lines are NN/radius 1-3, dashed/medium lines are 4-7, dotted/thicker lines are 8+.")
    end

    edge_run = joinpath(EDGE_ROOT, "experiments", "current", "edge_twoclass_zero_side16_nn1to10_e160")
    edge_out = joinpath(EDGE_ROOT, "aggregate_plots")
    mkpath(edge_out)
    edge_paths = collect_metric_paths(edge_run)
    for out in (
        plot_metric_comparison(edge_paths, joinpath(edge_out, "output_replicated_two_class_side16_nn1to10_curves.png"), "edge replicated two-class output side16 NN 1-10"),
        plot_best_summary(edge_paths, joinpath(edge_out, "output_replicated_two_class_side16_nn1to10_best_metrics.png"), "edge replicated two-class output side16 NN 1-10"),
    )
        isnothing(out) || push!(outputs, out)
    end
    open(joinpath(edge_out, "README.md"), "w") do io
        println(io, "# Edge Aggregate Plots")
        println(io)
        println(io, "Use of this folder: aggregate edge-architecture comparison PNGs only. Solid/thinner lines are NN 1-3, dashed/medium lines are 4-7, dotted/thicker lines are 8+.")
    end

    return outputs
end

"""Generate XOR plots in separated family folders; run data folders are untouched."""
function main()
    outputs = String[]
    append!(outputs, plot_family_aggregates())
    for family in RESULT_FAMILIES
        append!(outputs, plot_family_individuals(family))
    end
    println("Generated $(length(outputs)) XOR plot files at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    foreach(path -> println("  ", path), outputs)
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
