using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates

const SINGLE_HIDDEN_ROOT = joinpath(@__DIR__, "single-hidden-local-28x28-to-11x11-readout")
const TWO_HIDDEN_ROOT = joinpath(@__DIR__, "two-hidden-local-28x28-to-14x14-readout")
const INLAID_ROOT = joinpath(@__DIR__, "inlaid-55x55-pixel-readout")
const DIAGNOSTICS_ROOT = joinpath(@__DIR__, "diagnostics", "runs")
const CURRENT_ROOTS = (
    joinpath(SINGLE_HIDDEN_ROOT, "experiments", "current"),
    joinpath(TWO_HIDDEN_ROOT, "experiments", "current"),
    joinpath(INLAID_ROOT, "experiments", "current"),
)
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

"""Return a readable plot label without legacy experiment shorthand."""
function display_label(label::S) where {S<:AbstractString}
    paper_match = match(
        r"^paper_local_h(\d+)_h(\d+)_r(\d+)_traininternal_(\d+)pc_mean_lr(\d+)_b(\d+)_e(\d+)(.*)$",
        label,
    )
    if !isnothing(paper_match)
        h1, h2, radius, train_pc, lr, batch, epochs, suffix = paper_match.captures
        eval_match = match(r"^_eval(\d+)pc_r(\d+)_s(\d+)$", suffix)
        if !isnothing(eval_match)
            eval_pc, reads, sweeps = eval_match.captures
            return "local $(h1)x$(h1) -> $(h2)x$(h2) -> 40, radius $(radius), eval $(eval_pc)% reads $(reads) sweeps $(sweeps)"
        end
        return "local $(h1)x$(h1) -> $(h2)x$(h2) -> 40, radius $(radius), $(epochs) epochs"
    end

    cnn_match = match(r"^cnn_two_layer_h(\d+)_h(\d+)_r(\d+)_mean_lr(\d+)_b(\d+)_e(\d+)(.*)$", label)
    if !isnothing(cnn_match)
        h1, h2, radius, lr, batch, epochs, suffix = cnn_match.captures
        eval_match = match(r"^_eval(\d+)pc_r(\d+)_s(\d+)$", suffix)
        if !isnothing(eval_match)
            eval_pc, reads, sweeps = eval_match.captures
            return "two hidden $(h1)x$(h1) -> $(h2)x$(h2) -> 40, radius $(radius), eval $(eval_pc)% reads $(reads) sweeps $(sweeps)"
        end
        return "two hidden $(h1)x$(h1) -> $(h2)x$(h2) -> 40, radius $(radius), $(epochs) epochs"
    end

    cnn_child_match = match(r"^cnn_h1_(\d+)_h2_(\d+)_r(\d+)$", label)
    if !isnothing(cnn_child_match)
        h1, h2, radius = cnn_child_match.captures
        return "two hidden $(h1)x$(h1) -> $(h2)x$(h2) -> 40, radius $(radius)"
    end

    inlaid_match = match(r"^inlaid_pixelreadout_beta(\d+)_comp(\d+)_lr(\d+)_e(\d+)_(\d+)pc$", label)
    if !isnothing(inlaid_match)
        beta, comp, lr, epochs, train_pc = inlaid_match.captures
        return "inlaid 55x55 pixels -> 40 readout, $(epochs) epochs"
    end

    return replace(label, "paper_" => "", "paper" => "", "_" => " ")
end

"""Return a filesystem-safe readable folder segment for mirrored plot output."""
function display_path_segment(label::S) where {S<:AbstractString}
    label == "_diagnostics_inlaid_input" && return "diagnostics_inlaid_input"
    readable = display_label(label)
    readable = replace(
        readable,
        " -> " => "_to_",
        "," => "",
        "%" => "pct",
        "/" => "_",
        r"\s+" => "_",
    )
    readable = replace(readable, r"[^A-Za-z0-9_.-]" => "")
    return lowercase(strip(readable, '_'))
end

"""Collect MNIST epoch CSV files below a run folder."""
function collect_epoch_paths(root::P) where {P<:AbstractString}
    paths = String[]
    isdir(root) || return paths
    for (dir, _, files) in walkdir(root)
        "mnist_local_paper_like_ep.csv" in files && push!(paths, joinpath(dir, "mnist_local_paper_like_ep.csv"))
        "metrics.csv" in files && push!(paths, joinpath(dir, "metrics.csv"))
    end
    sort!(unique!(paths); by = path -> (nn_value(basename(dirname(path))), basename(dirname(path))))
    return paths
end

"""Draw train/test accuracy and loss curves for one MNIST comparison group."""
function plot_learning_comparison(root::P, outpath::Q, title::S) where {P<:AbstractString,Q<:AbstractString,S<:AbstractString}
    paths = collect_epoch_paths(root)
    isempty(paths) && return nothing

    fig = Figure(size = (1800, 1100))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "$title: accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "$title: loss")
    ax_time = Axis(fig[1:2, 2], xlabel = "epoch", ylabel = "seconds", title = "$title: epoch time")
    handles = Any[]
    labels = String[]

    for (idx, path) in enumerate(paths)
        raw_label = basename(dirname(path))
        label = display_label(raw_label)
        nn = nn_value(raw_label)
        color = contrast_color(idx)
        style = line_style_for_nn(nn)
        _, rows = read_csv_rows(path)
        isempty(rows) && continue
        epoch = numeric_column(rows, "epoch")
        train_acc = numeric_column(rows, "train_accuracy")
        test_acc = numeric_column(rows, "test_accuracy")
        train_loss = numeric_column(rows, "train_loss")
        test_loss = numeric_column(rows, "test_loss")
        seconds = "epoch_time_s" in keys(first(rows)) ? numeric_column(rows, "epoch_time_s") :
            "seconds" in keys(first(rows)) ? numeric_column(rows, "seconds") : fill(missing, length(rows))

        first_handle = nothing
        x_plot, y_plot = finite_xy(epoch, train_acc)
        if !isempty(x_plot)
            first_handle = lines!(ax_acc, x_plot, y_plot; color, linestyle = style, linewidth = 2)
        end
        x_plot, y_plot = finite_xy(epoch, test_acc)
        !isempty(x_plot) && lines!(ax_acc, x_plot, y_plot; color, linestyle = style, linewidth = 3)
        x_plot, y_plot = finite_xy(epoch, train_loss)
        !isempty(x_plot) && lines!(ax_loss, x_plot, y_plot; color, linestyle = style, linewidth = 2)
        x_plot, y_plot = finite_xy(epoch, test_loss)
        !isempty(x_plot) && lines!(ax_loss, x_plot, y_plot; color, linestyle = style, linewidth = 3)
        x_plot, y_plot = finite_xy(epoch, seconds)
        !isempty(x_plot) && lines!(ax_time, x_plot, y_plot; color, linestyle = style, linewidth = 2)

        if isnothing(first_handle)
            x_plot, y_plot = finite_xy(epoch, test_loss)
            !isempty(x_plot) && (first_handle = lines!(ax_loss, x_plot, y_plot; color, linestyle = style, linewidth = 2))
        end
        isnothing(first_handle) || (push!(handles, first_handle); push!(labels, label))
    end
    Label(fig[3, 1:2], "Within each color/style, thicker accuracy/loss lines are test curves; thinner lines are train curves.", tellwidth = false)
    Legend(fig[:, 3], handles, labels, "configuration"; tellheight = false)
    save(outpath, fig)
    return outpath
end

"""Return the directory where plots for one experiment CSV should be written."""
function experiment_plot_dir(dir::P) where {P<:AbstractString}
    mkpath(dir)
    return dir
end

"""Parse final prediction-count strings written as either `;` or `-` separated."""
function parse_counts(value::S) where {S<:AbstractString}
    stripped = strip(value)
    isempty(stripped) && return Int[]
    delim = occursin(';', stripped) ? ';' : '-'
    return [parse(Int, part) for part in split(stripped, delim) if !isempty(strip(part))]
end

"""Draw the individual MNIST learning plot outside the run tree."""
function plot_individual_learning(path::P) where {P<:AbstractString}
    _, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    epoch = numeric_column(rows, "epoch")
    isempty(epoch) && return nothing

    train_acc = numeric_column(rows, "train_accuracy")
    test_acc = numeric_column(rows, "test_accuracy")
    train_loss = numeric_column(rows, "train_loss")
    test_loss = numeric_column(rows, "test_loss")
    seconds = "epoch_time_s" in keys(first(rows)) ? numeric_column(rows, "epoch_time_s") :
        "seconds" in keys(first(rows)) ? numeric_column(rows, "seconds") : fill(missing, length(rows))
    pred_col = "test_pred_counts" in keys(first(rows)) ? "test_pred_counts" : "pred_counts"
    final_counts = parse_counts(get(last(rows), pred_col, ""))

    fig = Figure(size = (1300, 880))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Loss")
    ax_time = Axis(fig[1, 2], xlabel = "epoch", ylabel = "seconds", title = "Epoch time")
    ax_pred = Axis(fig[2, 2], xlabel = "digit", ylabel = "count", title = "Final prediction counts")

    for (axis, column, color, label, style) in (
        (ax_acc, train_acc, :dodgerblue3, "train", :solid),
        (ax_acc, test_acc, :orange, "test", :dash),
        (ax_loss, train_loss, :dodgerblue3, "train", :solid),
        (ax_loss, test_loss, :orange, "test", :dash),
        (ax_time, seconds, :black, "seconds", :solid),
    )
        x_plot, y_plot = finite_xy(epoch, column)
        isempty(x_plot) || lines!(axis, x_plot, y_plot; color, label, linestyle = style, linewidth = 2)
    end
    !isempty(final_counts) && barplot!(ax_pred, 0:(length(final_counts) - 1), final_counts; color = :gray35)
    axislegend(ax_acc, position = :rb)
    axislegend(ax_loss, position = :rt)

    stem = splitext(basename(path))[1]
    out = joinpath(experiment_plot_dir(dirname(path)), stem * ".png")
    save(out, fig)
    return out
end

"""Draw the individual MNIST checkpoint/posthoc plot outside the run tree."""
function plot_individual_eval(path::P) where {P<:AbstractString}
    _, rows = read_csv_rows(path)
    isempty(rows) && return nothing
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
    accuracy = [v for v in numeric_column(rows, "accuracy") if !ismissing(v)]
    loss = [v for v in numeric_column(rows, "loss") if !ismissing(v)]
    isempty(accuracy) && isempty(loss) && return nothing
    counts = isempty(accuracy) ? Int[] : parse_counts(get(rows[argmax(Float64.(accuracy))], "pred_counts", ""))

    x = collect(1:length(labels))
    fig = Figure(size = (1200, 760))
    ax_acc = Axis(fig[1, 1], xlabel = "evaluation", ylabel = "accuracy", title = "Evaluation accuracy", xticks = (x, labels))
    ax_loss = Axis(fig[2, 1], xlabel = "evaluation", ylabel = "loss", title = "Evaluation loss", xticks = (x, labels))
    ax_pred = Axis(fig[:, 2], xlabel = "digit", ylabel = "count", title = "Best prediction counts")
    !isempty(accuracy) && barplot!(ax_acc, x, Float64.(accuracy); color = :seagreen)
    !isempty(loss) && barplot!(ax_loss, x, Float64.(loss); color = :firebrick)
    !isempty(counts) && barplot!(ax_pred, 0:(length(counts) - 1), counts; color = :gray35)

    stem = splitext(basename(path))[1]
    out = joinpath(experiment_plot_dir(dirname(path)), stem * ".png")
    save(out, fig)
    return out
end

"""Draw one per-run MNIST summary CSV as accuracy and loss bars outside the run tree."""
function plot_individual_summary(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    "config" in header || return nothing

    labels = [display_label(get(row, "config", string(idx))) for (idx, row) in enumerate(rows)]
    x = collect(1:length(labels))
    best_acc = "best_test_accuracy" in header ? numeric_column(rows, "best_test_accuracy") : fill(missing, length(rows))
    final_acc = "final_test_accuracy" in header ? numeric_column(rows, "final_test_accuracy") : fill(missing, length(rows))
    best_loss = "best_test_loss" in header ? numeric_column(rows, "best_test_loss") : fill(missing, length(rows))
    best_epoch = "best_epoch" in header ? numeric_column(rows, "best_epoch") : fill(missing, length(rows))

    fig = Figure(size = (1500, 900))
    ax_acc = Axis(fig[1, 1], xlabel = "configuration", ylabel = "accuracy", title = "Test accuracy", xticks = (x, labels))
    ax_loss = Axis(fig[2, 1], xlabel = "configuration", ylabel = "loss", title = "Best test loss", xticks = (x, labels))
    ax_epoch = Axis(fig[:, 2], xlabel = "configuration", ylabel = "epoch", title = "Best epoch", xticks = (x, labels))
    colors = [contrast_color(nn_value(label) == typemax(Int) ? idx : nn_value(label)) for (idx, label) in enumerate(labels)]
    barplot!(ax_acc, x .- 0.18, [ismissing(v) ? NaN : Float64(v) for v in best_acc]; width = 0.34, color = colors, label = "best")
    barplot!(ax_acc, x .+ 0.18, [ismissing(v) ? NaN : Float64(v) for v in final_acc]; width = 0.34, color = (:gray55, 0.7), label = "final")
    barplot!(ax_loss, x, [ismissing(v) ? NaN : Float64(v) for v in best_loss]; color = colors)
    barplot!(ax_epoch, x, [ismissing(v) ? NaN : Float64(v) for v in best_epoch]; color = colors)
    for axis in (ax_acc, ax_loss, ax_epoch)
        axis.xticklabelrotation = pi / 6
    end
    axislegend(ax_acc, position = :rb)

    stem = splitext(basename(path))[1]
    out = joinpath(experiment_plot_dir(dirname(path)), stem * ".png")
    save(out, fig)
    return out
end

"""Draw the inlaid-input scaling diagnostic with comparable quantities separated."""
function plot_inlaid_scaling(path::P) where {P<:AbstractString}
    _, rows = read_csv_rows(path)
    isempty(rows) && return nothing

    workers = numeric_column(rows, "workers")
    jobs_per_s = numeric_column(rows, "jobs_per_s")
    elapsed_s = numeric_column(rows, "elapsed_s")
    internal_s = numeric_column(rows, "worker_internal_s_per_job")
    x_workers, y_throughput = finite_xy(workers, jobs_per_s)
    isempty(x_workers) && return nothing

    base_throughput = first(y_throughput)
    speedup = [value / base_throughput for value in y_throughput]
    ideal = [worker / first(x_workers) for worker in x_workers]
    _, y_elapsed = finite_xy(workers, elapsed_s)
    _, y_internal_ms = finite_xy(workers, internal_s .* 1000)

    fig = Figure(size = (1500, 950))
    worker_ticks = (x_workers, string.(Int.(x_workers)))
    ax_throughput = Axis(fig[1, 1], xlabel = "workers", ylabel = "jobs / second", title = "Inlaid-input diagnostic throughput", xticks = worker_ticks)
    ax_speedup = Axis(fig[1, 2], xlabel = "workers", ylabel = "speedup vs 1 worker", title = "Scaling", xticks = worker_ticks)
    ax_elapsed = Axis(fig[2, 1], xlabel = "workers", ylabel = "seconds", title = "End-to-end batch time", xticks = worker_ticks)
    ax_internal = Axis(fig[2, 2], xlabel = "workers", ylabel = "ms / job", title = "Worker-local time per job", xticks = worker_ticks)

    barplot!(ax_throughput, x_workers, y_throughput; color = :dodgerblue3)
    lines!(ax_throughput, x_workers, base_throughput .* ideal; color = :gray35, linestyle = :dash, linewidth = 2.5, label = "ideal from 1 worker")
    scatter!(ax_throughput, x_workers, y_throughput; color = :black, markersize = 10)

    lines!(ax_speedup, x_workers, speedup; color = :seagreen, linewidth = 3, label = "actual")
    scatter!(ax_speedup, x_workers, speedup; color = :seagreen, markersize = 10)
    lines!(ax_speedup, x_workers, ideal; color = :gray35, linestyle = :dash, linewidth = 2.5, label = "ideal")

    !isempty(y_elapsed) && lines!(ax_elapsed, x_workers, y_elapsed; color = :firebrick, linewidth = 3)
    !isempty(y_internal_ms) && lines!(ax_internal, x_workers, y_internal_ms; color = :purple4, linewidth = 3)
    !isempty(y_elapsed) && scatter!(ax_elapsed, x_workers, y_elapsed; color = :firebrick, markersize = 10)
    !isempty(y_internal_ms) && scatter!(ax_internal, x_workers, y_internal_ms; color = :purple4, markersize = 10)

    axislegend(ax_throughput, position = :lt)
    axislegend(ax_speedup, position = :lt)

    out = joinpath(experiment_plot_dir(dirname(path)), "scaling.png")
    save(out, fig)
    return out
end

"""Draw a generic numeric MNIST CSV outside the run tree when no custom recipe applies."""
function plot_generic_csv(path::P) where {P<:AbstractString}
    header, rows = read_csv_rows(path)
    isempty(rows) && return nothing
    numeric_names = String[]
    for name in header
        values = numeric_column(rows, name)
        count(v -> !ismissing(v), values) > 0 && push!(numeric_names, name)
    end
    isempty(numeric_names) && return nothing

    x_name = "epoch" in numeric_names ? "epoch" :
        "sweeps" in numeric_names ? "sweeps" :
        "workers" in numeric_names ? "workers" : ""
    x_values = isempty(x_name) ? collect(1:length(rows)) : numeric_column(rows, x_name)
    plot_names = isempty(x_name) ? numeric_names : [name for name in numeric_names if name != x_name]
    isempty(plot_names) && return nothing

    fig = Figure(size = (1400, 850))
    axis = Axis(fig[1, 1], xlabel = isempty(x_name) ? "row" : x_name, ylabel = "value", title = basename(path))
    handles = Any[]
    labels = String[]
    for (idx, name) in enumerate(plot_names[1:min(end, 10)])
        nn = nn_value(name)
        color = contrast_color(nn == typemax(Int) ? idx : nn)
        style = line_style_for_nn(nn == typemax(Int) ? idx : nn)
        width = line_width_for_nn(nn == typemax(Int) ? idx : nn)
        x_plot, y_plot = finite_xy(x_values, numeric_column(rows, name))
        isempty(x_plot) && continue
        handle = lines!(axis, x_plot, y_plot; color, linestyle = style, linewidth = width)
        push!(handles, handle)
        push!(labels, name)
    end
    isempty(handles) && return nothing
    Legend(fig[1, 2], handles, labels, "columns"; tellheight = false)

    stem = splitext(basename(path))[1]
    out = joinpath(experiment_plot_dir(dirname(path)), stem * ".png")
    save(out, fig)
    return out
end

"""Generate individual MNIST plots directly beside the CSV files they summarize."""
function plot_individual_results()
    outputs = String[]
    for root in (CURRENT_ROOTS..., DIAGNOSTICS_ROOT)
        isdir(root) || continue
        for (dir, _, files) in walkdir(root)
            for file in files
                path = joinpath(dir, file)
                out = if file in ("metrics.csv", "mnist_local_paper_like_ep.csv")
                    plot_individual_learning(path)
                elseif file in ("checkpoint_eval.csv", "posthoc_eval.csv")
                    plot_individual_eval(path)
                elseif endswith(file, "_summary.csv") || file == "summary.csv"
                    plot_individual_summary(path)
                elseif file == "scaling.csv" && startswith(normpath(dir), normpath(DIAGNOSTICS_ROOT))
                    plot_inlaid_scaling(path)
                elseif endswith(file, ".csv")
                    plot_generic_csv(path)
                else
                    nothing
                end
                isnothing(out) || push!(outputs, out)
            end
        end
    end
    return outputs
end

"""Generate MNIST aggregate comparison plots inside each architecture folder."""
function main()
    outputs = String[]
    specs = (
        (
            root = joinpath(SINGLE_HIDDEN_ROOT, "experiments", "current"),
            outdir = joinpath(SINGLE_HIDDEN_ROOT, "aggregate_plots"),
            stem = "learning_curves",
            title = "single-hidden local 28x28 -> 11x11 -> 40",
        ),
        (
            root = joinpath(TWO_HIDDEN_ROOT, "experiments", "current"),
            outdir = joinpath(TWO_HIDDEN_ROOT, "aggregate_plots"),
            stem = "learning_curves",
            title = "two-hidden local 28x28 -> 14x14 -> 40",
        ),
        (
            root = joinpath(INLAID_ROOT, "experiments", "current"),
            outdir = joinpath(INLAID_ROOT, "aggregate_plots"),
            stem = "learning_curves",
            title = "inlaid 55x55 pixel readout",
        ),
    )
    for spec in specs
        mkpath(spec.outdir)
        curves = plot_learning_comparison(spec.root, joinpath(spec.outdir, spec.stem * ".png"), spec.title)
        isnothing(curves) || push!(outputs, curves)
        readme = joinpath(spec.outdir, "README.md")
        open(readme, "w") do io
            println(io, "# MNIST Aggregate Plots")
            println(io)
            println(io, "Use of this folder: aggregate comparison plots for this architecture family. Per-experiment plots live beside the CSV files under `experiments/current/<experiment-name>`.")
            println(io)
            println(io, "Checkpoint and post-hoc evaluation CSVs are plotted only in their experiment folders because reads/sweeps differ between runs.")
            println(io)
            println(io, "Line styles encode NN/radius groups: solid for 1-3, dashed for 4-7, dotted for 8 and above. Colors are high-contrast and reused only when many configurations are shown.")
        end
        push!(outputs, readme)
    end
    append!(outputs, plot_individual_results())

    println("Generated $(length(outputs)) MNIST plot files at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    foreach(path -> println("  ", path), outputs)
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
