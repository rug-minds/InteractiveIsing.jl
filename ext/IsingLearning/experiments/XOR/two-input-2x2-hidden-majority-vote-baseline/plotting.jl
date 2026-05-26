using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CairoMakie
using Dates

include("xor_majority_vote_baseline.jl")

"""Read the metrics CSV written by `xor_majority_vote_baseline.jl`."""
function read_metrics(path::P) where {P<:AbstractString}
    lines = readlines(path)
    header = split(first(lines), ',')
    rows = Dict{String,String}[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        cells = split(line, ',')
        row = Dict{String,String}()
        for idx in eachindex(header)
            row[header[idx]] = idx <= length(cells) ? cells[idx] : ""
        end
        push!(rows, row)
    end
    return rows
end

"""Parse one numeric metric column."""
function metric_column(rows::R, name::S) where {R<:AbstractVector,S<:AbstractString}
    return [parse(Float64, row[name]) for row in rows]
end

"""Parse a pipe-separated vector cell."""
function vector_cell(row::R, name::S) where {R,S<:AbstractString}
    return [parse(Float64, value) for value in split(row[name], '|')]
end

"""Plot learning curves for the majority-vote baseline."""
function plot_metrics(outdir::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    epochs = metric_column(rows, "epoch")
    mse = metric_column(rows, "mse")
    output_mse = metric_column(rows, "output_mse")
    spin_score_mse = metric_column(rows, "spin_score_mse")
    accuracy = metric_column(rows, "accuracy")
    min_margin = metric_column(rows, "min_margin")
    mean_margin = metric_column(rows, "mean_margin")

    fig = Figure(size = (1400, 900))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", title = "Majority-vote MSE")
    ax_acc = Axis(fig[1, 2], xlabel = "epoch", ylabel = "accuracy", title = "Majority-vote accuracy")
    ax_margin = Axis(fig[2, 1:2], xlabel = "epoch", ylabel = "target * score", title = "Margins")

    lines!(ax_mse, epochs, mse; color = :black, linewidth = 3, label = "vote MSE")
    lines!(ax_mse, epochs, spin_score_mse; color = :dodgerblue3, linewidth = 2, linestyle = :dash, label = "analog score MSE")
    lines!(ax_mse, epochs, output_mse; color = :darkorange, linewidth = 2, linestyle = :dot, label = "replica MSE")
    lines!(ax_acc, epochs, accuracy; color = :seagreen, linewidth = 3)
    lines!(ax_margin, epochs, min_margin; color = :firebrick, linewidth = 3, label = "min")
    lines!(ax_margin, epochs, mean_margin; color = :dodgerblue3, linewidth = 2, linestyle = :dash, label = "mean")
    axislegend(ax_mse, position = :rt)
    axislegend(ax_margin, position = :rb)

    path = joinpath(outdir, "metrics.png")
    save(path, fig)
    return path
end

"""Plot final per-case scores and output-replica states."""
function plot_final_outputs(outdir::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    last_row = last(rows)
    scores = vector_cell(last_row, "scores")
    spin_scores = vector_cell(last_row, "spin_scores")
    outputs = reshape(vector_cell(last_row, "outputs"), 4, 4)
    targets = [-1.0, 1.0, 1.0, -1.0]
    labels = ["00", "01", "10", "11"]

    fig = Figure(size = (1300, 800))
    ax_scores = Axis(fig[1, 1], xlabel = "XOR input", ylabel = "score", title = "Majority-vote score", xticks = (1:4, labels))
    ax_outputs = Axis(fig[1, 2], xlabel = "XOR input", ylabel = "output replica", title = "Final output replicas", xticks = (1:4, labels), yticks = (1:4, string.(1:4)))

    barplot!(ax_scores, 1:4, scores; color = ifelse.(scores .>= 0, :seagreen, :firebrick))
    scatter!(ax_scores, 1:4, spin_scores; color = :dodgerblue3, markersize = 12)
    scatter!(ax_scores, 1:4, targets; color = :black, markersize = 14)
    heatmap!(ax_outputs, 1:4, 1:4, outputs'; colormap = :balance, colorrange = (-1, 1))

    path = joinpath(outdir, "final_outputs.png")
    save(path, fig)
    return path
end

"""Run the experiment and generate plots from the data it wrote."""
function main()
    outdir = run_experiment()
    rows = read_metrics(joinpath(outdir, "metrics.csv"))
    metric_plot = plot_metrics(outdir, rows)
    output_plot = plot_final_outputs(outdir, rows)
    println("generated plots at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("  ", metric_plot)
    println("  ", output_plot)
    return (; outdir, metric_plot, output_plot)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
