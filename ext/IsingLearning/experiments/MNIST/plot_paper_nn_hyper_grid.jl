using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Statistics

"""Parse one cell from the aggregate CSVs written by `plot_paper_nn_grid.jl`."""
function parse_cell(value::S) where {S<:AbstractString}
    value == "missing" && return missing
    parsed_int = tryparse(Int, value)
    isnothing(parsed_int) || return parsed_int
    parsed_float = tryparse(Float64, replace(value, "f0" => ""))
    isnothing(parsed_float) || return parsed_float
    return value
end

"""Read one scenario summary and tag every row with the scenario name."""
function read_scenario_summary(name::S, root::P) where {S<:AbstractString,P<:AbstractString}
    path = joinpath(root, "paper_nn_grid_summary.csv")
    isfile(path) || throw(ArgumentError("missing scenario summary: $path"))
    lines = readlines(path)
    names = Symbol.(split(first(lines), ","))
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        values = parse_cell.(split(line, ","))
        push!(rows, merge((; scenario = name), NamedTuple{Tuple(names)}(Tuple(values))))
    end
    return rows
end

"""Default scenario roots for the baseline and the three hyperparameter probes."""
function scenario_roots()
    baseline_root = get(
        ENV,
        "ISING_MNIST_PAPER_BASELINE_DIR",
        joinpath(@__DIR__, "runs", "20260523_local_paper_nn_grid"),
    )
    hyper_root = get(
        ENV,
        "ISING_MNIST_PAPER_HYPER_GRID_DIR",
        joinpath(@__DIR__, "runs", "20260523_local_paper_nn_hyper_grid"),
    )
    return [
        ("beta5_lr003_T001", baseline_root),
        ("beta3_lr003_T001", joinpath(hyper_root, "beta3_lr003_T001")),
        ("beta5_lr0015_T001", joinpath(hyper_root, "beta5_lr0015_T001")),
        ("beta5_lr003_T002_R15", joinpath(hyper_root, "beta5_lr003_T002_R15")),
    ]
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

"""Plot best test accuracy by radius for each hyperparameter scenario."""
function plot_hyper_grid(root::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    fig = Figure(size = (1400, 820))
    ax_lines = Axis(fig[1, 1], xlabel = "local radius", ylabel = "best test accuracy", title = "MNIST local NN robustness")
    ax_rank = Axis(fig[1, 2], xlabel = "scenario", ylabel = "winner radius", title = "Winning radius per scenario")

    scenarios = unique(row.scenario for row in rows)
    palette = Makie.wong_colors()
    for (idx, scenario) in enumerate(scenarios)
        subset = sort([row for row in rows if row.scenario == scenario]; by = row -> row.radius)
        color = palette[mod1(idx, length(palette))]
        lines!(ax_lines, [row.radius for row in subset], [row.best_test_accuracy for row in subset], color = color, label = scenario)
        scatter!(ax_lines, [row.radius for row in subset], [row.best_test_accuracy for row in subset], color = color)
    end

    winners = [first(sort([row for row in rows if row.scenario == scenario]; by = row -> row.best_test_accuracy, rev = true)) for scenario in scenarios]
    xvals = 1:length(winners)
    barplot!(ax_rank, xvals, [row.radius for row in winners], color = :steelblue)
    ax_rank.xticks = (xvals, [row.scenario for row in winners])
    ax_rank.xticklabelrotation = pi / 5
    axislegend(ax_lines, position = :lb)

    path = joinpath(root, "paper_nn_hyper_grid_summary.png")
    save(path, fig)
    return path
end

"""Write a compact markdown conclusion for the hyperparameter robustness probe."""
function write_note!(path::P, rows::R, plot_path::Q) where {P<:AbstractString,R<:AbstractVector,Q<:AbstractString}
    scenarios = unique(row.scenario for row in rows)
    open(path, "w") do io
        println(io, "# Paper-Style MNIST Hyperparameter NN Grid")
        println(io)
        println(io, "Aggregates the baseline paper-style local MNIST radius sweep plus three hyperparameter probes.")
        println(io)
        println(io, "| Scenario | Best Radius | Best Accuracy | Runner Up | Mean Accuracy |")
        println(io, "|---|---:|---:|---|---:|")
        for scenario in scenarios
            subset = sort([row for row in rows if row.scenario == scenario]; by = row -> row.best_test_accuracy, rev = true)
            best = first(subset)
            runner_up = length(subset) > 1 ? subset[2] : best
            mean_accuracy = mean(row.best_test_accuracy for row in subset)
            println(
                io,
                "| `$(scenario)` | $(best.radius) | $(round(best.best_test_accuracy; digits = 4)) | r$(runner_up.radius) ($(round(runner_up.best_test_accuracy; digits = 4))) | $(round(mean_accuracy; digits = 4)) |",
            )
        end
        println(io)
        println(io, "Plot: `$(basename(plot_path))`")
        println(io, "Combined summary: `paper_nn_hyper_grid_summary.csv`")
    end
    return path
end

"""Aggregate all scenario summaries and write a combined plot."""
function main()
    output_root = get(
        ENV,
        "ISING_MNIST_PAPER_HYPER_GRID_DIR",
        joinpath(@__DIR__, "runs", "20260523_local_paper_nn_hyper_grid"),
    )
    mkpath(output_root)

    rows = NamedTuple[]
    for (name, root) in scenario_roots()
        append!(rows, read_scenario_summary(name, root))
    end

    summary_path = joinpath(output_root, "paper_nn_hyper_grid_summary.csv")
    isfile(summary_path) && rm(summary_path)
    for row in rows
        append_row!(summary_path, row)
    end
    plot_path = plot_hyper_grid(output_root, rows)
    note_path = write_note!(joinpath(output_root, "README.md"), rows, plot_path)
    println("saved combined summary ", summary_path)
    println("saved combined plot ", plot_path)
    println("saved note ", note_path)
    return (; output_root, rows, summary_path, plot_path, note_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
