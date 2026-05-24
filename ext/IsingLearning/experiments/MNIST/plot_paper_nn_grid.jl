using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie

"""Parse one scalar from the simple CSV emitted by `MNISTLocalPaperLikeEP.jl`."""
function parse_cell(value::S) where {S<:AbstractString}
    value == "missing" && return missing
    occursin("-", value) && return value
    parsed_int = tryparse(Int, value)
    isnothing(parsed_int) || return parsed_int
    parsed_float = tryparse(Float64, replace(value, "f0" => ""))
    isnothing(parsed_float) || return parsed_float
    return value
end

"""Load all epoch metrics for one radius subdirectory."""
function read_radius_metrics(root::P, radius::I) where {P<:AbstractString,I<:Integer}
    path = joinpath(root, "r$(radius)", "mnist_local_paper_like_ep.csv")
    isfile(path) || throw(ArgumentError("missing metrics file: $path"))
    lines = readlines(path)
    names = Symbol.(split(first(lines), ","))
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        values = parse_cell.(split(line, ","))
        push!(rows, merge((; radius = Int(radius), config = "r$(radius)"), NamedTuple{Tuple(names)}(Tuple(values))))
    end
    return rows
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

"""Compute one best-result row for a radius."""
function best_radius_row(rows::R) where {R<:AbstractVector}
    best = first(rows)
    for row in rows
        row.test_accuracy > best.test_accuracy && (best = row)
    end
    return (;
        config = best.config,
        radius = best.radius,
        best_epoch = best.epoch,
        best_test_accuracy = best.test_accuracy,
        best_test_loss = best.test_loss,
        final_train_accuracy = last(rows).train_accuracy,
        final_test_accuracy = last(rows).test_accuracy,
        best_path = best.best_path,
        run_dir = dirname(joinpath(best.best_path)),
    )
end

"""Plot accuracy and loss curves plus a best-accuracy comparison."""
function plot_grid(root::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    fig = Figure(size = (1450, 900))
    ax_test = Axis(fig[1, 1], xlabel = "epoch", ylabel = "test accuracy", title = "Paper-style MNIST local NN sweep")
    ax_train = Axis(fig[2, 1], xlabel = "epoch", ylabel = "train accuracy", title = "Training accuracy")
    ax_loss = Axis(fig[1, 2], xlabel = "epoch", ylabel = "test loss", title = "Test loss")
    ax_best = Axis(fig[2, 2], xlabel = "radius", ylabel = "best test accuracy", title = "Best by radius")

    palette = Makie.wong_colors()
    radii = sort(unique(row.radius for row in rows))
    for (idx, radius) in enumerate(radii)
        subset = [row for row in rows if row.radius == radius]
        color = palette[mod1(idx, length(palette))]
        lines!(ax_test, [row.epoch for row in subset], [row.test_accuracy for row in subset], color = color, label = "r$(radius)")
        train_subset = [row for row in subset if !ismissing(row.train_accuracy)]
        lines!(ax_train, [row.epoch for row in train_subset], [row.train_accuracy for row in train_subset], color = color)
        lines!(ax_loss, [row.epoch for row in subset], [row.test_loss for row in subset], color = color)
    end

    sorted_summary = sort(summary_rows; by = row -> row.radius)
    barplot!(ax_best, [row.radius for row in sorted_summary], [row.best_test_accuracy for row in sorted_summary], color = :steelblue)
    ax_best.xticks = ([row.radius for row in sorted_summary], string.([row.radius for row in sorted_summary]))
    axislegend(ax_test, position = :rb, nbanks = 2)

    path = joinpath(root, "paper_nn_grid_summary.png")
    save(path, fig)
    return path
end

"""Write a markdown summary next to the aggregate files."""
function write_note!(path::P, summary_rows::S, plot_path::Q) where {P<:AbstractString,S<:AbstractVector,Q<:AbstractString}
    open(path, "w") do io
        println(io, "# Paper-Style MNIST Local NN Grid")
        println(io)
        println(io, "This aggregates local paper-style EP runs using the working `784 -> 784 -> 121 -> 40` recipe.")
        println(io)
        println(io, "| Rank | Radius | Best Test Accuracy | Best Epoch | Best Test Loss |")
        println(io, "|---:|---:|---:|---:|---:|")
        for (rank, row) in enumerate(sort(summary_rows; by = row -> row.best_test_accuracy, rev = true))
            println(io, "| $rank | $(row.radius) | $(round(row.best_test_accuracy; digits = 4)) | $(row.best_epoch) | $(round(row.best_test_loss; digits = 4)) |")
        end
        println(io)
        println(io, "Plot: `$(basename(plot_path))`")
        println(io, "Metrics: `paper_nn_grid_metrics.csv`")
        println(io, "Summary: `paper_nn_grid_summary.csv`")
    end
    return path
end

"""Aggregate existing radius subruns and produce plots."""
function main()
    root = get(ENV, "ISING_MNIST_PAPER_NN_GRID_DIR", joinpath(@__DIR__, "runs", "20260523_local_paper_nn_grid"))
    radii = [parse(Int, part) for part in split(get(ENV, "ISING_MNIST_PAPER_NN_RADII", "1,2,3,5,7,10"), ",")]

    rows = NamedTuple[]
    summary_rows = NamedTuple[]
    for radius in radii
        radius_rows = read_radius_metrics(root, radius)
        append!(rows, radius_rows)
        push!(summary_rows, best_radius_row(radius_rows))
    end

    metrics_path = joinpath(root, "paper_nn_grid_metrics.csv")
    summary_path = joinpath(root, "paper_nn_grid_summary.csv")
    isfile(metrics_path) && rm(metrics_path)
    isfile(summary_path) && rm(summary_path)
    for row in rows
        append_row!(metrics_path, row)
    end
    for row in summary_rows
        append_row!(summary_path, row)
    end
    plot_path = plot_grid(root, rows, summary_rows)
    note_path = write_note!(joinpath(root, "README.md"), summary_rows, plot_path)
    println("saved metrics ", metrics_path)
    println("saved summary ", summary_path)
    println("saved plot ", plot_path)
    println("saved note ", note_path)
    return (; root, rows, summary_rows, plot_path, note_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
