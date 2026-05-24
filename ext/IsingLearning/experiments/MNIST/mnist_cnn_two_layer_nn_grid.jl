using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "mnist_local_paper_manager_grid.jl"))

"""Build the two-hidden-layer CNN-style MNIST configuration grid.

The sampled graph is hidden1-hidden2-output. Pixels enter hidden1 as fields,
hidden1 connects locally to hidden2, and hidden2 connects densely to replicated
class outputs. `local_radius` is the square NN/fanout radius used for both
input-to-hidden1 and hidden1-to-hidden2 maps.
"""
function cnn_two_layer_configs(base::C, root::P) where {C<:PaperMNISTManagerConfig,P<:AbstractString}
    hidden2_sides = parse_int_list(get(ENV, "ISING_MNIST_CNN_H2_SIDES", string(base.hidden2_side)), [base.hidden2_side])
    radii = parse_int_list(get(ENV, "ISING_MNIST_CNN_RADII", string(base.local_radius)), [base.local_radius])
    configs = PaperMNISTManagerConfig[]
    for hidden2_side in hidden2_sides, radius in radii
        name = "cnn_h1_$(base.hidden1_side)_h2_$(hidden2_side)_r$(radius)"
        outdir = joinpath(root, name)
        push!(configs, copy_config(base; name, hidden2_side, local_radius = radius, outdir))
    end
    return configs
end

"""Return one summary row from one completed CNN-style run."""
function cnn_summary_row(result::R) where {R}
    rows = result.rows
    best = first(rows)
    for row in rows
        row.test_accuracy > best.test_accuracy && (best = row)
    end
    config = result.config
    return (;
        config = config.name,
        hidden1_side = config.hidden1_side,
        hidden2_side = config.hidden2_side,
        radius = config.local_radius,
        train_per_class = config.train_per_class,
        test_per_class = config.test_per_class,
        epochs = config.epochs,
        batchsize = config.batchsize,
        workers = config.workers,
        best_epoch = best.epoch,
        best_test_accuracy = best.test_accuracy,
        best_test_loss = best.test_loss,
        final_test_accuracy = last(rows).test_accuracy,
        final_train_accuracy = last(rows).train_accuracy,
        best_path = best.best_path,
        run_dir = config.outdir,
    )
end

"""Plot learning curves and best accuracy for a CNN-style MNIST grid."""
function plot_cnn_grid(root::P, results::R, summaries::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    fig = Figure(size = (1450, 900))
    ax_test = Axis(fig[1, 1], xlabel = "epoch", ylabel = "test accuracy", title = "CNN-style MNIST test accuracy")
    ax_train = Axis(fig[2, 1], xlabel = "epoch", ylabel = "train accuracy", title = "Training accuracy")
    ax_loss = Axis(fig[1, 2], xlabel = "epoch", ylabel = "test loss", title = "Test loss")
    ax_best = Axis(fig[2, 2], xlabel = "configuration", ylabel = "best test accuracy", title = "Best by NN/fanout")

    palette = Makie.wong_colors()
    for (idx, result) in enumerate(results)
        color = palette[mod1(idx, length(palette))]
        label = result.config.name
        rows = result.rows
        train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
        lines!(ax_test, [row.epoch for row in rows], [row.test_accuracy for row in rows], color = color, label = label)
        lines!(ax_train, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], color = color)
        lines!(ax_loss, [row.epoch for row in rows], [row.test_loss for row in rows], color = color)
    end

    sorted = sort(summaries; by = row -> row.best_test_accuracy, rev = true)
    xvals = 1:length(sorted)
    barplot!(ax_best, xvals, [row.best_test_accuracy for row in sorted], color = :steelblue)
    ax_best.xticks = (xvals, [row.config for row in sorted])
    ax_best.xticklabelrotation = pi / 3
    axislegend(ax_test, position = :rb, nbanks = 2)

    path = joinpath(root, "cnn_two_layer_nn_summary.png")
    save(path, fig)
    return path
end

"""Write the aggregate summary for a CNN-style MNIST grid."""
function write_cnn_note!(path::P, base::C, summaries::S, plot_path::Q) where {P<:AbstractString,C<:PaperMNISTManagerConfig,S<:AbstractVector,Q<:AbstractString}
    open(path, "w") do io
        println(io, "# MNIST CNN-Style Two-Layer NN Grid")
        println(io)
        println(io, "Use of this folder: compare local square NN/fanout radii for the two-hidden-layer CNN-style MNIST architecture.")
        println(io)
        println(io, "- architecture family: `28^2 input fields -> $(base.hidden1_side)^2 hidden1 -> H2^2 hidden2 -> $(PMNIST_NCLASSES * base.output_replicas) outputs`")
        println(io, "- hidden2 sides: `$(get(ENV, "ISING_MNIST_CNN_H2_SIDES", string(base.hidden2_side)))`")
        println(io, "- local NN/fanout radii: `$(get(ENV, "ISING_MNIST_CNN_RADII", string(base.local_radius)))`")
        println(io, "- workers / batchsize / epochs: `$(base.workers)` / `$(base.batchsize)` / `$(base.epochs)`")
        println(io, "- train/test per class: `$(base.train_per_class)` / `$(base.test_per_class)`")
        println(io, "- gradient normalization: `$(base.gradient_normalization)`")
        println(io)
        println(io, "| Rank | Config | H2 Side | NN Radius | Best Test Accuracy | Best Epoch | Final Test Accuracy |")
        println(io, "|---:|---|---:|---:|---:|---:|---:|")
        for (rank, row) in enumerate(sort(summaries; by = row -> row.best_test_accuracy, rev = true))
            println(io, "| $rank | `$(row.config)` | $(row.hidden2_side) | $(row.radius) | $(round(row.best_test_accuracy; digits = 4)) | $(row.best_epoch) | $(round(row.final_test_accuracy; digits = 4)) |")
        end
        println(io)
        println(io, "Plot: `$(basename(plot_path))`")
        println(io, "Summary: `cnn_two_layer_nn_summary.csv`")
    end
    return path
end

"""Run the CNN-style two-hidden-layer MNIST NN/fanout grid."""
function main()
    base = PaperMNISTManagerConfig()
    root = get(
        ENV,
        "ISING_MNIST_CNN_OUTDIR",
        joinpath(@__DIR__, "runs", "current", "cnn_two_layer_nn_grid_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(root)
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers

    configs = cnn_two_layer_configs(base, root)
    println("Running ", length(configs), " CNN-style MNIST config(s)")
    println("root = ", root)
    results = NamedTuple[]
    summaries = NamedTuple[]
    for (idx, config) in enumerate(configs)
        println("[$idx/", length(configs), "] ", config.name)
        result = run_config!(config)
        push!(results, result)
        push!(summaries, cnn_summary_row(result))
    end

    summary_path = joinpath(root, "cnn_two_layer_nn_summary.csv")
    isfile(summary_path) && rm(summary_path)
    for row in summaries
        append_row!(summary_path, row)
    end
    plot_path = plot_cnn_grid(root, results, summaries)
    note_path = write_cnn_note!(joinpath(root, "README.md"), base, summaries, plot_path)
    println("saved summary ", summary_path)
    println("saved plot ", plot_path)
    println("saved note ", note_path)
    return (; root, results, summaries, plot_path, note_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
