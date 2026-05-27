using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "mnist_local_paper_manager_grid.jl"))

"""Build the selected two-hidden `(r1, r2)` configuration grid."""
function two_hidden_configs(base::C, root::P) where {C<:LocalMNISTManagerConfig,P<:AbstractString}
    default_pairs = [(1, 1), (3, 2), (5, 3), (7, 4), (10, 5)]
    pairs = parse_radius_pairs(get(ENV, "ISING_MNIST_2H_PAIRS", "1:1,3:2,5:3,7:4,10:5"), default_pairs)
    configs = LocalMNISTManagerConfig[]
    for (r1, r2) in pairs
        name = "r1_$(r1)_r2_$(r2)"
        outdir = joinpath(root, name)
        push!(configs, copy_config(base; name, input_hidden_radius = r1, hidden_hidden_radius = r2, outdir))
    end
    return configs
end

"""Return one compact summary row from one completed two-hidden run."""
function two_hidden_summary_row(result::R) where {R}
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
        r1 = config.input_hidden_radius,
        r2 = config.hidden_hidden_radius,
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

"""Plot learning curves and best accuracy for the selected two-hidden grid."""
function plot_two_hidden_grid(root::P, results::R, summaries::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    CM = ensure_cairomakie()
    fig = CM.Figure(size = (1450, 900))
    ax_test = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "test accuracy", title = "Two-hidden MNIST test accuracy")
    ax_train = CM.Axis(fig[2, 1], xlabel = "epoch", ylabel = "train accuracy", title = "Training accuracy")
    ax_loss = CM.Axis(fig[1, 2], xlabel = "epoch", ylabel = "test loss", title = "Test loss")
    ax_best = CM.Axis(fig[2, 2], xlabel = "configuration", ylabel = "best test accuracy", title = "Best by r1/r2")

    palette = CM.Makie.wong_colors()
    for (idx, result) in enumerate(results)
        color = palette[mod1(idx, length(palette))]
        label = result.config.name
        rows = result.rows
        train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
        CM.lines!(ax_test, [row.epoch for row in rows], [row.test_accuracy for row in rows], color = color, label = label)
        CM.lines!(ax_train, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], color = color)
        CM.lines!(ax_loss, [row.epoch for row in rows], [row.test_loss for row in rows], color = color)
    end

    sorted = sort(summaries; by = row -> row.best_test_accuracy, rev = true)
    xvals = 1:length(sorted)
    CM.barplot!(ax_best, xvals, [row.best_test_accuracy for row in sorted], color = :steelblue)
    ax_best.xticks = (xvals, [row.config for row in sorted])
    ax_best.xticklabelrotation = pi / 3
    CM.axislegend(ax_test, position = :rb, nbanks = 2)

    path = joinpath(root, "pair_summary.png")
    CM.save(path, fig)
    return path
end

"""Write the exact grid settings needed to reproduce the selected pair sweep."""
function write_grid_settings!(path::P, base::C, configs::V) where {P<:AbstractString,C<:LocalMNISTManagerConfig,V<:AbstractVector}
    pair_labels = ["$(config.input_hidden_radius):$(config.hidden_hidden_radius)" for config in configs]
    open(path, "w") do io
        println(io, "# Two-Hidden Local MNIST Pair Grid")
        println(io)
        println(io, "- architecture family: `28^2 input fields -> $(base.hidden1_side)^2 hidden1 -> $(base.hidden2_side)^2 hidden2 -> $(PMNIST_NCLASSES * base.output_replicas) outputs`")
        println(io, "- selected pairs: `$(join(pair_labels, ", "))`")
        println(io, "- workers / batchsize / epochs: `$(base.workers)` / `$(base.batchsize)` / `$(base.epochs)`")
        println(io, "- train/test per class: `$(base.train_per_class)` / `$(base.test_per_class)`")
        println(io, "- optimizer: `$(base.optimizer)`")
        println(io, "- learning rates W0/W12/W2O/B: `$(base.lr_w0)` / `$(base.lr_w12)` / `$(base.lr_w2o)` / `$(base.lr_b)`")
        println(io, "- free/nudge sweeps: `$(base.free_sweeps)` / `$(base.nudge_sweeps)`")
        println(io, "- beta: `$(base.β)`")
        println(io, "- gradient normalization: `$(base.gradient_normalization)`")
    end
    return path
end

"""Run the selected two-hidden `(r1, r2)` grid."""
function main()
    base = LocalMNISTManagerConfig()
    root = get(
        ENV,
        "ISING_MNIST_2H_GRID_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "r1_r2_pair_grid_e$(base.epochs)_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(root)
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers

    configs = two_hidden_configs(base, root)
    write_grid_settings!(joinpath(root, "grid_settings.md"), base, configs)
    println("Running ", length(configs), " two-hidden MNIST config(s)")
    println("root = ", root)
    results = NamedTuple[]
    summaries = NamedTuple[]
    for (idx, config) in enumerate(configs)
        println("[$idx/", length(configs), "] ", config.name)
        result = run_config!(config)
        push!(results, result)
        push!(summaries, two_hidden_summary_row(result))
    end

    summary_path = joinpath(root, "pair_summary.csv")
    isfile(summary_path) && rm(summary_path)
    for row in summaries
        append_row!(summary_path, row)
    end
    plot_path = plot_two_hidden_grid(root, results, summaries)
    println("saved summary ", summary_path)
    println("saved plot ", plot_path)
    return (; root, results, summaries, plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
