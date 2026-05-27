using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using Dates

include(joinpath(@__DIR__, "..", "mnist_784_120_40_adam.jl"))

"""Parse a comma-separated integer list, returning `default` for an empty string."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

Base.@kwdef struct BaselineWorkerScalingConfig{S<:AbstractString,V<:AbstractVector{Int}}
    workers_list::V = parse_int_list(get(ENV, "ISING_MNIST_IF_DIAG_WORKERS", "1,8,16,32"), [1, 8, 16, 32])
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_EPOCHS", "1"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_BATCHSIZE", "128"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_TRAIN_PER_CLASS", "500"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_TEST_PER_CLASS", "100"))
    train_eval_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_TRAIN_EVAL_PER_CLASS", "20"))
    eval_every::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_EVAL_EVERY", "0"))
    sweeps::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_SWEEPS", "100"))
    β::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_BETA", "5.0"))
    lr::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_LR", "0.003"))
    temp::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_STEPSIZE", "0.5"))
    weight_scale::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_WEIGHT_SCALE", "0.005"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_MNIST_IF_DIAG_WEIGHT_DECAY", "0.0"))
    hidden::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_HIDDEN", "120"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_OUTPUT_REPLICAS", "4"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_IF_DIAG_SEED", "20260527"))
    outdir::S = get(
        ENV,
        "ISING_MNIST_IF_DIAG_OUTDIR",
        joinpath(@__DIR__, "current", "worker_scaling_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

"""Return a copy of the baseline config with selected keyword fields replaced."""
function copy_config(config::C; kwargs...) where {C<:InputFieldMNISTConfig}
    fields = Dict{Symbol,Any}(field => getfield(config, field) for field in fieldnames(C))
    for (field, value) in kwargs
        fields[field] = value
    end
    return InputFieldMNISTConfig(; fields...)
end

"""Return the last non-missing value for one metrics field, or `missing` if absent."""
function latest_metric(rows::R, field::F) where {R<:AbstractVector,F<:Symbol}
    for idx in length(rows):-1:1
        value = getproperty(rows[idx], field)
        ismissing(value) || return value
    end
    return missing
end

"""Build the baseline config used for one worker-count diagnostic run."""
function diagnostic_baseline_config(config::C, workers::I, outdir::S) where {C<:BaselineWorkerScalingConfig,I<:Integer,S<:AbstractString}
    base = InputFieldMNISTConfig()
    return copy_config(
        base;
        workers = Int(workers),
        epochs = config.epochs,
        batchsize = config.batchsize,
        train_per_class = config.train_per_class,
        test_per_class = config.test_per_class,
        train_eval_per_class = config.train_eval_per_class,
        eval_every = config.eval_every,
        hidden = config.hidden,
        output_replicas = config.output_replicas,
        sweeps = config.sweeps,
        β = config.β,
        lr = config.lr,
        temp = config.temp,
        stepsize = config.stepsize,
        weight_scale = config.weight_scale,
        weight_decay = config.weight_decay,
        seed = config.seed,
        outdir = outdir,
    )
end

"""Summarize one completed baseline run into a compact worker-scaling row."""
function summarize_run(workers::I, elapsed_seconds::T, result::R) where {I<:Integer,T<:Real,R<:NamedTuple}
    rows = result.rows
    train_seconds = sum(row.seconds for row in rows if row.epoch > 0)
    final_train_accuracy = latest_metric(rows, :train_accuracy)
    final_train_loss = latest_metric(rows, :train_loss)
    final_test_accuracy = latest_metric(rows, :test_accuracy)
    final_test_loss = latest_metric(rows, :test_loss)
    return (;
        workers = Int(workers),
        total_seconds = Float64(elapsed_seconds),
        train_seconds = Float64(train_seconds),
        overhead_seconds = Float64(elapsed_seconds) - Float64(train_seconds),
        final_train_accuracy,
        final_train_loss,
        final_test_accuracy,
        final_test_loss,
        best_accuracy = result.best_accuracy,
        outdir = result.config.outdir,
    )
end

"""Plot total/train/overhead timings and speedups against worker count."""
function plot_scaling_summary(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    workers = [row.workers for row in rows]
    total_seconds = [row.total_seconds for row in rows]
    train_seconds = [row.train_seconds for row in rows]
    overhead_seconds = [row.overhead_seconds for row in rows]
    base_total = first(total_seconds)
    base_train = first(train_seconds)
    total_speedup = [base_total / seconds for seconds in total_seconds]
    train_speedup = [base_train / seconds for seconds in train_seconds]

    fig = Figure(size = (1200, 900))
    ax_total = Axis(fig[1, 1], xlabel = "workers", ylabel = "seconds", title = "MNIST baseline worker-scaling timings")
    ax_speedup = Axis(fig[2, 1], xlabel = "workers", ylabel = "speedup", title = "Speedup relative to the first run")

    lines!(ax_total, workers, total_seconds, label = "total", color = :steelblue)
    scatter!(ax_total, workers, total_seconds, color = :steelblue)
    lines!(ax_total, workers, train_seconds, label = "train", color = :orange)
    scatter!(ax_total, workers, train_seconds, color = :orange)
    lines!(ax_total, workers, overhead_seconds, label = "overhead", color = :seagreen)
    scatter!(ax_total, workers, overhead_seconds, color = :seagreen)
    axislegend(ax_total, position = :rt)

    lines!(ax_speedup, workers, total_speedup, label = "total", color = :steelblue)
    scatter!(ax_speedup, workers, total_speedup, color = :steelblue)
    lines!(ax_speedup, workers, train_speedup, label = "train", color = :orange)
    scatter!(ax_speedup, workers, train_speedup, color = :orange)
    lines!(ax_speedup, workers, Float64.(workers) ./ first(workers), label = "ideal", color = :gray, linestyle = :dash)
    axislegend(ax_speedup, position = :lt)

    save(path, fig)
    return path
end

"""Run the quick worker-scaling diagnostic with the baseline training path."""
function run_worker_scaling_diagnostic!(config::C) where {C<:BaselineWorkerScalingConfig}
    mkpath(config.outdir)
    summary_path = joinpath(config.outdir, "worker_scaling_summary.csv")
    rows = NamedTuple[]

    for workers in config.workers_list
        run_outdir = joinpath(config.outdir, "workers_$(workers)")
        run_config = diagnostic_baseline_config(config, workers, run_outdir)
        println("starting worker-scaling run with $(workers) workers")
        flush(stdout)

        elapsed_seconds = @elapsed result = run_config!(run_config)
        row = summarize_run(workers, elapsed_seconds, result)
        append_row!(summary_path, row)
        push!(rows, row)

        println(row)
        flush(stdout)
    end

    if !isempty(rows)
        base_total = rows[1].total_seconds
        base_train = rows[1].train_seconds
        speedup_rows = map(rows) do row
            (;
                workers = row.workers,
                total_speedup = base_total / row.total_seconds,
                train_speedup = base_train / row.train_seconds,
            )
        end
        speedup_path = joinpath(config.outdir, "worker_scaling_speedup.csv")
        for row in speedup_rows
            append_row!(speedup_path, row)
        end
        plot_scaling_summary(joinpath(config.outdir, "worker_scaling_summary.png"), rows)
    end

    return rows
end

"""Command-line entry point for the baseline worker-scaling diagnostic."""
function main()
    return run_worker_scaling_diagnostic!(BaselineWorkerScalingConfig())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
