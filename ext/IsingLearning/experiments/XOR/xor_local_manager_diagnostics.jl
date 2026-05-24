using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie
using Dates
using SparseArrays
using Statistics

include(joinpath(@__DIR__, "xor_local_cnn_like_grid.jl"))

"""Parse a comma-separated integer list for diagnostic worker sweeps."""
function diagnostic_ints(value::S) where {S<:AbstractString}
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Append one named-tuple row to a CSV file."""
function diagnostic_append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return counts of unique shared model arrays across manager workers."""
function sharing_counts(manager::M) where {M<:Processes.ProcessManager}
    adj_ids = UInt[]
    nz_ids = UInt[]
    bias_ids = UInt[]
    for worker in Processes.workers(manager)
        model = worker_context(worker).base_context.model
        push!(adj_ids, objectid(II.adj(model)))
        push!(nz_ids, objectid(SparseArrays.nonzeros(II.adj(model))))
        push!(bias_ids, objectid(II.getparam(model.hamiltonian, II.MagField, :b)))
    end
    return (;
        unique_adjacencies = length(unique(adj_ids)),
        unique_nzvals = length(unique(nz_ids)),
        unique_biases = length(unique(bias_ids)),
    )
end

"""Time a few manager training batches for one local CNN-like XOR setting."""
function time_local_cnn_setting(workers::Integer; radius::Integer, periodic::Bool, repeats::Integer, chunks::Integer, sweeps::Integer, nbatches::Integer)
    config = LocalCNNXORConfig(
        name = "diag_w$(workers)",
        workers = Int(workers),
        repeats_per_case = Int(repeats),
        chunks_per_case = Int(chunks),
        local_radius = Int(radius),
        hidden_periodic = periodic,
        free_sweeps = Int(sweeps),
        nudged_sweeps = Int(sweeps),
        epochs = Int(nbatches),
        log_every = 1,
    )
    graph = cnn_xor_graph(config)
    layer = cnn_xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 10), layer)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 11), layer)
    x, y = cnn_xor_dataset(config, FT)
    jobs = cnn_xor_jobs(config, x, y)

    build_seconds = @elapsed manager = cnn_xor_manager(layer, graph, ps, config)
    counts = sharing_counts(manager)
    before = evaluate_cnn_xor(layer, manager.state.params[], st, x, y, config)

    batch_times = Float64[]
    try
        # One warm-up batch removes first-use scheduler and generated-code noise.
        warmup_seconds = @elapsed run_cnn_batch!(manager, jobs, config)
        for _ in 1:Int(nbatches)
            push!(batch_times, @elapsed run_cnn_batch!(manager, jobs, config))
        end
        after = evaluate_cnn_xor(layer, manager.state.params[], st, x, y, config)
        return (;
            workers = Int(workers),
            radius = Int(radius),
            periodic,
            repeats_per_case = Int(repeats),
            chunks_per_case = Int(chunks),
            jobs = length(jobs),
            sweeps = Int(sweeps),
            nbatches = Int(nbatches),
            build_seconds,
            warmup_seconds,
            mean_batch_seconds = mean(batch_times),
            min_batch_seconds = minimum(batch_times),
            max_batch_seconds = maximum(batch_times),
            initial_mse = before.mse,
            final_mse = after.mse,
            initial_accuracy = before.accuracy,
            final_accuracy = after.accuracy,
            counts...,
        )
    finally
        close(manager)
    end
end

"""Plot batch latency versus worker count for the diagnostic run."""
function plot_diagnostic(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "manager workers", ylabel = "mean batch seconds", title = "Local CNN-like XOR manager timing")
    xs = [row.workers for row in rows]
    ys = [row.mean_batch_seconds for row in rows]
    scatterlines!(ax, xs, ys, markersize = 12)
    save(path, fig)
    return path
end

"""Run the local manager timing diagnostic and write CSV, plot, and note files."""
function main()
    workers = diagnostic_ints(get(ENV, "ISING_XOR_DIAG_WORKERS", "1,4,8,16,32"))
    radius = parse(Int, get(ENV, "ISING_XOR_DIAG_RADIUS", "3"))
    periodic = parse(Bool, lowercase(get(ENV, "ISING_XOR_DIAG_PERIODIC", "true")))
    repeats = parse(Int, get(ENV, "ISING_XOR_DIAG_REPEATS", "32"))
    chunks = parse(Int, get(ENV, "ISING_XOR_DIAG_CHUNKS_PER_CASE", "0"))
    sweeps = parse(Int, get(ENV, "ISING_XOR_DIAG_SWEEPS", "2"))
    nbatches = parse(Int, get(ENV, "ISING_XOR_DIAG_BATCHES", "3"))
    outdir = get(ENV, "ISING_XOR_DIAG_OUTDIR", joinpath(@__DIR__, "runs", "xor_local_manager_diagnostics_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(outdir)

    rows = NamedTuple[]
    csv_path = joinpath(outdir, "diagnostics.csv")
    isfile(csv_path) && rm(csv_path)
    for worker_count in workers
        row = time_local_cnn_setting(worker_count; radius, periodic, repeats, chunks, sweeps, nbatches)
        push!(rows, row)
        diagnostic_append_row!(csv_path, row)
        println(row)
        flush(stdout)
    end

    plot_path = plot_diagnostic(joinpath(outdir, "diagnostics.png"), rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# XOR Local Manager Diagnostics")
        println(io)
        println(io, "- radius: `$(radius)`")
        println(io, "- periodic hidden layers: `$(periodic)`")
        println(io, "- repeats per XOR case: `$(repeats)`")
        println(io, "- chunks per XOR case: `$(chunks == 0 ? cld(maximum(workers), length(CNN_XOR_CASES)) : chunks)`")
        println(io, "- sweeps per phase: `$(sweeps)`")
        println(io, "- measured batches after warm-up: `$(nbatches)`")
        println(io, "- expected sharing counts for worker graph arrays: `1` adjacency, `1` nonzero array, `1` bias vector")
        println(io)
        println(io, "Plot: `$(basename(plot_path))`")
        println(io, "Metrics: `$(basename(csv_path))`")
    end
    println("saved diagnostics ", outdir)
    return (; outdir, rows, plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
