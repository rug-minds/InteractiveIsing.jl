using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include("edge_signal_xor.jl")

"""
    SnapshotTrace

Stores state snapshots and scalar diagnostics from one relaxation trajectory.
"""
mutable struct SnapshotTrace
    sweeps::Vector{Int}
    states::Vector{Vector{FT}}
    outputs::Vector{FT}
    hidden_norms::Vector{FT}
end

"""Create an empty snapshot trace."""
SnapshotTrace() = SnapshotTrace(Int[], Vector{FT}[], FT[], FT[])

"""
    record_snapshot!(trace, graph, sweep, hidden_range, output_idx)

Copy the current graph state into `trace` and label it by the completed number
of full sweeps.
"""
function record_snapshot!(trace::SnapshotTrace, graph, sweep::Int, hidden_range, output_idx::Int)
    s = copy(II.state(graph))
    push!(trace.sweeps, sweep)
    push!(trace.states, s)
    push!(trace.outputs, FT(s[output_idx]))
    push!(trace.hidden_norms, FT(norm(@view s[hidden_range])))
    return trace
end

"""
    RelaxationSnapshotLogger(trace, next_sweep, sweep_increment, hidden_range, output_idx)

ProcessAlgorithm that copies the graph state whenever it is scheduled.
"""
struct RelaxationSnapshotLogger{Trace,HiddenRange} <: Processes.ProcessAlgorithm
    trace::Trace
    next_sweep::Base.RefValue{Int}
    sweep_increment::Int
    hidden_range::HiddenRange
    output_idx::Int
end

Processes.init(::RelaxationSnapshotLogger, context) = (;)

function Processes.step!(logger::RelaxationSnapshotLogger, context)
    record_snapshot!(logger.trace, context.model, logger.next_sweep[], logger.hidden_range, logger.output_idx)
    logger.next_sweep[] += logger.sweep_increment
    return (;)
end

"""Return the four physical XOR input vectors used by this experiment."""
function snapshot_xor_inputs()
    x = Matrix{FT}(undef, 2, 4)
    for (idx, (a, b)) in enumerate(XOR_CASES)
        x[:, idx] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
    end
    return x
end

"""Cosine similarity between two state vectors."""
function cosine_similarity(a, b)
    denom = norm(a) * norm(b)
    return denom == 0 ? NaN : FT(dot(a, b) / denom)
end

"""Cosine similarity after subtracting each vector's mean."""
function centered_cosine_similarity(a, b)
    ac = a .- mean(a)
    bc = b .- mean(b)
    return cosine_similarity(ac, bc)
end

"""Write one row per saved state snapshot."""
function write_snapshot_csv(path, trace::SnapshotTrace)
    mkpath(dirname(path))
    open(path, "w") do io
        n = length(first(trace.states))
        println(io, join(vcat(["snapshot", "sweep", "output", "hidden_norm"], ["s$i" for i in 1:n]), ","))
        for idx in eachindex(trace.states)
            println(io, join(vcat(
                Any[idx - 1, trace.sweeps[idx], trace.outputs[idx], trace.hidden_norms[idx]],
                trace.states[idx],
            ), ","))
        end
    end
    return path
end

"""Write cosine similarities between consecutive saved snapshots."""
function write_similarity_csv(path, trace::SnapshotTrace)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "from_snapshot,to_snapshot,from_sweep,to_sweep,cosine,centered_cosine,relative_change_norm,output_from,output_to")
        for idx in 1:(length(trace.states)-1)
            a = trace.states[idx]
            b = trace.states[idx+1]
            delta_norm = norm(b .- a) / max(norm(a), eps(FT))
            println(io, join((
                idx - 1,
                idx,
                trace.sweeps[idx],
                trace.sweeps[idx+1],
                cosine_similarity(a, b),
                centered_cosine_similarity(a, b),
                delta_norm,
                trace.outputs[idx],
                trace.outputs[idx+1],
            ), ","))
        end
    end
    return path
end

"""Plot output value, hidden norm, and consecutive cosine similarity over snapshots."""
function plot_snapshot_summary(path, trace::SnapshotTrace)
    cosines = [cosine_similarity(trace.states[i], trace.states[i+1]) for i in 1:(length(trace.states)-1)]
    centered = [centered_cosine_similarity(trace.states[i], trace.states[i+1]) for i in 1:(length(trace.states)-1)]
    fig = Figure(size = (1000, 760))
    ax1 = Axis(fig[1, 1], title = "output during relaxation", xlabel = "sweep", ylabel = "output spin")
    ax2 = Axis(fig[1, 2], title = "hidden norm", xlabel = "sweep", ylabel = "||hidden||")
    ax3 = Axis(fig[2, 1], title = "cosine between snapshots", xlabel = "from sweep", ylabel = "cosine")
    ax4 = Axis(fig[2, 2], title = "centered cosine between snapshots", xlabel = "from sweep", ylabel = "centered cosine")
    lines!(ax1, trace.sweeps, trace.outputs)
    lines!(ax2, trace.sweeps, trace.hidden_norms)
    lines!(ax3, trace.sweeps[1:end-1], cosines)
    lines!(ax4, trace.sweeps[1:end-1], centered)
    save(path, fig)
    return path
end

"""Plot hidden-layer snapshots as 8x8 heatmaps."""
function plot_hidden_snapshots(path, graph, trace::SnapshotTrace; max_panels::Int = 12)
    hidden_range = collect(II.layerrange(graph[2]))
    nplots = min(max_panels, length(trace.states))
    picks = unique(round.(Int, range(1, length(trace.states), length = nplots)))
    fig = Figure(size = (1200, 800))
    for (panel_idx, snap_idx) in enumerate(picks)
        row = (panel_idx - 1) ÷ 4 + 1
        col = (panel_idx - 1) % 4 + 1
        ax = Axis(fig[row, col], title = "snapshot $(snap_idx - 1), sweep $(trace.sweeps[snap_idx])")
        h = reshape(trace.states[snap_idx][hidden_range], 8, 8)
        heatmap!(ax, h; colorrange = (-1, 1), colormap = :balance)
        hidedecorations!(ax)
    end
    save(path, fig)
    return path
end

"""Run one learned-graph relaxation and save snapshots plus consecutive similarities."""
function main()
    graph_path = get(
        ENV,
        "EDGE_SNAPSHOT_GRAPH",
        joinpath(@__DIR__, "runs", "edge_signal_grid_20260512_154608", "05_NN2_T0p025_io0p16_h0p025_lr0p002", "best_graph.jld2"),
    )
    case_idx = parse(Int, get(ENV, "EDGE_SNAPSHOT_CASE", "1"))
    total_sweeps = parse(Int, get(ENV, "EDGE_SNAPSHOT_SWEEPS", "500"))
    every_sweeps = parse(Int, get(ENV, "EDGE_SNAPSHOT_EVERY", "10"))
    stepsize = parse(FT, get(ENV, "EDGE_SNAPSHOT_STEPSIZE", "0.4"))
    temperature_override = get(ENV, "EDGE_SNAPSHOT_TEMP", "")
    seed = parse(Int, get(ENV, "EDGE_SNAPSHOT_SEED", "910000"))
    outdir = get(
        ENV,
        "EDGE_SNAPSHOT_DIR",
        joinpath(@__DIR__, "runs", "relaxation_snapshots_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)

    graph = II.load_isinggraph(graph_path)
    if !isempty(temperature_override)
        II.temp!(graph, parse(FT, temperature_override))
    end
    x = snapshot_xor_inputs()
    Random.seed!(seed)
    randomize_graph_state!(graph)
    apply_edge_input!(graph, view(x, :, case_idx))

    sweep_steps = length(II.sampling_indices(graph.index_set))
    interval_steps = every_sweeps * sweep_steps
    total_steps = total_sweeps * sweep_steps
    trace = SnapshotTrace()
    hidden_range = collect(II.layerrange(graph[2]))
    output_idx = only(II.layerrange(graph[end]))

    sampler = II.LocalLangevin(
        stepsize = stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )

    record_snapshot!(trace, graph, 0, hidden_range, output_idx)
    logger = RelaxationSnapshotLogger(trace, Ref(every_sweeps), every_sweeps, hidden_range, output_idx)
    dynamics = deepcopy(sampler)
    routine = Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @every interval_steps logger(model = dynamics.model)
    end
    wrapped = Processes.@Routine begin
        @repeat total_steps routine()
    end
    inputs = II._merge_graph_inputs(wrapped, graph, Processes.Input(dynamics, rng = Random.MersenneTwister(seed + 1)))
    process = Processes.Process(Processes.resolve(wrapped), inputs...; repeats = 1)
    run(process)
    wait(process)
    close(process)

    snapshots_csv = write_snapshot_csv(joinpath(outdir, "snapshots.csv"), trace)
    similarities_csv = write_similarity_csv(joinpath(outdir, "snapshot_similarities.csv"), trace)
    summary_png = plot_snapshot_summary(joinpath(outdir, "snapshot_summary.png"), trace)
    hidden_png = plot_hidden_snapshots(joinpath(outdir, "hidden_snapshots.png"), graph, trace)

    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Relaxation Snapshot Diagnostic")
        println(io)
        println(io, "- graph: `$graph_path`")
        println(io, "- case index: `$case_idx`")
        println(io, "- total sweeps: `$total_sweeps`")
        println(io, "- saved every sweeps: `$every_sweeps`")
        println(io, "- steps per sweep: `$sweep_steps`")
        println(io, "- stepsize: `$stepsize`")
        println(io, "- temperature: `$(graph.temp)`")
        println(io)
        println(io, "Files:")
        println(io, "- `snapshots.csv`: full state at each saved snapshot")
        println(io, "- `snapshot_similarities.csv`: cosine similarities from snapshot `i` to `i+1`")
        println(io, "- `snapshot_summary.png`")
        println(io, "- `hidden_snapshots.png`")
    end

    println("Saved relaxation snapshot diagnostic: ", outdir)
    return (; outdir, snapshots_csv, similarities_csv, summary_png, hidden_png)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
