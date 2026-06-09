using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Random
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const StatefulAlgorithms = II.StatefulAlgorithms
const INLAID_FT = Float32
const INLAID_INPUT_SIDE = 28
const INLAID_SIDE = 2 * INLAID_INPUT_SIDE - 1
const INLAID_NCLASSES = 10

"""
Static active-index set for inlaid-input experiments.

The fixed MNIST pixel sites live inside the same 2D input layer as the separator
spins, but they are excluded from the Monte Carlo proposal list. This keeps the
pixels clamped by state while every separator site and output spin remains live.
"""
struct StaticActiveIndexSet{V<:AbstractVector{Int32}} <: II.UniformIndexPicker
    active::V
end

"""Return the active graph indices used by sweep-style algorithms."""
II.sampling_indices(index_set::StaticActiveIndexSet) = index_set.active

"""Sample one active graph index uniformly."""
II.pick_idx(rng::Random.AbstractRNG, index_set::StaticActiveIndexSet) = rand(rng, index_set.active)

Base.length(index_set::StaticActiveIndexSet) = length(index_set.active)

Base.@kwdef struct InlaidDiagnosticConfig
    workers_list::Vector{Int} = parse_int_list(get(ENV, "ISING_MNIST_INLAID_WORKERS_LIST", "1,8,16,32"), [1, 8, 16, 32])
    jobs_per_worker::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_JOBS_PER_WORKER", "2"))
    repeats::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_REPEATS", "2"))
    sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_SWEEPS", "50"))
    relaxation_sweeps::Vector{Int} = parse_int_list(get(ENV, "ISING_MNIST_INLAID_RELAX_SWEEPS", "10,25,50,75,100"), [10, 25, 50, 75, 100])
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_OUTPUT_REPLICAS", "4"))
    input_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_INPUT_RADIUS", "1"))
    output_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_OUTPUT_RADIUS", "1"))
    input_internal_scale::INLAID_FT = parse(INLAID_FT, get(ENV, "ISING_MNIST_INLAID_INPUT_SCALE", "0.15"))
    readout_scale::INLAID_FT = parse(INLAID_FT, get(ENV, "ISING_MNIST_INLAID_READOUT_SCALE", "0.10"))
    output_internal_scale::INLAID_FT = parse(INLAID_FT, get(ENV, "ISING_MNIST_INLAID_OUTPUT_SCALE", "0.02"))
    hot_temp::INLAID_FT = parse(INLAID_FT, get(ENV, "ISING_MNIST_INLAID_HOT_TEMP", "5.0"))
    cold_temp::INLAID_FT = parse(INLAID_FT, get(ENV, "ISING_MNIST_INLAID_COLD_TEMP", "0.05"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_SEED", "4242"))
    outdir::String = get(ENV, "ISING_MNIST_INLAID_OUTDIR", joinpath(@__DIR__, "runs", "inlaid-input"))
end

mutable struct InlaidDiagnosticModel{C,G,P,A,O,R}
    config::C
    graph::G
    pixel_idxs::P
    active_idxs::A
    output_idxs::O
    rng::R
end

struct InlaidRelaxJob{X<:AbstractVector{INLAID_FT}}
    x::X
end

mutable struct InlaidDiagnosticState{M}
    model::M
    elapsed::Base.RefValue{Float64}
    jobs::Base.RefValue{Int}
    checksum::Base.RefValue{INLAID_FT}
end

"""Parse a comma-separated integer list."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Return a compact rectangular shape for a replica output layer."""
function output_shape(units::I) where {I<:Integer}
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Return graph indices for the clamped pixel sites inside the inlaid layer."""
function inlaid_pixel_indices(graph::G) where {G}
    layer = graph[1]
    idxs = reshape(collect(II.layerrange(layer)), INLAID_SIDE, INLAID_SIDE)
    pixels = Vector{Int32}(undef, INLAID_INPUT_SIDE^2)
    out_idx = 1
    @inbounds for col in 1:INLAID_INPUT_SIDE, row in 1:INLAID_INPUT_SIDE
        pixels[out_idx] = Int32(idxs[2 * row - 1, 2 * col - 1])
        out_idx += 1
    end
    return pixels
end

"""Build the static active set: separator sites and output spins only."""
function inlaid_active_indices(graph::G) where {G}
    pixels = Set(inlaid_pixel_indices(graph))
    active = Int32[]
    for idx in II.layerrange(graph[1])
        idx32 = Int32(idx)
        idx32 in pixels && continue
        push!(active, idx32)
    end
    append!(active, Int32.(collect(II.layerrange(graph[2]))))
    return active
end

"""Create the inlaid-input graph with fixed-pixel active-set semantics."""
function sampled_inlaid_graph(config::C, rng::R; shared_adj = nothing) where {C<:InlaidDiagnosticConfig,R<:Random.AbstractRNG}
    output_rows, output_cols = output_shape(INLAID_NCLASSES * config.output_replicas)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    input = II.Layer(INLAID_SIDE, INLAID_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, INLAID_SIDE + 3, 0); periodic = false)
    graph = II.IsingGraph(
        input,
        zero_wg,
        out,
        II.Bilinear() + II.MagField(b = g -> II.filltype(Vector, 0f0, II.statelen(g)));
        precision = INLAID_FT,
        adj = shared_adj,
        index_set = g -> StaticActiveIndexSet(inlaid_active_indices(g)),
    )
    if isnothing(shared_adj)
        add_grid_edges!(graph, 1, rng; radius = config.input_internal_radius, scale = config.input_internal_scale)
        add_grid_edges!(graph, 2, rng; radius = config.output_internal_radius, scale = config.output_internal_scale)
        add_dense_readout_edges!(graph, rng; scale = config.readout_scale)
    end
    II.temp!(graph, config.cold_temp)
    return graph
end

"""Add random symmetric local couplings inside one rectangular graph layer."""
function add_grid_edges!(
    graph::G,
    layer_idx::I,
    rng::R;
    radius::J,
    scale::T,
) where {G,I<:Integer,R<:Random.AbstractRNG,J<:Integer,T<:AbstractFloat}
    radius <= 0 && return graph
    scale == 0 && return graph
    layer = graph[Int(layer_idx)]
    rows, cols = size(layer)
    idxs = reshape(collect(II.layerrange(layer)), rows, cols)
    adj = II.adj(graph)
    @inbounds for col in 1:cols, row in 1:rows
        src = idxs[row, col]
        for dcol in -Int(radius):Int(radius), drow in -Int(radius):Int(radius)
            drow == 0 && dcol == 0 && continue
            dst_row = row + drow
            dst_col = col + dcol
            (1 <= dst_row <= rows && 1 <= dst_col <= cols) || continue
            dst = idxs[dst_row, dst_col]
            dst <= src && continue
            w = scale * randn(rng, INLAID_FT)
            adj[dst, src] = w
            adj[src, dst] = w
        end
    end
    return graph
end

"""Add random dense couplings from the inlaid layer to the output replicas."""
function add_dense_readout_edges!(graph::G, rng::R; scale::T) where {G,R<:Random.AbstractRNG,T<:AbstractFloat}
    scale == 0 && return graph
    input_idxs = collect(II.layerrange(graph[1]))
    output_idxs = collect(II.layerrange(graph[2]))
    adj = II.adj(graph)
    weight_scale = INLAID_FT(scale) / sqrt(INLAID_FT(length(input_idxs)))
    @inbounds for output_idx in output_idxs, input_idx in input_idxs
        w = weight_scale * randn(rng, INLAID_FT)
        adj[output_idx, input_idx] = w
        adj[input_idx, output_idx] = w
    end
    return graph
end

"""Create the model wrapper used by diagnostics and manager workers."""
function init_inlaid_model(config::C, seed::I = config.seed; shared_adj = nothing) where {C<:InlaidDiagnosticConfig,I<:Integer}
    rng = Random.MersenneTwister(Int(seed))
    graph = sampled_inlaid_graph(config, rng; shared_adj)
    return InlaidDiagnosticModel(
        config,
        graph,
        inlaid_pixel_indices(graph),
        inlaid_active_indices(graph),
        Int32.(collect(II.layerrange(graph[2]))),
        rng,
    )
end

"""Make one worker model with local state and a shared read-only adjacency."""
function worker_model(source::M, worker_idx::I) where {M<:InlaidDiagnosticModel,I<:Integer}
    model = init_inlaid_model(source.config, source.config.seed + 10_000 + Int(worker_idx); shared_adj = II.adj(source.graph))
    return model
end

"""Write one synthetic input pattern into the fixed pixel sites."""
function apply_inlaid_pattern!(model::M, x::X) where {M<:InlaidDiagnosticModel,X<:AbstractVector{INLAID_FT}}
    @inbounds II.state(model.graph)[model.pixel_idxs] .= x
    return model
end

"""Randomize only the live spins, leaving inlaid pixel values untouched."""
function randomize_live_state!(model::M) where {M<:InlaidDiagnosticModel}
    state = II.state(model.graph)
    @inbounds for idx in model.active_idxs
        state[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    return model
end

"""Evaluate the current sparse Hamiltonian energy."""
function graph_energy(model::M) where {M<:InlaidDiagnosticModel}
    state = II.state(model.graph)
    bias = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    adj = II.adj(model.graph)
    colptrs = SparseArrays.getcolptr(adj)
    rowvals = SparseArrays.rowvals(adj)
    nzvals = SparseArrays.nonzeros(adj)
    energy = 0f0
    @inbounds for col in 1:size(adj, 2)
        for ptr in colptrs[col]:(colptrs[col + 1] - 1)
            energy -= 0.5f0 * nzvals[ptr] * state[rowvals[ptr]] * state[col]
        end
    end
    @inbounds for idx in eachindex(state)
        energy -= bias[idx] * state[idx]
    end
    return energy
end

"""Run full active-spin Metropolis sweeps with a geometric cooling schedule."""
function anneal_inlaid!(model::M, context::C, sweeps::I) where {M<:InlaidDiagnosticModel,C,I<:Integer}
    steps = max(1, Int(sweeps) * length(model.active_idxs))
    dynamics_algorithm = II.Metropolis()
    temperature = GeometricTemperatureSchedule(; start_T = model.config.hot_temp, stop_T = model.config.cold_temp, n_steps = steps)
    algorithm = StatefulAlgorithms.resolve(StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias relax_temperature = temperature

        @repeat steps begin
            relax_temperature(dynamics.model)
            dynamics()
        end
        II.temp!(dynamics.model, model.config.cold_temp)
    end)
    process = StatefulAlgorithms.Process(algorithm, StatefulAlgorithms.Init(:dynamics; model = model.graph); repeat = 1)
    run(process)
    wait(process)
    return model
end

"""Return output replica means for a relaxed state."""
function class_scores(model::M) where {M<:InlaidDiagnosticModel}
    output = @view II.state(model.graph)[model.output_idxs]
    scores = zeros(INLAID_FT, INLAID_NCLASSES)
    replicas = model.config.output_replicas
    @inbounds for digit in 1:INLAID_NCLASSES
        start_idx = (digit - 1) * replicas + 1
        scores[digit] = sum(view(output, start_idx:(start_idx + replicas - 1))) / replicas
    end
    return scores
end

"""Copy fixed inlaid pixel states into a reusable diagnostic buffer."""
function copy_pixel_state!(dest::D, model::M) where {D<:AbstractVector,M<:InlaidDiagnosticModel}
    dest .= @view II.state(model.graph)[model.pixel_idxs]
    return dest
end

"""Assert that a diagnostic relaxation did not modify fixed pixel states."""
function assert_pixel_state!(model::M, expected::E) where {M<:InlaidDiagnosticModel,E<:AbstractVector}
    all(expected .== @view II.state(model.graph)[model.pixel_idxs]) || error("Inlaid pixel states changed during relaxation")
    return nothing
end

"""Accumulate diagnostic timing and checksum counters for one worker job."""
function update_diagnostic_stats!(
    elapsed::Base.RefValue{F},
    jobs::Base.RefValue{I},
    checksum::Base.RefValue{T},
    started_at::U,
    model::M,
) where {F<:Real,I<:Integer,T<:Real,U<:Integer,M<:InlaidDiagnosticModel}
    elapsed[] += (time_ns() - started_at) / 1.0e9
    jobs[] += 1
    checksum[] += sum(class_scores(model))
    return nothing
end

"""Build one temperature-scheduled diagnostic relaxation routine."""
function inlaid_relax_algorithm(config::C, nactive::I) where {C<:InlaidDiagnosticConfig,I<:Integer}
    steps = max(1, Int(config.sweeps) * Int(nactive))
    temperature = GeometricTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = steps)
    dynamics_algorithm = II.Metropolis()
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias relax_temperature = temperature
        @state inlaid_model
        @state x
        @state pixel_state
        @state elapsed
        @state jobs
        @state checksum

        started_at = time_ns()
        apply_inlaid_pattern!(inlaid_model, x)
        copy_pixel_state!(pixel_state, inlaid_model)
        randomize_live_state!(inlaid_model)
        @repeat steps begin
            relax_temperature(dynamics.model)
            dynamics()
        end
        II.temp!(dynamics.model, inlaid_model.config.cold_temp)
        assert_pixel_state!(inlaid_model, pixel_state)
        update_diagnostic_stats!(elapsed, jobs, checksum, started_at, inlaid_model)
    end
end

"""Return the mutable process subcontext used by a manager worker."""
function worker_context(worker::W) where {W}
    return StatefulAlgorithms.context(worker)._state
end

"""Create one reusable manager-owned inlaid-input worker."""
function inlaid_worker(source::M, worker_idx::I, algorithm::A) where {M<:InlaidDiagnosticModel,I<:Integer,A}
    model = worker_model(source, worker_idx)
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            inlaid_model = model,
            x = zeros(INLAID_FT, INLAID_INPUT_SIDE^2),
            pixel_state = zeros(INLAID_FT, length(model.pixel_idxs)),
            elapsed = Ref(0.0),
            jobs = Ref(0),
            checksum = Ref(0f0),
        ),
        StatefulAlgorithms.Init(:dynamics; model = model.graph);
        repeat = 1,
    )
end

"""Create a ProcessManager for inlaid-input diagnostic relaxation jobs."""
function inlaid_manager(source::M, nworkers::I) where {M<:InlaidDiagnosticModel,I<:Integer}
    state = InlaidDiagnosticState(source, Ref(0.0), Ref(0), Ref(0f0))
    algorithm = StatefulAlgorithms.resolve(inlaid_relax_algorithm(source.config, length(source.active_idxs)))
    recipe = (;
        makeworker = (idx, manager) -> inlaid_worker(manager.state.model, idx, algorithm),
        loadjob! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x .= job.x
            StatefulAlgorithms.resetworker!(slot)
            return nothing
        end,
        sync_to_state! = manager -> begin
            manager.state.elapsed[] = 0.0
            manager.state.jobs[] = 0
            manager.state.checksum[] = 0f0
            for worker in StatefulAlgorithms.workers(manager)
                ctx = worker_context(worker)
                manager.state.elapsed[] += ctx.elapsed[]
                manager.state.jobs[] += ctx.jobs[]
                manager.state.checksum[] += ctx.checksum[]
            end
            return nothing
        end,
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        config = source.config,
        state,
        sync_policy = StatefulAlgorithms.SyncAtEnd(),
        execution = StatefulAlgorithms.ChannelWorkers(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = InlaidRelaxJob{Vector{INLAID_FT}},
    )
end

"""Clear accumulated diagnostic timers on all manager workers."""
function reset_diagnostic_worker_stats!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    manager.state.elapsed[] = 0.0
    manager.state.jobs[] = 0
    manager.state.checksum[] = 0f0
    for worker in StatefulAlgorithms.workers(manager)
        ctx = worker_context(worker)
        ctx.elapsed[] = 0.0
        ctx.jobs[] = 0
        ctx.checksum[] = 0f0
    end
    return manager
end

"""Build deterministic random binary patterns for diagnostics."""
function synthetic_jobs(config::C, n::I) where {C<:InlaidDiagnosticConfig,I<:Integer}
    rng = Random.MersenneTwister(config.seed + 123)
    jobs = Vector{InlaidRelaxJob{Vector{INLAID_FT}}}(undef, Int(n))
    for idx in eachindex(jobs)
        x = Vector{INLAID_FT}(undef, INLAID_INPUT_SIDE^2)
        @inbounds for j in eachindex(x)
            x[j] = rand(rng, Bool) ? 1f0 : -1f0
        end
        jobs[idx] = InlaidRelaxJob(x)
    end
    return jobs
end

"""Measure manager latency and worker-internal relaxation time."""
function run_scaling_probe(config::C) where {C<:InlaidDiagnosticConfig}
    results = NamedTuple[]
    for nworkers in config.workers_list
        Threads.nthreads() < nworkers && @warn "Julia has fewer threads than requested workers" threads = Threads.nthreads() workers = nworkers
        source = init_inlaid_model(config, config.seed + nworkers)
        manager = inlaid_manager(source, nworkers)
        jobs = synthetic_jobs(config, nworkers * config.jobs_per_worker)

        # Warm the compiled path and worker contexts before timing.
        StatefulAlgorithms.run!(manager, jobs)

        elapsed_runs = Float64[]
        internal_runs = Float64[]
        for _ in 1:config.repeats
            reset_diagnostic_worker_stats!(manager)
            start = time_ns()
            StatefulAlgorithms.run!(manager, jobs)
            elapsed = (time_ns() - start) / 1.0e9
            push!(elapsed_runs, elapsed)
            push!(internal_runs, manager.state.elapsed[] / max(manager.state.jobs[], 1))
        end
        push!(results, (;
            workers = nworkers,
            jobs = length(jobs),
            sweeps = config.sweeps,
            elapsed_s = mean(elapsed_runs),
            jobs_per_s = length(jobs) / mean(elapsed_runs),
            worker_internal_s_per_job = mean(internal_runs),
            checksum = manager.state.checksum[],
        ))
    end
    return results
end

"""Check whether relaxation sweeps materially change energy and output state."""
function run_relaxation_probe(config::C) where {C<:InlaidDiagnosticConfig}
    model = init_inlaid_model(config, config.seed + 99)
    context = StatefulAlgorithms.init(II.Metropolis(), (; model = model.graph))
    rng = Random.MersenneTwister(config.seed + 321)
    x = [rand(rng, Bool) ? 1f0 : -1f0 for _ in 1:INLAID_INPUT_SIDE^2]
    rows = NamedTuple[]
    for sweeps in config.relaxation_sweeps
        energies = INLAID_FT[]
        margins = INLAID_FT[]
        pixel_ok = true
        for _ in 1:3
            apply_inlaid_pattern!(model, x)
            before_pixels = copy(@view II.state(model.graph)[model.pixel_idxs])
            randomize_live_state!(model)
            anneal_inlaid!(model, context, sweeps)
            pixel_ok &= all(before_pixels .== @view II.state(model.graph)[model.pixel_idxs])
            scores = class_scores(model)
            sorted = sort(scores; rev = true)
            push!(energies, graph_energy(model))
            push!(margins, sorted[1] - sorted[2])
        end
        push!(rows, (;
            sweeps,
            energy_mean = mean(energies),
            energy_std = std(energies),
            margin_mean = mean(margins),
            margin_std = std(margins),
            pixel_ok,
        ))
    end
    return rows
end

"""Write CSV diagnostics and a concise architecture-specific findings note."""
function write_diagnostics!(config::C, scaling_rows::R, relaxation_rows::Q) where {C<:InlaidDiagnosticConfig,R,Q}
    mkpath(config.outdir)
    scaling_path = joinpath(config.outdir, "scaling.csv")
    open(scaling_path, "w") do io
        println(io, "workers,jobs,sweeps,elapsed_s,jobs_per_s,worker_internal_s_per_job,checksum")
        for row in scaling_rows
            println(io, join((row.workers, row.jobs, row.sweeps, row.elapsed_s, row.jobs_per_s, row.worker_internal_s_per_job, row.checksum), ","))
        end
    end

    relaxation_path = joinpath(config.outdir, "relaxation.csv")
    open(relaxation_path, "w") do io
        println(io, "sweeps,energy_mean,energy_std,margin_mean,margin_std,pixel_ok")
        for row in relaxation_rows
            println(io, join((row.sweeps, row.energy_mean, row.energy_std, row.margin_mean, row.margin_std, row.pixel_ok), ","))
        end
    end

    findings_path = joinpath(@__DIR__, "inlaid_input_findings.md")
    open(findings_path, "w") do io
        println(io, "# Inlaid Input Findings")
        println(io)
        println(io, "Use of this file: concise notes for the MNIST architecture where the 28x28 pixels are inlaid into a 55x55 layer and separated by live spins.")
        println(io)
        println(io, "## Current Diagnostic")
        println(io)
        println(io, "- Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia threads: $(Threads.nthreads())")
        println(io, "- Architecture: 55x55 input layer with 784 fixed pixel sites, $(INLAID_SIDE^2 - INLAID_INPUT_SIDE^2) live separator sites, and $(INLAID_NCLASSES * config.output_replicas) output spins.")
        println(io, "- Active set: static proposal list excluding only the inlaid pixel sites; this keeps pixels clamped by state without turning off the separator spins.")
        println(io, "- Diagnostic output: `$scaling_path` and `$relaxation_path`.")
        println(io)
        println(io, "## Relaxation Probe")
        println(io)
        for row in relaxation_rows
            println(io, "- $(row.sweeps) sweeps: energy mean $(round(row.energy_mean; digits = 3)), margin mean $(round(row.margin_mean; digits = 3)), pixels fixed = $(row.pixel_ok).")
        end
        println(io)
        println(io, "## Scaling Probe")
        println(io)
        base_throughput = first(scaling_rows).jobs_per_s
        base_workers = first(scaling_rows).workers
        for row in scaling_rows
            speedup = row.jobs_per_s / base_throughput
            ideal = row.workers / base_workers
            worker_label = row.workers == 1 ? "1 worker" : "$(row.workers) workers"
            base_label = base_workers == 1 ? "1 worker" : "$(base_workers) workers"
            println(io, "- $worker_label: $(round(row.elapsed_s; digits = 3)) s for $(row.jobs) jobs, $(round(row.jobs_per_s; digits = 2)) jobs/s, throughput speedup $(round(speedup; digits = 2))x vs $base_label; ideal $(round(ideal; digits = 2))x.")
        end
        println(io)
        println(io, "## Next Run Requirements")
        println(io)
        println(io, "- Keep this diagnostic separate from saved training runs.")
        println(io, "- Use 32 workers for the actual MNIST runs if scaling remains acceptable.")
        println(io, "- Keep pixel sites fixed by the active-set design; do not switch to whole-layer toggling for this architecture.")
    end
    return (; scaling_path, relaxation_path, findings_path)
end

"""Run all diagnostics for the inlaid-input MNIST architecture."""
function main()
    config = InlaidDiagnosticConfig()
    println("threads = $(Threads.nthreads())")
    println("architecture = 55x55 inlaid input -> $(INLAID_NCLASSES * config.output_replicas) outputs")
    println("sweeps = $(config.sweeps), workers_list = $(config.workers_list)")
    relaxation_rows = run_relaxation_probe(config)
    scaling_rows = run_scaling_probe(config)
    paths = write_diagnostics!(config, scaling_rows, relaxation_rows)
    println("wrote $(paths.findings_path)")
    println("scaling:")
    for row in scaling_rows
        println(row)
    end
    println("relaxation:")
    for row in relaxation_rows
        println(row)
    end
    return (; config, scaling_rows, relaxation_rows, paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
