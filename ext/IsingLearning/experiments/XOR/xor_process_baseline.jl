using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using LuxCore
using Optimisers
using Random
using Serialization
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float32

Base.@kwdef struct XORBaselineConfig
    epochs::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_EPOCHS", "1000"))
    workers::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_WORKERS", "32"))
    log_every::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_LOG_EVERY", "50"))
    repeats::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_REPEATS", "32"))
    chunks_per_case::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_CHUNKS_PER_CASE", "0"))
    hidden::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_HIDDEN", "8"))
    free_steps::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_FREE_STEPS", "1000"))
    nudged_steps::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_NUDGED_STEPS", "1000"))
    β::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_BETA", "1.0"))
    lr::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_LR", "0.002"))
    optimizer::String = lowercase(get(ENV, "ISING_XOR_BASELINE_OPTIMIZER", "adam"))
    temp::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_STEPSIZE", "0.05"))
    weight_scale::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_WEIGHT_SCALE", "0.2"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_XOR_BASELINE_BIAS_SCALE", "0.05"))
    seed::Int = parse(Int, get(ENV, "ISING_XOR_BASELINE_SEED", "13"))
    outdir::String = get(
        ENV,
        "ISING_XOR_BASELINE_OUTDIR",
        joinpath(@__DIR__, "runs", "xor_process_baseline_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

struct XORBaselineJob{X<:AbstractVector,Y<:AbstractVector}
    case_idx::Int
    x::X
    y::Y
    repeats::Int
end

struct AveragedBaselineContrastiveStep{S} <: Processes.ProcessAlgorithm
    base::S
end

mutable struct XORBaselineManagerState{P,B,O}
    params::Base.RefValue{P}
    batch_gradient::B
    total_repeats::Base.RefValue{Int}
    opt_state::O
end

"""Return the four bipolar one-hot XOR inputs and two-spin bipolar targets."""
function xor_dataset(::Type{T} = FT) where {T<:AbstractFloat}
    inputs = fill(-one(T), 4, 4)
    targets = Matrix{T}(undef, 2, 4)
    cases = ((false, false), (false, true), (true, false), (true, true))
    for (idx, (a, b)) in enumerate(cases)
        inputs[idx, idx] = one(T)
        targets[:, idx] .= xor(a, b) ? (-one(T), one(T)) : (one(T), -one(T))
    end
    return inputs, targets, cases
end

"""Build a small all-to-all `4 -> hidden -> 2` continuous Ising graph."""
function xor_graph(config::XORBaselineConfig)
    rng_w = Random.MersenneTwister(config.seed)
    rng_b = Random.MersenneTwister(config.seed + 1)
    input = II.Layer(4, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0); periodic = false)
    hidden = II.Layer(config.hidden, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 2, 0); periodic = false)
    output = II.Layer(2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 4, 0); periodic = false)
    weights = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> config.weight_scale * randn(rng_w, FT))
    bias = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    graph = II.IsingGraph(
        input,
        weights,
        hidden,
        deepcopy(weights),
        output,
        II.Bilinear() + II.MagField(b = bias) + II.Clamping(β = II.UniformArray(zero(FT)), y = target, mask = mask);
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Create the Lux graph layer with current process-based free/nudged dynamics."""
function xor_layer(graph::G, config::XORBaselineConfig) where {G}
    dynamics = II.BlockLangevin(
        stepsize = config.stepsize,
        adjusted = false,
        block_size = 1,
        group_steps = 1,
    )
    return LayeredIsingGraphLayer(
        graph;
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_steps,
        free_relaxation_steps = config.free_steps,
        nudged_relaxation_steps = config.nudged_steps,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Evaluate deterministic sign accuracy and mean squared error over XOR cases."""
function evaluate_xor(layer::L, ps::P, st::S, x::X, y::Y) where {L<:LayeredIsingGraphLayer,P,S,X,Y}
    outputs = similar(y)
    for sample_idx in axes(x, 2)
        out, st = layer(view(x, :, sample_idx), ps, st)
        outputs[:, sample_idx] .= out
    end
    predictions = getindex.(argmax(outputs; dims = 1), 1)
    targets = getindex.(argmax(y; dims = 1), 1)
    return (;
        mse = mean(abs2, outputs .- y),
        accuracy = mean(vec(predictions) .== vec(targets)),
        outputs = copy(outputs),
    )
end

"""Initialize the repeat wrapper around the standard `LayerContrastiveStep`."""
function Processes.init(step::AveragedBaselineContrastiveStep{S}, context) where {S}
    base_context = Processes.init(step.base, context)
    repeats = get(context, :repeats, Ref(1))
    repeats_ref = repeats isa Base.RefValue ? repeats : Ref(Int(repeats))
    return (; base_context, repeats = repeats_ref)
end

"""Run several contrastive repeats inside one already-started manager worker."""
function Processes.step!(step::AveragedBaselineContrastiveStep{S}, context) where {S}
    @inbounds for _ in 1:context.repeats[]
        Processes.step!(step.base, context.base_context)
    end
    return nothing
end

"""Keep the averaged baseline context reusable across manager jobs."""
function Processes.cleanup(step::AveragedBaselineContrastiveStep{S}, context) where {S}
    return nothing
end

"""Return the mutable training subcontext stored in one manager worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Create a worker graph with fresh state and shared static parameter arrays."""
function worker_graph(prototype::G, ps::P, config::C) where {G,P,C<:XORBaselineConfig}
    IsingLearning.sync_params!(prototype, ps)
    base_bias = II.Force(II.getparam(prototype.hamiltonian, II.MagField, :b))
    target = g -> II.filltype(Vector, zero(eltype(g)), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(eltype(g)), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = base_bias) + II.Clamping(β = II.UniformArray(zero(eltype(prototype))), y = target, mask = mask)
    graph = II.IsingGraph(
        getfield(prototype, :layers)...,
        hamiltonian;
        precision = eltype(prototype),
        adj = II.adj(prototype),
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Build one reusable worker process for the baseline contrastive step."""
function averaged_worker(layer::L, graph::G) where {L<:LayeredIsingGraphLayer,G}
    step = AveragedBaselineContrastiveStep(LayerContrastiveStep(layer))
    return Processes.Process(
        :_state => step,
        Processes.Init(:_state;
            model = graph,
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            repeats = Ref(1),
        );
        repeat = 1,
    )
end

"""Allocate a mutable gradient buffer matching the Lux parameter tree."""
function parameter_buffer(ps::P) where {P}
    buffer = (;
        w = zeros(eltype(ps.w), length(ps.w)),
        b = zeros(eltype(ps.b), length(ps.b)),
    )
    hasproperty(ps, :α) || return buffer
    return merge(buffer, (; α = zeros(eltype(ps.α), length(ps.α))))
end

"""Set every array in a gradient buffer to zero."""
function clear_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Add one worker-local gradient buffer into the batch buffer."""
function add_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

"""Scale one gradient buffer in place."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Clear batch and worker-local buffers before a new manager minibatch."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        clear_buffer!(worker_context(worker).base_context.buffers)
    end
    return manager
end

"""Flush worker-local buffers into one averaged baseline gradient."""
function flush_baseline_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        ctx = worker_context(worker).base_context
        add_buffer!(manager.state.batch_gradient, ctx.buffers)
        clear_buffer!(ctx.buffers)
    end
    total = manager.state.total_repeats[]
    total > 0 || throw(ArgumentError("cannot flush an XOR batch with zero repeats"))
    scale_buffer!(manager.state.batch_gradient, inv(FT(2) * FT(manager.config.β) * FT(total)))
    return manager.state.batch_gradient
end

"""Install updated shared parameters once after a manager batch."""
function sync_worker_params!(manager::M, ps::P) where {M<:Processes.ProcessManager,P}
    worker = first(Processes.workers(manager))
    IsingLearning.sync_params!(worker_context(worker).base_context.model, ps)
    return manager
end

"""Create the `ProcessManager` that owns all baseline training workers."""
function xor_baseline_manager(layer::L, graph::G, ps::P, config::C) where {L<:LayeredIsingGraphLayer,G,P,C<:XORBaselineConfig}
    optimizer_name = lowercase(config.optimizer)
    optimiser = optimizer_name == "adam" ? Optimisers.Adam(config.lr) :
        optimizer_name in ("descent", "sgd") ? Optimisers.Descent(config.lr) :
        throw(ArgumentError("unknown XOR optimizer `$(config.optimizer)`; use `adam` or `descent`"))
    state = XORBaselineManagerState(Ref(ps), parameter_buffer(ps), Ref(0), Optimisers.setup(optimiser, ps))
    recipe = (;
        makeworker = (idx, manager) -> averaged_worker(layer, worker_graph(graph, manager.state.params[], manager.config)),
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.base_context.x .= job.x
            ctx.base_context.y .= job.y
            ctx.repeats[] = job.repeats
            Processes.resetworker!(slot)
            return nothing
        end,
        flush! = manager -> flush_baseline_buffers!(manager),
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = XORBaselineJob{Vector{FT},Vector{FT}},
    )
end

"""Create manager jobs, defaulting to enough chunks to fill the workers.

Each job stores the number of random-start repeats to run. The worker-side
`AveragedBaselineContrastiveStep` executes those repeats inside one
`ProcessAlgorithm` step, so a 32-worker run gets 32 scheduled jobs without
turning each random init into its own manager job.
"""
function xor_baseline_jobs(config::C, x::X, y::Y) where {C<:XORBaselineConfig,X<:AbstractMatrix,Y<:AbstractMatrix}
    chunks_per_case = config.chunks_per_case > 0 ?
        max(1, min(config.repeats, config.chunks_per_case)) :
        max(1, min(config.repeats, cld(config.workers, size(x, 2))))
    jobs = XORBaselineJob{Vector{FT},Vector{FT}}[]
    for case_idx in axes(x, 2)
        base = config.repeats ÷ chunks_per_case
        extra = config.repeats % chunks_per_case
        for chunk in 1:chunks_per_case
            repeats = base + (chunk <= extra ? 1 : 0)
            repeats == 0 && continue
            push!(jobs, XORBaselineJob(case_idx, copy(view(x, :, case_idx)), copy(view(y, :, case_idx)), repeats))
        end
    end
    return jobs
end

"""Run one manager minibatch and synchronize updated parameters."""
function run_baseline_batch!(manager::M, jobs::J, config::C) where {M<:Processes.ProcessManager,J,C<:XORBaselineConfig}
    clear_manager_buffers!(manager)
    manager.state.total_repeats[] = sum(job.repeats for job in jobs)
    Processes.run!(manager, jobs, Processes.Dynamic())
    manager.state.opt_state, ps_new = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = ps_new
    sync_worker_params!(manager, ps_new)
    return ps_new
end

"""Append one row to a CSV metrics file."""
function append_metrics!(path::P, row) where {P<:AbstractString}
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        names = propertynames(row)
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Train the baseline and write compact shareable artifacts."""
function main()
    config = XORBaselineConfig()
    config.workers > 0 || throw(ArgumentError("ISING_XOR_BASELINE_WORKERS must be positive"))
    config.repeats > 0 || throw(ArgumentError("ISING_XOR_BASELINE_REPEATS must be positive"))
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    mkpath(config.outdir)

    graph = xor_graph(config)
    layer = xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 10), layer)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 11), layer)
    x, y, cases = xor_dataset(FT)
    jobs = xor_baseline_jobs(config, x, y)
    manager = xor_baseline_manager(layer, graph, ps, config)
    metrics_path = joinpath(config.outdir, "metrics.csv")

    try
        for epoch in 0:config.epochs
            if epoch == 0 || epoch % config.log_every == 0 || epoch == config.epochs
                metrics = evaluate_xor(layer, manager.state.params[], st, x, y)
                append_metrics!(
                    metrics_path,
                    (;
                        epoch,
                        mse = metrics.mse,
                        accuracy = metrics.accuracy,
                        outputs = repr(round.(metrics.outputs; digits = 4)),
                    ),
                )
                println("epoch=", epoch, " mse=", round(metrics.mse; digits = 6), " accuracy=", metrics.accuracy)
            end
            epoch == config.epochs && break
            run_baseline_batch!(manager, jobs, config)
        end

        open(joinpath(config.outdir, "final_params.bin"), "w") do io
            serialize(io, (; ps = manager.state.params[], config, cases))
        end
        open(joinpath(config.outdir, "README.md"), "w") do io
            println(io, "# XOR ProcessManager Baseline")
            println(io)
            println(io, "All-to-all continuous `4 -> $(config.hidden) -> 2` XOR baseline.")
            println(io, "Training uses `ProcessManager`, `LayerContrastiveStep`, worker-local input averaging, and one `FlushAtEnd()` merge per epoch.")
            println(io)
            println(io, "- epochs: `$(config.epochs)`")
            println(io, "- workers: `$(config.workers)`")
            println(io, "- repeats per XOR case: `$(config.repeats)`")
            println(io, "- chunks per XOR case: `$(config.chunks_per_case == 0 ? cld(config.workers, size(x, 2)) : config.chunks_per_case)`")
            println(io, "- jobs per epoch: `$(length(jobs))`")
            println(io, "- free/nudged steps: `$(config.free_steps)` / `$(config.nudged_steps)`")
            println(io, "- beta: `$(config.β)`")
            println(io, "- optimizer: `$(config.optimizer)`")
            println(io, "- optimizer learning rate: `$(config.lr)`")
            println(io, "- temperature: `$(config.temp)`")
            println(io, "- stepsize: `$(config.stepsize)`")
        end
    finally
        close(manager)
    end
    println("saved ", config.outdir)
    return config.outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
