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

const XOR_CASES = ((false, false), (false, true), (true, false), (true, true))

Base.@kwdef struct XORInputAverageConfig
    epochs::Int = parse(Int, get(ENV, "ISING_XOR_PM_EPOCHS", "1600"))
    workers::Int = parse(Int, get(ENV, "ISING_XOR_PM_WORKERS", "32"))
    log_every::Int = parse(Int, get(ENV, "ISING_XOR_PM_LOG_EVERY", "50"))
    repeats_per_case::Int = parse(Int, get(ENV, "ISING_XOR_PM_REPEATS", "32"))
    chunks_per_case::Int = parse(Int, get(ENV, "ISING_XOR_PM_CHUNKS_PER_CASE", "0"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_XOR_PM_EVAL_REPEATS", "16"))
    readout_replicas::Int = parse(Int, get(ENV, "ISING_XOR_PM_READOUT_REPLICAS", "4"))
    free_steps::Int = parse(Int, get(ENV, "ISING_XOR_PM_FREE_STEPS", "300"))
    nudged_steps::Int = parse(Int, get(ENV, "ISING_XOR_PM_NUDGED_STEPS", "300"))
    β::FT = FT(2.0)
    lr::FT = FT(0.002)
    optimizer::String = lowercase(get(ENV, "ISING_XOR_PM_OPTIMIZER", "adam"))
    temp::FT = FT(0.005)
    stepsize::FT = FT(0.4)
    max_drift_fraction::FT = FT(1.0)
    weight_scale::FT = FT(0.20)
    bias_scale::FT = FT(0.05)
    seed::Int = 13
    outdir::String = get(
        ENV,
        "ISING_XOR_PM_OUTDIR",
        joinpath(@__DIR__, "runs", "xor_manager_input_averaging_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

struct XORAverageJob{X<:AbstractVector,Y<:AbstractVector}
    case_idx::Int
    x::X
    y::Y
    repeats::Int
end

struct AveragedContrastiveStep{S} <: Processes.ProcessAlgorithm
    base::S
end

mutable struct XORManagerState{P,B,O}
    params::Base.RefValue{P}
    batch_gradient::B
    total_repeats::Base.RefValue{Int}
    opt_state::O
end

"""Return bipolar two-spin XOR inputs and replicated two-class targets."""
function xor_dataset(config::C, ::Type{T} = FT) where {C<:XORInputAverageConfig,T<:AbstractFloat}
    x = Matrix{T}(undef, 2, length(XOR_CASES))
    y = Matrix{T}(undef, 2 * config.readout_replicas, length(XOR_CASES))
    for (col, (a, b)) in enumerate(XOR_CASES)
        x[:, col] .= (a ? one(T) : -one(T), b ? one(T) : -one(T))
        target = xor(a, b) ? (-one(T), one(T)) : (one(T), -one(T))
        y[:, col] .= repeat(collect(target), inner = config.readout_replicas)
    end
    return x, y
end

"""Build the clean all-to-all `2 -> 4 -> 2xR` XOR graph."""
function xor_graph(config::C) where {C<:XORInputAverageConfig}
    rng_w = Random.MersenneTwister(config.seed)
    rng_b = Random.MersenneTwister(config.seed + 1)

    input = II.Layer(2, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 1, 0); periodic = false)
    hidden = II.Layer(4, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 2, 0); periodic = false)
    output = II.Layer(2 * config.readout_replicas, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 3, 0); periodic = false)

    weights = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> config.weight_scale * randn(rng_w, FT))
    bias = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))

    graph = II.IsingGraph(
        input,
        deepcopy(weights),
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

"""Create the Lux graph layer and reuse its standard contrastive process algorithm."""
function xor_layer(graph::G, config::C) where {G,C<:XORInputAverageConfig}
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        adjusted = false,
        order = :random,
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

"""Initialize the repeat wrapper around `LayerContrastiveStep`."""
function Processes.init(step::AveragedContrastiveStep{S}, context) where {S}
    base_context = Processes.init(step.base, context)
    repeats = get(context, :repeats, Ref(1))
    repeats_ref = repeats isa Base.RefValue ? repeats : Ref(Int(repeats))
    return (; base_context, repeats = repeats_ref)
end

"""Run several input-averaging repeats inside one managed worker execution."""
function Processes.step!(step::AveragedContrastiveStep{S}, context) where {S}
    @inbounds for _ in 1:context.repeats[]
        Processes.step!(step.base, context.base_context)
    end
    return nothing
end

"""Keep the averaged contrastive context reusable across manager jobs."""
function Processes.cleanup(step::AveragedContrastiveStep{S}, context) where {S}
    return nothing
end

"""Return the mutable training context stored in one manager worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Create a worker graph with fresh state and shared static parameter arrays."""
function worker_graph(prototype::G, ps::P, config::C) where {G,P,C<:XORInputAverageConfig}
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

"""Build one reusable process worker for the averaged contrastive step."""
function averaged_worker(layer::L, graph::G) where {L<:LayeredIsingGraphLayer,G}
    step = AveragedContrastiveStep(LayerContrastiveStep(layer))
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

"""Allocate a gradient buffer matching the current parameter tree."""
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

"""Clear batch and worker buffers before a new input-averaged minibatch."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        clear_buffer!(worker_context(worker).base_context.buffers)
    end
    return manager
end

"""Flush worker-local contrastive buffers into one averaged batch gradient."""
function flush_xor_buffers!(manager::M) where {M<:Processes.ProcessManager}
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

"""Create the manager that owns and reuses all XOR contrastive workers."""
function xor_manager(layer::L, graph::G, ps::P, config::C) where {L<:LayeredIsingGraphLayer,G,P,C<:XORInputAverageConfig}
    optimizer_name = lowercase(config.optimizer)
    optimiser = optimizer_name == "adam" ? Optimisers.Adam(config.lr) :
        optimizer_name in ("descent", "sgd") ? Optimisers.Descent(config.lr) :
        throw(ArgumentError("unknown XOR optimizer `$(config.optimizer)`; use `adam` or `descent`"))
    state = XORManagerState(Ref(ps), parameter_buffer(ps), Ref(0), Optimisers.setup(optimiser, ps))
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
        flush! = manager -> flush_xor_buffers!(manager),
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = XORAverageJob{Vector{FT},Vector{FT}},
    )
end

"""Create manager jobs, defaulting to enough chunks to fill the workers.

Each job stores the number of input-averaging repeats to run. The worker-side
`AveragedContrastiveStep` executes those repeats inside one `ProcessAlgorithm`
step, so a 32-worker run gets 32 scheduled jobs without turning each random
init into its own manager job.
"""
function xor_average_jobs(config::C, x::X, y::Y) where {C<:XORInputAverageConfig,X<:AbstractMatrix,Y<:AbstractMatrix}
    chunks_per_case = config.chunks_per_case > 0 ?
        max(1, min(config.repeats_per_case, config.chunks_per_case)) :
        max(1, min(config.repeats_per_case, cld(config.workers, length(XOR_CASES))))
    jobs = XORAverageJob{Vector{FT},Vector{FT}}[]
    for case_idx in axes(x, 2)
        base = config.repeats_per_case ÷ chunks_per_case
        extra = config.repeats_per_case % chunks_per_case
        for chunk in 1:chunks_per_case
            repeats = base + (chunk <= extra ? 1 : 0)
            repeats == 0 && continue
            push!(jobs, XORAverageJob(case_idx, copy(view(x, :, case_idx)), copy(view(y, :, case_idx)), repeats))
        end
    end
    return jobs
end

"""Run one averaged XOR batch and update all persistent worker graphs."""
function run_input_averaged_batch!(manager::M, jobs::J, config::C) where {M<:Processes.ProcessManager,J,C<:XORInputAverageConfig}
    clear_manager_buffers!(manager)
    manager.state.total_repeats[] = sum(job.repeats for job in jobs)
    Processes.run!(manager, jobs, Processes.Dynamic())
    manager.state.opt_state, ps_new = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = ps_new
    sync_worker_params!(manager, ps_new)
    return ps_new
end

"""Convert replicated physical output spins into two logical class scores."""
function class_scores(output::O, replicas::Integer) where {O<:AbstractVector}
    readouts = reshape(output, Int(replicas), 2)
    return vec(mean(readouts; dims = 1))
end

"""Evaluate averaged free-phase classification on the four XOR cases."""
function evaluate_xor(layer::L, ps::P, st::S, x::X, y::Y, config::C) where {L<:LayeredIsingGraphLayer,P,S,X<:AbstractMatrix,Y<:AbstractMatrix,C<:XORInputAverageConfig}
    outputs = zeros(FT, size(y, 1), size(y, 2))
    scores = zeros(FT, 2, size(y, 2))

    for _ in 1:config.eval_repeats
        for case_idx in axes(x, 2)
            out, st = layer(view(x, :, case_idx), ps, st)
            outputs[:, case_idx] .+= out
        end
    end
    outputs ./= FT(config.eval_repeats)

    for case_idx in axes(outputs, 2)
        scores[:, case_idx] .= class_scores(view(outputs, :, case_idx), config.readout_replicas)
    end
    predictions = vec(getindex.(argmax(scores; dims = 1), 1))
    targets = [xor(a, b) ? 2 : 1 for (a, b) in XOR_CASES]
    logical_targets = similar(scores)
    for (case_idx, target) in enumerate(targets)
        logical_targets[:, case_idx] .= target == 2 ? (-one(FT), one(FT)) : (one(FT), -one(FT))
    end
    return (;
        mse = mean(abs2, outputs .- y),
        score_mse = mean(abs2, scores .- logical_targets),
        accuracy = mean(predictions .== targets),
        outputs = copy(outputs),
        scores = copy(scores),
        predictions,
    )
end

"""Append one compact metrics row to a CSV file."""
function append_metrics!(path::P, row) where {P<:AbstractString}
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        names = propertynames(row)
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Write a small local note explaining the run and manager structure."""
function write_run_note!(path::P, config::C, jobs) where {P<:AbstractString,C<:XORInputAverageConfig}
    open(path, "w") do io
        println(io, "# XOR Manager Input Averaging")
        println(io)
        println(io, "Clean `2 -> 4 -> 2x$(config.readout_replicas)` XOR demonstrator.")
        println(io)
        println(io, "The worker algorithm is `AveragedContrastiveStep(LayerContrastiveStep(layer))`.")
        println(io, "Each manager job runs several repeats inside one already-started worker execution,")
        println(io, "then `FlushAtEnd()` merges all worker-local contrastive buffers once.")
        println(io)
        println(io, "- epochs: `$(config.epochs)`")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- repeats per XOR case: `$(config.repeats_per_case)`")
        println(io, "- chunks per XOR case: `$(config.chunks_per_case == 0 ? cld(config.workers, length(XOR_CASES)) : config.chunks_per_case)`")
        println(io, "- jobs per epoch: `$(length(jobs))`")
        println(io, "- free/nudged steps: `$(config.free_steps)` / `$(config.nudged_steps)`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- optimizer: `$(config.optimizer)`")
        println(io, "- optimizer learning rate: `$(config.lr)`")
        println(io, "- temperature: `$(config.temp)`")
        println(io, "- LocalLangevin stepsize: `$(config.stepsize)`")
    end
    return path
end

"""Train the clean ProcessManager-backed XOR demonstrator."""
function main()
    config = XORInputAverageConfig()
    config.workers > 0 || throw(ArgumentError("ISING_XOR_PM_WORKERS must be positive"))
    config.repeats_per_case > 0 || throw(ArgumentError("ISING_XOR_PM_REPEATS must be positive"))
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    config.readout_replicas > 0 || throw(ArgumentError("ISING_XOR_PM_READOUT_REPLICAS must be positive"))

    mkpath(config.outdir)
    graph = xor_graph(config)
    layer = xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 10), layer)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 11), layer)
    x, y = xor_dataset(config, FT)
    jobs = xor_average_jobs(config, x, y)
    manager = xor_manager(layer, graph, ps, config)

    metrics_path = joinpath(config.outdir, "metrics.csv")
    write_run_note!(joinpath(config.outdir, "README.md"), config, jobs)

    try
        for epoch in 0:config.epochs
            if epoch == 0 || epoch % config.log_every == 0 || epoch == config.epochs
                metrics = evaluate_xor(layer, manager.state.params[], st, x, y, config)
                append_metrics!(
                    metrics_path,
                    (;
                        epoch,
                        mse = metrics.mse,
                        accuracy = metrics.accuracy,
                        predictions = join(metrics.predictions, "|"),
                        scores = join(round.(vec(metrics.scores); digits = 4), "|"),
                    ),
                )
                println(
                    "epoch=", epoch,
                    " mse=", round(metrics.mse; digits = 6),
                    " acc=", round(metrics.accuracy; digits = 3),
                    " pred=", metrics.predictions,
                )
            end
            epoch == config.epochs && break
            run_input_averaged_batch!(manager, jobs, config)
        end

        open(joinpath(config.outdir, "final_params.bin"), "w") do io
            serialize(io, (; ps = manager.state.params[], config, cases = XOR_CASES))
        end
        println("saved ", config.outdir)
        return config.outdir
    finally
        close(manager)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
