using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "simple_langevin_xor", "simple_2_4_1_langevin.jl"))

using Logging

import IsingLearning.InteractiveIsing.Processes: Process, TaskData, Init, Override, NamedInput, NamedOverride,
    ProcessContext, normalize_process_algo, getregistry, resolve, get_target_name,
    getinputs, getoverrides, getlifetime, getalgo, taskdata, initcontext,
    processlist, remove_process!, RuntimeListeners, context, task, deletekeys

"""
    TimeAverageXorConfig

Configuration for relearning the physical `2 -> 4 -> 1` XOR graph while using
time-averaged output classification for validation.
"""
Base.@kwdef mutable struct TimeAverageXorConfig
    epochs::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_EPOCHS", "800"))
    log_every::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_LOG_EVERY", "50"))
    minit::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_MINIT", "4"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_EVAL_REPEATS", "8"))
    workers::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    free_relaxation::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_FREE", "80"))
    nudged_relaxation::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_NUDGED", "80"))
    β::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_BETA", "0.2"))
    lr::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_LR", "0.003"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_WEIGHT_DECAY", "1e-4"))
    grad_clip::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_GRAD_CLIP", "50"))
    temp::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_TEMP", "0.01"))
    eval_temp::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_EVAL_TEMP", get(ENV, "ISING_TIMEAVG_XOR_TEMP", "0.01")))
    stepsize::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_STEPSIZE", "0.2"))
    max_drift_fraction::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_MAX_DRIFT", "0.6"))
    weight_scale::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_WEIGHT_SCALE", "0.18"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_TIMEAVG_XOR_BIAS_SCALE", "0.02"))
    train_average_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_TRAIN_AVG", "20"))
    train_sample_every_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_TRAIN_EVERY", "1"))
    eval_burnin_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_EVAL_BURNIN", "8"))
    eval_average_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_EVAL_AVG", "50"))
    eval_sample_every_sweeps::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_EVAL_EVERY", "1"))
    weight_seed::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_WEIGHT_SEED", "31"))
    bias_seed::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_BIAS_SEED", "37"))
    base_seed::Int = parse(Int, get(ENV, "ISING_TIMEAVG_XOR_BASE_SEED", "84000"))
    outdir::String = get(
        ENV,
        "ISING_TIMEAVG_XOR_DIR",
        joinpath(@__DIR__, "runs", "simple_2_4_1_timeavg_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

CONFIG = isdefined(@__MODULE__, :CONFIG) ? CONFIG : TimeAverageXorConfig()

"""
    ManagedTimeAvgTrainer

Minimal local trainer state. The persistent `ProcessManager`s own their worker
processes and reuse worker contexts by calling the local `prepare!` recipe for
each job.
"""
mutable struct ManagedTimeAvgTrainer{Layer,Graph,Params,OptState,Optimiser}
    layer::Layer
    prototype_graph::Graph
    params::Params
    opt_state::OptState
    optimiser::Optimiser
    train_manager::Any
    eval_manager::Any
    current_batch_gradient::Any
    current_responses::Vector{FT}
    current_sample_outputs::Any
end

"""
    simple_config(config)

Translate the time-average experiment config to the existing simple XOR graph
and EqProp process configuration.
"""
function simple_config(config::TimeAverageXorConfig)
    return SimpleXorConfig(
        epochs = config.epochs,
        log_every = config.log_every,
        minit = config.minit,
        eval_repeats = config.eval_repeats,
        workers = config.workers,
        free_relaxation = config.free_relaxation,
        nudged_relaxation = config.nudged_relaxation,
        early_relaxation = min(20, config.free_relaxation),
        β = config.β,
        lr = config.lr,
        weight_decay = config.weight_decay,
        grad_clip = config.grad_clip,
        temp = config.temp,
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        weight_scale = config.weight_scale,
        bias_scale = config.bias_scale,
        init_mode = :random,
        weight_seed = config.weight_seed,
        bias_seed = config.bias_seed,
        base_seed = config.base_seed,
    )
end

"""Create the local trainer from a fresh `2 -> 4 -> 1` graph."""
function init_timeavg_trainer(config::TimeAverageXorConfig)
    sc = simple_config(config)
    graph = simple_graph(sc)
    layer = SimpleLayer(graph, sc)
    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    trainer = ManagedTimeAvgTrainer(
        layer,
        graph,
        params,
        opt_state,
        optimiser,
        nothing,
        nothing,
        nothing,
        FT[],
        nothing,
    )
    trainer.train_manager = Processes.ProcessManager(
        train_manager_recipe(layer, graph);
        nworkers = config.workers,
        config,
        state = trainer,
        flush_policy = Processes.FlushAtEnd(),
        poll_interval = 0.0,
        job_type = TrainJob,
    )
    trainer.eval_manager = Processes.ProcessManager(
        eval_manager_recipe(layer, graph);
        nworkers = config.workers,
        config,
        state = trainer,
        flush_policy = Processes.NoFlush(),
        poll_interval = 0.0,
        job_type = EvalJob,
    )
    return trainer
end

"""
    ReusableOutputAverager

Context-owned output averager for reusable validation workers. The algorithm
state is reset with `Processes.initcontext(context, :output_averager)`.
"""
struct ReusableOutputAverager <: Processes.ProcessAlgorithm
    output_idx::Int
    burnin_sweeps::Int
    average_sweeps::Int
end

Processes.init(::ReusableOutputAverager, context) =
    (; sum = zero(FT), sumsq = zero(FT), count = 0, seen_sweeps = 0)

function Processes.step!(averager::ReusableOutputAverager, context)
    seen = context.seen_sweeps + 1
    sum = context.sum
    sumsq = context.sumsq
    count = context.count
    if seen > averager.burnin_sweeps && count < averager.average_sweeps
        value = FT(II.state(context.model)[averager.output_idx])
        sum += value
        sumsq += value^2
        count += 1
    end
    return (; sum, sumsq, count, seen_sweeps = seen)
end

"""Mean of a reusable output-averager subcontext."""
reusable_average_mean(ctx) = ctx.count == 0 ? FT(NaN) : ctx.sum / ctx.count

"""Standard deviation of a reusable output-averager subcontext."""
function reusable_average_std(ctx)
    ctx.count <= 1 && return zero(FT)
    μ = reusable_average_mean(ctx)
    return sqrt(max(zero(FT), ctx.sumsq / ctx.count - μ^2))
end

"""
    StateAverager()

Accumulate the full graph state over time. The mean state is used as the
plus/minus EqProp state in `contrastive_gradient`, so this is output-state
averaging in the mathematically consistent sense: the output readout is averaged,
and the hidden/input coordinates needed for `J` and `b` gradients are averaged
over the same time window.
"""
struct StateAverager{Tag} <: Processes.ProcessAlgorithm end

Processes.init(::StateAverager, context) =
    (; sum = zeros(FT, length(II.state(context.model))), count = 0)

function Processes.step!(::StateAverager, context)
    sum = context.sum
    sum .+= II.state(context.model)
    return (; sum, count = context.count + 1)
end

"""Return a copy of the mean state accumulated by `StateAverager`."""
function averaged_state(ctx)
    ctx.count == 0 && error("StateAverager has no samples")
    return ctx.sum ./ FT(ctx.count)
end

"""One EqProp training trajectory property for `ProcessManager`."""
Base.@kwdef struct TrainJob
    sample_idx::Int
    init_idx::Int
    x::Vector{FT}
    y::Vector{FT}
    seed::Int
end

"""One time-averaged validation trajectory property for `ProcessManager`."""
Base.@kwdef struct EvalJob
    sample_idx::Int
    repeat_idx::Int
    x::Vector{FT}
    seed::Int
end

"""Build the list of stochastic EqProp trajectories for one epoch."""
function train_jobs(x, y, config::TimeAverageXorConfig, epoch::Integer)
    jobs = TrainJob[]
    for sample_idx in axes(x, 2), init_idx in 1:config.minit
        push!(jobs, TrainJob(
            sample_idx = sample_idx,
            init_idx = init_idx,
            x = copy(vec(view(x, :, sample_idx))),
            y = copy(vec(view(y, :, sample_idx))),
            seed = config.base_seed + 1_000_000 * epoch + 10_000 * sample_idx + init_idx,
        ))
    end
    return jobs
end

"""Build the list of time-averaged validation trajectories."""
function eval_jobs(x, config::TimeAverageXorConfig; seed_offset::Integer)
    jobs = EvalJob[]
    for sample_idx in axes(x, 2), repeat_idx in 1:config.eval_repeats
        push!(jobs, EvalJob(
            sample_idx = sample_idx,
            repeat_idx = repeat_idx,
            x = copy(vec(view(x, :, sample_idx))),
            seed = seed_offset + 10_000 * sample_idx + repeat_idx,
        ))
    end
    return jobs
end

"""Create one reusable training worker process."""
function template_train_worker(layer, prototype_graph, config::TimeAverageXorConfig, worker_idx::Integer)
    sc = simple_config(config)
    graph = managed_worker_graph(prototype_graph, config)
    return timeavg_training_worker_process(layer, graph, sc, config)
end

"""Build one nudged branch that samples full states after the normal burn-in."""
function timeavg_nudged_branch(graph, config::TimeAverageXorConfig, sc::SimpleXorConfig, beta_sign::Int)
    dynamics_algorithm = simple_dynamics(sc)
    state_averager = beta_sign > 0 ? StateAverager{:plus}() : StateAverager{:minus}()
    sample_interval = config.train_sample_every_sweeps * length(II.sampling_indices(graph.index_set))
    average_count = max(1, config.train_average_sweeps)
    beta_value = beta_sign > 0 ? sc.β : -sc.β
    sample = if beta_sign > 0
        Processes.@Routine begin
            @alias dynamics = dynamics_algorithm
            @repeat sample_interval dynamics()
            @alias plus_state_averager = state_averager
            plus_state_averager(model = dynamics.model)
        end
    else
        Processes.@Routine begin
            @alias dynamics = dynamics_algorithm
            @repeat sample_interval dynamics()
            @alias minus_state_averager = state_averager
            minus_state_averager(model = dynamics.model)
        end
    end
    branch = if beta_sign > 0
        Processes.@Routine begin
            @state x
            @state y
            @state equilibrium_state
            @alias dynamics = dynamics_algorithm
            IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
            IsingLearning.apply_input(dynamics.model, x)
            IsingLearning.apply_targets(dynamics.model, y)
            IsingLearning.set_clamping_beta!(dynamics.model, beta_value)
            @repeat sc.nudged_relaxation dynamics()
            @repeat average_count sample()
        end
    else
        Processes.@Routine begin
            @state x
            @state y
            @state equilibrium_state
            @alias dynamics = dynamics_algorithm
            IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
            IsingLearning.apply_input(dynamics.model, x)
            IsingLearning.apply_targets(dynamics.model, y)
            IsingLearning.set_clamping_beta!(dynamics.model, beta_value)
            @repeat sc.nudged_relaxation dynamics()
            @repeat average_count sample()
        end
    end
    return (; algorithm = branch, dynamics = branch.dynamics)
end

"""Create one worker whose EqProp plus/minus states are time-averaged states."""
function timeavg_training_worker_process(layer, graph, sc::SimpleXorConfig, config::TimeAverageXorConfig)
    forward = simple_forward(layer, sc; split = false).algorithm
    plus = timeavg_nudged_branch(graph, config, sc, +1)
    minus = timeavg_nudged_branch(graph, config, sc, -1)
    beta = layer.β
    final = Processes.@CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = plus.algorithm()
        @context c3 = minus.algorithm()
        IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
    end
    algo = Processes.resolve(final)
    buffers = IsingLearning.gradient_buffer(graph)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, 1),
            buffers = buffers,
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed);
        repeats = 1,
    )
end

"""Create one fresh worker graph with the current prototype parameters."""
function managed_worker_graph(prototype_graph, config::TimeAverageXorConfig)
    graph = IsingLearning._worker_graph(prototype_graph, IsingLearning.read_graph_params(prototype_graph))
    II.temp!(graph, config.temp)
    return graph
end

"""Constructor inputs for one copied training worker."""
function train_worker_inputs(graph, config::TimeAverageXorConfig)
    sc = simple_config(config)
    return (
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, 1),
            buffers = IsingLearning.gradient_buffer(graph),
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, sc.base_seed),
    )
end

"""Resolve process inputs against `template` and build a fresh worker-local context."""
function worker_context_from_inputs(template::Process, inputs)
    resolved = Processes._resolve_reinit_inputs(template, inputs...)
    return Processes.initcontext(Processes.taskdata(template), resolved...)
end

"""Build a copied training context with worker-local graph inputs."""
function train_worker_context(template::Process, prototype_graph, config::TimeAverageXorConfig)
    graph = managed_worker_graph(prototype_graph, config)
    return worker_context_from_inputs(template, train_worker_inputs(graph, config))
end

"""Create one reusable validation worker process with a context-owned averager."""
function template_eval_worker(layer, prototype_graph, config::TimeAverageXorConfig, worker_idx::Integer)
    sc = simple_config(config)
    graph = managed_worker_graph(prototype_graph, config)
    dynamics = simple_dynamics(sc)
    output_idx = only(II.layerrange(graph[end]))
    output_averager = ReusableOutputAverager(output_idx, config.eval_burnin_sweeps, config.eval_average_sweeps)
    sweep_steps = length(II.sampling_indices(graph.index_set))
    sample_interval = config.eval_sample_every_sweeps * sweep_steps
    total_sweeps = config.eval_burnin_sweeps + config.eval_average_sweeps * config.eval_sample_every_sweeps
    routine = Processes.@CompositeAlgorithm begin
        @alias dynamics = dynamics
        @every 1 dynamics()
        @alias output_averager = output_averager
        @every sample_interval output_averager(model = dynamics.model)
    end
    wrapped = Processes.@Routine begin
        @repeat (total_sweeps * sweep_steps) routine()
    end
    inputs = II._merge_graph_inputs(wrapped, graph, Processes.Init(dynamics, rng = Random.MersenneTwister(config.base_seed + 70_000 + worker_idx)))
    return Process(Processes.resolve(wrapped), inputs...; repeats = 1)
end

"""Constructor inputs for one copied validation worker."""
function eval_worker_inputs(graph, config::TimeAverageXorConfig, worker_idx::Integer)
    return (
        Init(:dynamics;
            isinggraph = graph,
            structure = graph,
            model = graph,
            rng = Random.MersenneTwister(config.base_seed + 70_000 + worker_idx),
        ),
    )
end

"""Build a copied validation context with worker-local graph inputs."""
function eval_worker_context(template::Process, prototype_graph, config::TimeAverageXorConfig, worker_idx::Integer)
    graph = managed_worker_graph(prototype_graph, config)
    return worker_context_from_inputs(template, eval_worker_inputs(graph, config, worker_idx))
end

"""Synchronize all graph models in a reusable worker context to current params."""
function sync_worker_params!(worker::Process, params)
    IsingLearning.sync_graph_params!(Processes.context(worker).dynamics.model, params)
    return worker
end

"""Synchronize every worker graph owned by a manager."""
function sync_manager_workers!(manager, params)
    for worker in Processes.workers(manager)
        Processes.isdone(worker) && close(worker)
        sync_worker_params!(worker, params)
    end
    return manager
end

"""Reset worker-local gradient buffers once before a managed training batch."""
function reset_manager_buffers!(manager)
    for worker in Processes.workers(manager)
        IsingLearning.zero_buffer!(Processes.context(worker)._state.buffers)
    end
    return manager
end

"""Prepare one reusable training worker for a new EqProp trajectory."""
function prepare_train_worker!(worker::Process, trainer::ManagedTimeAvgTrainer, config::TimeAverageXorConfig, job::TrainJob)
    Processes.isdone(worker) && close(worker)
    seed_worker!(worker, job.seed)
    IsingLearning._write_example!(worker, job.x, job.y)
    Processes.context(worker, Processes.initcontext(Processes.context(worker), :plus_state_averager))
    Processes.context(worker, Processes.initcontext(Processes.context(worker), :minus_state_averager))
    Processes.reset!(worker)
    return worker
end

"""Prepare one reusable validation worker for a new time-averaged readout."""
function prepare_eval_worker!(worker::Process, trainer::ManagedTimeAvgTrainer, config::TimeAverageXorConfig, job::EvalJob)
    Processes.isdone(worker) && close(worker)
    graph = Processes.context(worker).dynamics.model
    II.temp!(graph, config.eval_temp)
    Random.seed!(job.seed)
    simple_initstate!(graph, simple_config(config))
    IsingLearning.apply_input(graph, job.x)
    hasproperty(Processes.context(worker).dynamics, :rng) && Random.seed!(Processes.context(worker).dynamics.rng, job.seed + 1)
    Processes.context(worker, Processes.initcontext(Processes.context(worker), :output_averager))
    Processes.reset!(worker)
    return worker
end

"""Merge one managed worker result into a gradient buffer and response list."""
function collect_train_result!(batch_gradient, responses, result, beta)
    isnothing(result.error) || throw(result.error)
    ctx = result.context
    free_state = ctx._state.equilibrium_state
    plus_state = averaged_state(ctx.plus_state_averager)
    minus_state = averaged_state(ctx.minus_state_averager)
    response = (
        sqrt(sum(abs2, plus_state .- free_state) / FT(length(free_state))) +
        sqrt(sum(abs2, minus_state .- free_state) / FT(length(free_state)))
    ) / 2
    push!(responses, response)
    IsingLearning.contrastive_gradient(ctx.dynamics.model, plus_state, minus_state, beta, buffers = ctx._state.buffers)
    IsingLearning.add_buffer!(batch_gradient, ctx._state.buffers)
    return batch_gradient
end

"""Record trajectory diagnostics while leaving worker-local gradients local."""
function collect_train_response!(responses, result, beta)
    isnothing(result.error) || throw(result.error)
    ctx = result.context
    free_state = ctx._state.equilibrium_state
    plus_state = averaged_state(ctx.plus_state_averager)
    minus_state = averaged_state(ctx.minus_state_averager)
    response = (
        sqrt(sum(abs2, plus_state .- free_state) / FT(length(free_state))) +
        sqrt(sum(abs2, minus_state .- free_state) / FT(length(free_state)))
    ) / 2
    push!(responses, response)
    IsingLearning.contrastive_gradient(ctx.dynamics.model, plus_state, minus_state, beta, buffers = ctx._state.buffers)
    return responses
end

"""Flush all worker-local gradient buffers into the epoch batch gradient once."""
function flush_train_buffers!(manager)
    batch_gradient = manager.state.current_batch_gradient
    for worker in Processes.workers(manager)
        IsingLearning.add_buffer!(batch_gradient, Processes.context(worker)._state.buffers)
    end
    reset_manager_buffers!(manager)
    return batch_gradient
end

"""Recipe used by the persistent training `ProcessManager`."""
function train_manager_recipe(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> template_train_worker(layer, prototype_graph, manager.config, idx),
        makecontext = (idx, manager, template) -> train_worker_context(template, prototype_graph, manager.config),
        prepare! = (slot, job, manager) -> prepare_train_worker!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes.isdone(slot.worker),
        consume! = (slot, job, manager) -> collect_train_response!(
            manager.state.current_responses,
            (; context = Processes.context(slot.worker), error = nothing),
            layer.β,
        ),
        flush! = flush_train_buffers!,
    )
end

"""Recipe used by the persistent validation `ProcessManager`."""
function eval_manager_recipe(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> template_eval_worker(layer, prototype_graph, manager.config, idx),
        makecontext = (idx, manager, template) -> eval_worker_context(template, prototype_graph, manager.config, idx),
        prepare! = (slot, job, manager) -> prepare_eval_worker!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes.isdone(slot.worker),
        consume! = (slot, job, manager) -> begin
            ctx = Processes.context(slot.worker).output_averager
            push!(manager.state.current_sample_outputs[job.sample_idx], (;
                mean = reusable_average_mean(ctx),
                std = reusable_average_std(ctx),
            ))
        end,
    )
end

"""Train one epoch using reusable `ProcessManager` worker slots."""
function train_epoch_managed!(trainer::ManagedTimeAvgTrainer, x, y, batch_gradient, epoch::Integer, config::TimeAverageXorConfig)
    IsingLearning.zero_buffer!(batch_gradient)
    reset_manager_buffers!(trainer.train_manager)
    trainer.current_batch_gradient = batch_gradient
    empty!(trainer.current_responses)
    jobs = train_jobs(x, y, config, epoch)
    Processes.run!(trainer.train_manager, jobs)

    ntraj = length(jobs)
    IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
    config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
    config.grad_clip > 0 && clamp!(batch_gradient.w, -config.grad_clip, config.grad_clip)
    config.grad_clip > 0 && clamp!(batch_gradient.b, -config.grad_clip, config.grad_clip)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    sync_manager_workers!(trainer.train_manager, trainer.params)
    return (; grad_norm, response_norm = isempty(trainer.current_responses) ? zero(FT) : mean(trainer.current_responses))
end

"""Evaluate time-averaged outputs using reusable `ProcessManager` worker slots."""
function evaluate_timeavg!(trainer::ManagedTimeAvgTrainer, x, y, config::TimeAverageXorConfig; seed_offset::Integer)
    sync_manager_workers!(trainer.eval_manager, trainer.params)
    jobs = eval_jobs(x, config; seed_offset)
    trainer.current_sample_outputs = [NamedTuple{(:mean, :std),Tuple{FT,FT}}[] for _ in axes(x, 2)]
    Processes.run!(trainer.eval_manager, jobs)

    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        sample_means = [output.mean for output in trainer.current_sample_outputs[sample_idx]]
        means[sample_idx] = mean(sample_means)
        stds[sample_idx] = std(sample_means)
    end
    targets = vec(y)
    return (;
        mse = mean(abs2, means .- targets),
        acc = mean(sign.(means) .== sign.(targets)),
        margin = minimum(abs.(means)),
        means,
        stds,
    )
end

"""Create one CSV/plot row for the managed time-average experiment."""
function timeavg_metric_row(epoch, metrics, grad)
    row = Dict{String,Any}(
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.acc,
        "margin" => metrics.margin,
        "grad_norm" => grad.grad_norm,
        "response_norm" => grad.response_norm,
    )
    for i in eachindex(metrics.means)
        row["mean_$i"] = metrics.means[i]
        row["std_$i"] = metrics.stds[i]
    end
    return row
end

"""Plot managed time-average learning diagnostics."""
function plot_timeavg_learning(path, rows)
    fig = Figure(size = (1100, 800))
    ax1 = Axis(fig[1, 1], title = "2->4->1 time-averaged XOR MSE", xlabel = "epoch", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy")
    ax3 = Axis(fig[2, 1], title = "output margin", xlabel = "epoch", ylabel = "min |mean output|")
    ax4 = Axis(fig[2, 2], title = "gradient norm", xlabel = "epoch", ylabel = "||grad||")
    epochs = [row["epoch"] for row in rows]
    lines!(ax1, epochs, [row["mse"] for row in rows])
    lines!(ax2, epochs, [row["accuracy"] for row in rows])
    lines!(ax3, epochs, [row["margin"] for row in rows])
    lines!(ax4, epochs, [row["grad_norm"] for row in rows])
    save(path, fig)
    return path
end

"""Remove construction-time weight generators before saving a trained graph."""
function strip_weight_generators!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

"""Write a short run README with the exact training/evaluation settings."""
function write_timeavg_readme(path, config::TimeAverageXorConfig, best, csv_path, png_path)
    open(path, "w") do io
        println(io, "# Simple 2->4->1 Time-Averaged XOR Learning")
        println(io)
        println(io, "This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not")
        println(io, "reuse a learned graph. EqProp worker trajectories and validation")
        println(io, "trajectories are launched and synchronized with `ProcessManager`.")
        println(io)
        println(io, "Validation classifies by averaging the scalar output spin after a")
        println(io, "burn-in period:")
        println(io)
        println(io, "- burn-in full sweeps: `$(config.eval_burnin_sweeps)`")
        println(io, "- averaged samples: `$(config.eval_average_sweeps)`")
        println(io, "- sampling interval: one sample every `$(config.eval_sample_every_sweeps)` full sweep(s)")
        println(io)
        println(io, "Training settings:")
        println(io)
        println(io, "- epochs: `$(config.epochs)`")
        println(io, "- free/nudged relaxation: `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- temperature: `$(config.temp)`")
        println(io, "- validation temperature: `$(config.eval_temp)`")
        println(io, "- stepsize: `$(config.stepsize)`")
        println(io, "- nudged state-average samples: `$(config.train_average_sweeps)` every `$(config.train_sample_every_sweeps)` full sweep(s)")
        println(io, "- Minit/eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- workers: `$(config.workers)`")
        println(io)
        println(io, "Best logged result:")
        println(io)
        println(io, "- epoch: `$(best.epoch)`")
        println(io, "- MSE: `$(best.mse)`")
        println(io, "- accuracy: `$(best.acc)`")
        println(io, "- means: `$(best.means)`")
        println(io)
        println(io, "CSV: `$(basename(csv_path))`")
        println(io, "Plot: `$(basename(png_path))`")
    end
    return path
end

"""Run the managed relearning experiment."""
function run_timeavg_learning(; config::TimeAverageXorConfig = CONFIG)
    disable_logging(Logging.Warn)
    mkpath(config.outdir)
    trainer = init_timeavg_trainer(config)
    x, y = simple_dataset()
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    metrics = evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    push!(rows, timeavg_metric_row(0, metrics, zero_grad))
    best = (epoch = 0, mse = metrics.mse, acc = metrics.acc, means = copy(metrics.means), params = deepcopy(trainer.params))
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        grad = train_epoch_managed!(trainer, x, y, batch_gradient, epoch, config)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_timeavg!(trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
            push!(rows, timeavg_metric_row(epoch, metrics, grad))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (epoch = epoch, mse = metrics.mse, acc = metrics.acc, means = copy(metrics.means), params = deepcopy(trainer.params))
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad.grad_norm, digits = 4), " response=", round(grad.response_norm, digits = 4),
                " means=", round.(metrics.means, digits = 3))
        end
    end

    csv_path = write_csv(joinpath(config.outdir, "timeavg_learning_metrics.csv"), rows)
    png_path = plot_timeavg_learning(joinpath(config.outdir, "timeavg_learning_progress.png"), rows)

    trainer.params = best.params
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    best_graph = strip_weight_generators!(deepcopy(trainer.prototype_graph))
    graph_path = II.save_isinggraph(joinpath(config.outdir, "best_graph.jld2"), best_graph)
    md_path = write_timeavg_readme(joinpath(config.outdir, "README.md"), config, best, csv_path, png_path)

    println("Saved metrics: ", csv_path)
    println("Saved plot: ", png_path)
    println("Saved graph: ", graph_path)
    println("Saved docs: ", md_path)
    return (; config, rows, best, csv_path, png_path, graph_path, md_path)
end

"""Script entrypoint for the managed time-average experiment."""
main(; config::TimeAverageXorConfig = CONFIG) = run_timeavg_learning(config = config)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
