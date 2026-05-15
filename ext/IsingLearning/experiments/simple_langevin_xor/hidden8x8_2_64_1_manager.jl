using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "hidden8x8_2_64_1.jl"))

using IsingLearning.InteractiveIsing.Processes

const Processes8x8 = IsingLearning.InteractiveIsing.Processes

"""Number of persistent worker slots used by the manager-backed experiment."""
manager_workers_8x8() = parse(Int, get(ENV, "ISING_8X8_THREADS", string(max(1, min(Threads.nthreads(), 8)))))

"""One managed EqProp trajectory for a specific XOR sample and random start."""
Base.@kwdef struct Hidden8x8TrainJob
    sample_idx::Int
    init_idx::Int
    x::Vector{FT}
    y::Vector{FT}
    seed::Int
end

"""One managed validation trajectory for a specific XOR sample and random start."""
Base.@kwdef struct Hidden8x8EvalJob
    sample_idx::Int
    repeat_idx::Int
    x::Vector{FT}
    seed::Int
end

"""
    ManagedHidden8x8Trainer

Manager-backed trainer for the same `2 -> 8x8 -> 1` scalar XOR recipe used by
`hidden8x8_2_64_1.jl`. The managers own persistent worker processes; worker
contexts are reused across epochs and synchronized after each parameter update.
"""
mutable struct ManagedHidden8x8Trainer{Layer,Graph,Params,OptState,Optimiser}
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

"""Create a worker graph with current prototype parameters and experiment temperature."""
function managed_worker_graph_8x8(prototype_graph, params, config::Hidden8x8Config)
    return IsingLearning._worker_graph(prototype_graph, params)
end

"""Create one reusable training worker for the manager."""
function make_train_worker_8x8(layer, prototype_graph, params, config::Hidden8x8Config, idx::Integer)
    graph = managed_worker_graph_8x8(prototype_graph, params, config)
    worker = IsingLearning._worker_process(layer, graph)
    Random.seed!(Processes.context(worker).dynamics.rng, config.base_seed + idx)
    return worker
end

"""Create one reusable validation worker for the manager."""
function make_eval_worker_8x8(layer, prototype_graph, params, config::Hidden8x8Config, idx::Integer)
    graph = managed_worker_graph_8x8(prototype_graph, params, config)
    worker = IsingLearning._validation_process(layer, graph)
    Random.seed!(Processes.context(worker).dynamics.rng, config.base_seed + 50_000 + idx)
    return worker
end

"""Seed all RNGs visible from one managed worker context."""
function seed_managed_worker_8x8!(worker, seed::Integer; seed_global::Bool = true)
    seed_global && Random.seed!(seed)
    Random.seed!(Processes.context(worker).dynamics.rng, seed + 10_000)
    return worker
end

"""Synchronize one managed worker graph from the trainer parameter vector."""
function sync_managed_worker_8x8!(worker, params)
    IsingLearning.sync_graph_params!(Processes.context(worker).dynamics.model, params)
    return worker
end

"""Synchronize all workers owned by a manager after a parameter update."""
function sync_manager_workers_8x8!(manager, params)
    for worker in Processes8x8.workers(manager)
        Processes8x8.isdone(worker) && close(worker)
        sync_managed_worker_8x8!(worker, params)
    end
    return manager
end

"""Zero all worker-local gradient buffers before one managed training epoch."""
function reset_manager_buffers_8x8!(manager)
    for worker in Processes8x8.workers(manager)
        IsingLearning.zero_buffer!(Processes.context(worker)._state.buffers)
    end
    return manager
end

"""Prepare one managed training worker for a fresh EqProp trajectory."""
function prepare_train_worker_8x8!(worker, trainer::ManagedHidden8x8Trainer, config::Hidden8x8Config, job::Hidden8x8TrainJob)
    Processes8x8.isdone(worker) && close(worker)
    IsingLearning._write_example!(worker, job.x, job.y)
    Processes8x8.reset!(worker)
    return worker
end

"""Prepare one managed validation worker for a fresh free relaxation."""
function prepare_eval_worker_8x8!(worker, trainer::ManagedHidden8x8Trainer, config::Hidden8x8Config, job::Hidden8x8EvalJob)
    Processes8x8.isdone(worker) && close(worker)
    seed_managed_worker_8x8!(worker, job.seed)
    IsingLearning._write_input!(worker, job.x)
    Processes8x8.reset!(worker)
    return worker
end

"""Record the response norm of one completed managed training trajectory."""
function collect_train_response_8x8!(responses, worker)
    free_state = Processes.context(worker)._state.equilibrium_state
    plus_state = Processes.context(worker).plus_capture.captured
    minus_state = Processes.context(worker).minus_capture.captured
    response = (
        sqrt(sum(abs2, plus_state .- free_state) / FT(length(free_state))) +
        sqrt(sum(abs2, minus_state .- free_state) / FT(length(free_state)))
    ) / 2
    push!(responses, response)
    return responses
end

"""Flush worker-local accumulated gradient buffers into one epoch buffer."""
function flush_train_buffers_8x8!(manager)
    batch_gradient = manager.state.current_batch_gradient
    for worker in Processes8x8.workers(manager)
        IsingLearning.add_buffer!(batch_gradient, Processes.context(worker)._state.buffers)
    end
    reset_manager_buffers_8x8!(manager)
    return batch_gradient
end

"""Build all managed EqProp jobs for one epoch."""
function train_jobs_8x8(x, y, config::Hidden8x8Config, epoch::Integer)
    jobs = Hidden8x8TrainJob[]
    for sample_idx in axes(x, 2), init_idx in 1:config.minit
        push!(jobs, Hidden8x8TrainJob(
            sample_idx = sample_idx,
            init_idx = init_idx,
            x = copy(vec(view(x, :, sample_idx))),
            y = copy(vec(view(y, :, sample_idx))),
            seed = config.base_seed + 1_000_000 * epoch + 10_000 * sample_idx + init_idx,
        ))
    end
    return jobs
end

"""Build all managed validation jobs for one logged evaluation."""
function eval_jobs_8x8(x, config::Hidden8x8Config; seed_offset::Integer)
    jobs = Hidden8x8EvalJob[]
    for sample_idx in axes(x, 2), repeat_idx in 1:config.eval_repeats
        push!(jobs, Hidden8x8EvalJob(
            sample_idx = sample_idx,
            repeat_idx = repeat_idx,
            x = copy(vec(view(x, :, sample_idx))),
            seed = seed_offset + 10_000 * sample_idx + repeat_idx,
        ))
    end
    return jobs
end

"""Recipe used by the persistent training manager."""
function train_manager_recipe_8x8(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> make_train_worker_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        prepare! = (slot, job, manager) -> prepare_train_worker_8x8!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes8x8.isdone(slot.worker),
        consume! = (slot, job, manager) -> collect_train_response_8x8!(manager.state.current_responses, slot.worker),
        flush! = flush_train_buffers_8x8!,
    )
end

"""Recipe used by the persistent validation manager."""
function eval_manager_recipe_8x8(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> make_eval_worker_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        prepare! = (slot, job, manager) -> prepare_eval_worker_8x8!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes8x8.isdone(slot.worker),
        consume! = (slot, job, manager) -> begin
            output = only(II.state(Processes.context(slot.worker).dynamics.model[end]))
            push!(manager.state.current_sample_outputs[job.sample_idx], output)
        end,
    )
end

"""Initialize the manager-backed trainer for the `2 -> 8x8 -> 1` recipe."""
function managed_trainer_8x8(config::Hidden8x8Config; workers::Integer = manager_workers_8x8())
    graph = graph_8x8(config)
    layer = layer_8x8(graph, config)
    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    trainer = ManagedHidden8x8Trainer(
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
    trainer.train_manager = Processes8x8.ProcessManager(
        train_manager_recipe_8x8(layer, graph);
        nworkers = workers,
        config,
        state = trainer,
        flush_policy = Processes8x8.FlushAtEnd(),
        poll_interval = 0.0,
        job_type = Hidden8x8TrainJob,
    )
    trainer.eval_manager = Processes8x8.ProcessManager(
        eval_manager_recipe_8x8(layer, graph);
        nworkers = 1,
        config,
        state = trainer,
        flush_policy = Processes8x8.NoFlush(),
        poll_interval = 0.0,
        job_type = Hidden8x8EvalJob,
    )
    return trainer
end

"""Train one epoch through the persistent manager and apply Adam."""
function train_epoch_managed_8x8!(trainer::ManagedHidden8x8Trainer, x, y, batch_gradient, epoch::Integer, config::Hidden8x8Config)
    sync_manager_workers_8x8!(trainer.train_manager, trainer.params)
    IsingLearning.zero_buffer!(batch_gradient)
    reset_manager_buffers_8x8!(trainer.train_manager)
    trainer.current_batch_gradient = batch_gradient
    empty!(trainer.current_responses)
    jobs = train_jobs_8x8(x, y, config, epoch)
    Processes8x8.run!(trainer.train_manager, jobs)

    ntraj = length(jobs)
    IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
    config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    sync_manager_workers_8x8!(trainer.train_manager, trainer.params)
    sync_manager_workers_8x8!(trainer.eval_manager, trainer.params)
    return (; grad_norm, response_norm = isempty(trainer.current_responses) ? zero(FT) : mean(trainer.current_responses))
end

"""Evaluate repeated free-relaxation scalar outputs through the validation manager."""
function evaluate_managed_8x8!(trainer::ManagedHidden8x8Trainer, x, y, config::Hidden8x8Config; seed_offset::Integer)
    sync_manager_workers_8x8!(trainer.eval_manager, trainer.params)
    trainer.current_sample_outputs = [FT[] for _ in axes(x, 2)]
    Processes8x8.run!(trainer.eval_manager, eval_jobs_8x8(x, config; seed_offset))

    means = zeros(FT, size(x, 2))
    stds = zeros(FT, size(x, 2))
    for sample_idx in axes(x, 2)
        samples = trainer.current_sample_outputs[sample_idx]
        means[sample_idx] = mean(samples)
        stds[sample_idx] = std(samples)
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

"""Close all processes owned by the manager-backed trainer."""
function close_managed_trainer_8x8!(trainer::ManagedHidden8x8Trainer)
    !isnothing(trainer.train_manager) && close(trainer.train_manager)
    !isnothing(trainer.eval_manager) && close(trainer.eval_manager)
    return trainer
end

"""Run the manager-backed `2 -> 8x8 -> 1` XOR experiment."""
function main()
    config = Hidden8x8Config()
    workers = manager_workers_8x8()
    outdir = get(
        ENV,
        "ISING_8X8_DIR",
        joinpath(@__DIR__, "runs", "hidden8x8_2_64_1_manager_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)

    trainer = managed_trainer_8x8(config; workers)
    x, y = xor_dataset_8x8(config)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_managed_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_row_8x8!(rows, 0, metrics, zero(FT))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        grad = train_epoch_managed_8x8!(trainer, x, y, batch_gradient, epoch, config)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_managed_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            push_row_8x8!(rows, epoch, metrics, grad.grad_norm)
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad.grad_norm, digits = 4), " means=", round.(metrics.means, digits = 3))
        end
    end

    trainer.params = best_params
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    csv_path = write_csv_8x8(joinpath(outdir, "metrics.csv"), rows)
    png_path = plot_8x8(joinpath(outdir, "progress.png"), rows)
    graph_path = II.save_isinggraph(
        joinpath(outdir, "hidden8x8_2_64_1_manager_best_graph.jld2"),
        strip_weight_generators_8x8!(deepcopy(trainer.prototype_graph)),
    )
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Manager-Backed 2 -> 8x8 -> 1 Scalar XOR")
        println(io)
        println(io, "This is the same recipe as `hidden8x8_2_64_1.jl`, but EqProp")
        println(io, "training trajectories and validation trajectories are dispatched")
        println(io, "through persistent `ProcessManager` worker slots.")
        println(io)
        println(io, "- workers: `$(workers)`")
        println(io, "- epochs/log_every: `$(config.epochs)` / `$(config.log_every)`")
        println(io, "- free/nudged: `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- Minit/eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- β/lr/T/stepsize: `$(config.β)` / `$(config.lr)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- hidden local NN: `$(config.hidden_nn)`")
        println(io, "- scales input-hidden/local/hidden-output/bias: `$(config.input_hidden_scale)` / `$(config.hidden_local_scale)` / `$(config.hidden_output_scale)` / `$(config.bias_scale)`")
        println(io)
        println(io, "Best logged: epoch `$(best.epoch)`, MSE `$(round(best.mse, digits = 6))`, accuracy `$(best.acc)`.")
        println(io)
        println(io, "CSV: `metrics.csv`")
        println(io, "Plot: `progress.png`")
        println(io, "Best graph: `hidden8x8_2_64_1_manager_best_graph.jld2`")
    end
    close_managed_trainer_8x8!(trainer)
    println("Saved manager run: ", outdir)
    return (; outdir, best, csv_path, png_path, graph_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
