using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "hidden8x8_2_64_1_manager.jl"))

"""
    SharedParamHidden8x8Trainer

Manager-backed trainer for the `2 -> 8x8 -> 1` XOR recipe where worker
Hamiltonians share the trainable parameter storage of the prototype graph.

The state, clamping target/mask, RNG, captures, and gradient buffers remain
worker-local. Only the bilinear adjacency weights and magnetic-field bias vector
are shared, so a batch reads one fixed parameter set and the optimizer writes one
updated parameter set back into the shared arrays after the batch.
"""
mutable struct SharedParamHidden8x8Trainer{Layer,Graph,Params,OptState,Optimiser}
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

"""Return trainable graph parameters as aliases to the graph's live storage."""
function shared_graph_params_8x8(graph)
    return (;
        w = SparseArrays.getnzval(II.adj(graph)),
        b = II.getparam(graph.hamiltonian, II.MagField, :b),
    )
end

"""
    shared_hamiltonian_8x8(graph, params)

Create a Hamiltonian for `graph` whose trainable terms read from `params`.
The bilinear term uses `graph.adj`, and the magnetic-field term receives
`params.b` through `NoEnsure` so the template system does not copy the vector.
The clamping term is newly allocated because each worker writes its own target,
mask, and beta during the nudged phase.
"""
function shared_hamiltonian_8x8(graph, params)
    T = eltype(graph)
    target = g -> II.filltype(Vector, zero(T), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(T), II.statelen(g))
    hamiltonian = II.Bilinear() +
        II.MagField(b = II.NoEnsure(params.b)) +
        II.Clamping(β = II.UniformArray(zero(T)), y = target, mask = mask)
    return II.instantiate(hamiltonian, graph)
end

"""
    shared_worker_graph_8x8(prototype_graph, params, config)

Create one worker graph with local spin state but shared trainable parameter
storage. This uses the existing graph fields directly: `adj` is the prototype
adjacency object, and `MagField.b` is `params.b`.
"""
function shared_worker_graph_8x8(prototype_graph, params, config::Hidden8x8Config)
    graph = deepcopy(prototype_graph)
    graph.adj = II.adj(prototype_graph)
    graph.hamiltonian = shared_hamiltonian_8x8(graph, params)
    II.temp!(graph, FT(1e-4))
    return graph
end

"""Copy an optimizer result into the existing shared parameter arrays."""
function write_shared_params_8x8!(params, updated)
    params.w .= updated.w
    params.b .= updated.b
    return params
end

"""Assert that one worker still aliases the shared trainable parameter arrays."""
function assert_shared_worker_params_8x8(worker, params)
    graph = worker.context.dynamics.model
    SparseArrays.getnzval(II.adj(graph)) === params.w ||
        error("worker adjacency weights do not alias shared parameter storage")
    II.getparam(graph.hamiltonian, II.MagField, :b) === params.b ||
        error("worker magnetic-field bias does not alias shared parameter storage")
    return true
end

"""Create one reusable training worker with shared trainable Hamiltonian storage."""
function make_train_worker_shared_8x8(layer, prototype_graph, params, config::Hidden8x8Config, idx::Integer)
    graph = shared_worker_graph_8x8(prototype_graph, params, config)
    worker = IsingLearning._worker_process(layer, graph)
    Random.seed!(worker.context.dynamics.rng, config.base_seed + idx)
    assert_shared_worker_params_8x8(worker, params)
    return worker
end

"""Create one reusable validation worker with shared trainable Hamiltonian storage."""
function make_eval_worker_shared_8x8(layer, prototype_graph, params, config::Hidden8x8Config, idx::Integer)
    graph = shared_worker_graph_8x8(prototype_graph, params, config)
    worker = IsingLearning._validation_process(layer, graph)
    Random.seed!(worker.context.dynamics.rng, config.base_seed + 50_000 + idx)
    assert_shared_worker_params_8x8(worker, params)
    return worker
end

"""Prepare one shared-parameter training worker for a fresh EqProp trajectory."""
function prepare_train_worker_8x8!(worker, trainer::SharedParamHidden8x8Trainer, config::Hidden8x8Config, job::Hidden8x8TrainJob)
    Processes8x8.isdone(worker) && close(worker)
    assert_shared_worker_params_8x8(worker, trainer.params)
    IsingLearning._write_example!(worker, job.x, job.y)
    Processes8x8.reset!(worker)
    return worker
end

"""Prepare one shared-parameter validation worker for a fresh free relaxation."""
function prepare_eval_worker_8x8!(worker, trainer::SharedParamHidden8x8Trainer, config::Hidden8x8Config, job::Hidden8x8EvalJob)
    Processes8x8.isdone(worker) && close(worker)
    assert_shared_worker_params_8x8(worker, trainer.params)
    seed_managed_worker_8x8!(worker, job.seed)
    IsingLearning._write_input!(worker, job.x)
    Processes8x8.reset!(worker)
    return worker
end

"""Recipe used by the persistent shared-parameter training manager."""
function train_manager_recipe_shared_8x8(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> make_train_worker_shared_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        copyworker = (template, idx, manager) -> make_train_worker_shared_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        prepare! = (slot, job, manager) -> prepare_train_worker_8x8!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes8x8.isdone(slot.worker),
        consume! = (slot, job, manager) -> collect_train_response_8x8!(manager.state.current_responses, slot.worker),
        flush! = flush_train_buffers_8x8!,
    )
end

"""Recipe used by the persistent shared-parameter validation manager."""
function eval_manager_recipe_shared_8x8(layer, prototype_graph)
    return (;
        makeworker = (idx, manager) -> make_eval_worker_shared_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        copyworker = (template, idx, manager) -> make_eval_worker_shared_8x8(layer, prototype_graph, manager.state.params, manager.config, idx),
        prepare! = (slot, job, manager) -> prepare_eval_worker_8x8!(slot.worker, manager.state, manager.config, job),
        isdone = (slot, manager) -> Processes8x8.isdone(slot.worker),
        consume! = (slot, job, manager) -> begin
            output = only(II.state(slot.worker.context.dynamics.model[end]))
            push!(manager.state.current_sample_outputs[job.sample_idx], output)
        end,
    )
end

"""Initialize the shared-parameter manager-backed trainer."""
function shared_managed_trainer_8x8(config::Hidden8x8Config; workers::Integer = manager_workers_8x8())
    graph = graph_8x8(config)
    layer = layer_8x8(graph, config)
    params = shared_graph_params_8x8(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    trainer = SharedParamHidden8x8Trainer(
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
        train_manager_recipe_shared_8x8(layer, graph);
        nworkers = workers,
        config,
        state = trainer,
        flush_policy = Processes8x8.FlushAtEnd(),
        poll_interval = 0.0,
        job_type = Hidden8x8TrainJob,
    )
    trainer.eval_manager = Processes8x8.ProcessManager(
        eval_manager_recipe_shared_8x8(layer, graph);
        nworkers = 1,
        config,
        state = trainer,
        flush_policy = Processes8x8.NoFlush(),
        poll_interval = 0.0,
        job_type = Hidden8x8EvalJob,
    )
    return trainer
end

"""Train one epoch through shared-parameter manager workers and apply Adam in-place."""
function train_epoch_shared_managed_8x8!(trainer::SharedParamHidden8x8Trainer, x, y, batch_gradient, epoch::Integer, config::Hidden8x8Config)
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
    trainer.opt_state, updated = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    write_shared_params_8x8!(trainer.params, updated)
    return (; grad_norm, response_norm = isempty(trainer.current_responses) ? zero(FT) : mean(trainer.current_responses))
end

"""Evaluate repeated free-relaxation scalar outputs through shared eval workers."""
function evaluate_shared_managed_8x8!(trainer::SharedParamHidden8x8Trainer, x, y, config::Hidden8x8Config; seed_offset::Integer)
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

"""Close all processes owned by the shared-parameter trainer."""
function close_shared_managed_trainer_8x8!(trainer::SharedParamHidden8x8Trainer)
    !isnothing(trainer.train_manager) && close(trainer.train_manager)
    !isnothing(trainer.eval_manager) && close(trainer.eval_manager)
    return trainer
end

"""Run the shared-parameter manager-backed `2 -> 8x8 -> 1` XOR experiment."""
function main()
    config = Hidden8x8Config()
    workers = manager_workers_8x8()
    outdir = get(
        ENV,
        "ISING_8X8_DIR",
        joinpath(@__DIR__, "runs", "hidden8x8_2_64_1_manager_sharedparams_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)

    trainer = shared_managed_trainer_8x8(config; workers)
    x, y = xor_dataset_8x8(config)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = evaluate_shared_managed_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
    push_row_8x8!(rows, 0, metrics, zero(FT))
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0)
    best_params = deepcopy(trainer.params)
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc, " means=", round.(metrics.means, digits = 3))

    for epoch in 1:config.epochs
        grad = train_epoch_shared_managed_8x8!(trainer, x, y, batch_gradient, epoch, config)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_shared_managed_8x8!(trainer, x, y, config; seed_offset = config.base_seed + 30_000_000)
            push_row_8x8!(rows, epoch, metrics, grad.grad_norm)
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch)
                best_params = deepcopy(trainer.params)
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad.grad_norm, digits = 4), " means=", round.(metrics.means, digits = 3))
        end
    end

    write_shared_params_8x8!(trainer.params, best_params)
    csv_path = write_csv_8x8(joinpath(outdir, "metrics.csv"), rows)
    png_path = plot_8x8(joinpath(outdir, "progress.png"), rows)
    graph_path = II.save_isinggraph(
        joinpath(outdir, "hidden8x8_2_64_1_manager_sharedparams_best_graph.jld2"),
        strip_weight_generators_8x8!(deepcopy(trainer.prototype_graph)),
    )
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Shared-Parameter Manager 2 -> 8x8 -> 1 Scalar XOR")
        println(io)
        println(io, "This duplicates `hidden8x8_2_64_1_manager.jl`, but worker")
        println(io, "Hamiltonians derive trainable parameters from the same memory:")
        println(io, "the prototype graph adjacency `nzval` vector and `MagField.b` vector.")
        println(io, "Worker state, clamping target/mask, RNG, captures, and gradient buffers")
        println(io, "remain local to each worker.")
        println(io)
        println(io, "- workers: `$(workers)`")
        println(io, "- epochs/log_every: `$(config.epochs)` / `$(config.log_every)`")
        println(io, "- free/nudged: `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- Minit/eval repeats: `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- β/lr/T/stepsize: `$(config.β)` / `$(config.lr)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io)
        println(io, "Best logged: epoch `$(best.epoch)`, MSE `$(round(best.mse, digits = 6))`, accuracy `$(best.acc)`.")
        println(io)
        println(io, "CSV: `metrics.csv`")
        println(io, "Plot: `progress.png`")
        println(io, "Best graph: `hidden8x8_2_64_1_manager_sharedparams_best_graph.jld2`")
    end
    close_shared_managed_trainer_8x8!(trainer)
    println("Saved shared-parameter manager run: ", outdir)
    return (; outdir, best, csv_path, png_path, graph_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
