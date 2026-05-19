export MNISTDataLoader,
       init_mnist_trainer,
       fit_mnist_threaded!,
       close_trainer!,
       load_mnist_arrays,
       init_mnist_process_manager,
       dispatch_mnist_forward!,
       poll_mnist_process_manager!,
       run_mnist_forward!,
       mnist_forward_active

using Optimisers
using ProgressMeter: Progress, next!, finish!
using LinearAlgebra: diag
import MLDatasets
import Random

const MNIST_REDUCTION_THREADS_THRESHOLD = 1_000_000
const MNIST_BUFFER_RESET_THREADS_THRESHOLD = 1_000_000

struct MNISTDataLoader{TX<:AbstractMatrix, TY<:AbstractMatrix, TI<:AbstractVector{Int}}
    x::TX
    y::TY
    batchsize::Int
    indices::TI
end

mutable struct MNISTThreadedTrainer{L,G,P,S,W<:Process,V<:Process,O,M}
    layer::L
    prototype_graph::G
    params::P
    opt_state::S
    worker_graphs::Vector{G}
    workers::Vector{W}
    validation_graph::G
    validation_worker::V
    optimiser::O
    manager::M
end

mutable struct MNISTForwardManager{L,G,M}
    layer::L
    graph::G
    manager::M
end

function gradient_buffer(graph)
    base = (;
        w = zeros(eltype(graph), length(SparseArrays.getnzval(adj(graph)))),
        b = zeros(eltype(graph), nstates(graph)),
    )
    isnothing(hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)) && return base
    return merge(base, (; α = zeros(eltype(graph), nstates(graph))))
end

function read_graph_params(graph)
    base = (;
        w = copy(SparseArrays.getnzval(adj(graph))),
        b = copy(InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)),
    )
    isnothing(hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)) && return base
    return merge(base, (; α = copy(diag(adj(graph)))))
end

function sync_graph_params!(graph, params)
    SparseArrays.getnzval(adj(graph)) .= params.w
    InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b) .= params.b
    quadratic = hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)
    if isnothing(quadratic)
        hasproperty(params, :α) && error("graph has no Quadratic local potential but params contains α")
    else
        hasproperty(params, :α) || error("graph has Quadratic local potential but params has no α")
        diag(adj(graph)) .= params.α
        InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.Quadratic, :lp) .= params.α
    end
    return graph
end

function zero_buffer!(buffer)
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

function add_buffer!(dest, src)
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

function scale_buffer!(buffer, scale)
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

function _threaded_fill_zero!(xs::AbstractVector)
    length(xs) < MNIST_BUFFER_RESET_THREADS_THRESHOLD && return fill!(xs, zero(eltype(xs)))
    Threads.@threads for thread_idx in 1:Threads.nthreads()
        first_idx = ((thread_idx - 1) * length(xs)) ÷ Threads.nthreads() + 1
        last_idx = (thread_idx * length(xs)) ÷ Threads.nthreads()
        @inbounds for idx in first_idx:last_idx
            xs[idx] = zero(eltype(xs))
        end
    end
    return xs
end

function zero_buffer_threaded!(buffer)
    _threaded_fill_zero!(buffer.w)
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

function _threaded_sum_scale!(dest::AbstractVector, srcs, scale)
    n = length(dest)
    if n < MNIST_REDUCTION_THREADS_THRESHOLD || Threads.nthreads() == 1
        fill!(dest, zero(eltype(dest)))
        @inbounds for src in srcs
            dest .+= src
        end
        dest .*= scale
        return dest
    end

    Threads.@threads for thread_idx in 1:Threads.nthreads()
        first_idx = ((thread_idx - 1) * n) ÷ Threads.nthreads() + 1
        last_idx = (thread_idx * n) ÷ Threads.nthreads()
        @inbounds for idx in first_idx:last_idx
            acc = zero(eltype(dest))
            for src in srcs
                acc += src[idx]
            end
            dest[idx] = acc * scale
        end
    end
    return dest
end

function _sum_scale_small!(dest::AbstractVector, srcs, scale)
    fill!(dest, zero(eltype(dest)))
    @inbounds for src in srcs
        dest .+= src
    end
    dest .*= scale
    return dest
end

function _layer_state_bounds(layer)
    T = eltype(layer)
    layer_stateset = InteractiveIsing.stateset(layer)
    return T(first(layer_stateset)), T(last(layer_stateset))
end

function _normalize_mnist_images(images, lo::T, hi::T) where {T<:AbstractFloat}
    x = T.(images)
    maximum(x) > one(T) && (x ./= T(255))
    x .*= hi - lo
    x .+= lo
    return reshape(x, :, size(images, ndims(images)))
end

function _onehot(labels, nclasses::Integer; off_value::T, on_value::T) where {T<:AbstractFloat}
    y = fill(off_value, nclasses, length(labels))
    @inbounds for (col, label) in enumerate(labels)
        y[Int(label) + 1, col] = on_value
    end
    return y
end

function load_mnist_arrays(layer::LayeredIsingGraphLayer; split::Symbol = :train, limit::Union{Nothing,Int} = nothing)
    images, labels =
        split === :train ? MLDatasets.MNIST(split = :train)[:] :
        split === :test ? MLDatasets.MNIST(split = :test)[:] :
        throw(ArgumentError("split must be :train or :test, got $(split)"))

    input_lo, input_hi = _layer_state_bounds(layer.model_graph[1])
    output_lo, output_hi = _layer_state_bounds(layer.model_graph[end])

    x = _normalize_mnist_images(images, input_lo, input_hi)
    y = _onehot(labels, 10; off_value = output_lo, on_value = output_hi)

    if !isnothing(limit)
        last_idx = min(limit, size(x, 2))
        x = @view x[:, 1:last_idx]
        y = @view y[:, 1:last_idx]
    end

    return x, y
end

function MNISTDataLoader(x::AbstractMatrix, y::AbstractMatrix; batchsize::Integer, shuffle::Bool = true, rng = Random.default_rng())
    size(x, 2) == size(y, 2) || throw(ArgumentError("x and y must contain the same number of examples"))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))

    indices = collect(1:size(x, 2))
    shuffle && Random.shuffle!(rng, indices)
    return MNISTDataLoader(x, y, batchsize, indices)
end

Base.length(loader::MNISTDataLoader) = cld(length(loader.indices), loader.batchsize)

function Base.iterate(loader::MNISTDataLoader, state::Int = 1)
    state > length(loader.indices) && return nothing
    last_idx = min(state + loader.batchsize - 1, length(loader.indices))
    batch_idxs = @view loader.indices[state:last_idx]
    xbatch = @view loader.x[:, batch_idxs]
    ybatch = @view loader.y[:, batch_idxs]
    return ((xbatch, ybatch), last_idx + 1)
end

function _worker_graph(prototype_graph, params)
    worker_graph = deepcopy(prototype_graph)
    sync_graph_params!(worker_graph, params)
    InteractiveIsing.temp!(worker_graph, prototype_graph.temp)
    return worker_graph
end

function _worker_process(layer, worker_graph)
    algo = resolve(Forward_and_Nudged(layer).algorithm)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = gradient_buffer(worker_graph)

    Process(
        algo,
        Init(:_state;
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(state(worker_graph)),
        ),
        Init(:dynamics, model = worker_graph),
        Init(:plus_capture, state = worker_graph),
        Init(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

function _validation_process(layer, worker_graph)
    algo = resolve(ForwardDynamics(layer; dynamics_algorithm = layer.validation_algorithm).algorithm)
    xdim = length(layer.input_layer)

    Process(
        algo,
        Init(:_state;
            x = zeros(eltype(worker_graph), xdim),
            equilibrium_state = copy(state(worker_graph)),
        ),
        Init(:dynamics, model = worker_graph);
        repeat = 1,
    )
end

function _mnist_training_manager(workers)
    recipe = (;
        prepare! = (slot, job, manager) -> begin
            _write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(recipe; workers, flush_policy = NoFlush(), poll_interval = 0.0)
end

function _mnist_training_manager(layer, graph, params, nworkers::Integer)
    recipe = (;
        makeworker = (idx, manager) -> _worker_process(layer, _worker_graph(graph, params)),
        prepare! = (slot, job, manager) -> begin
            _write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(recipe; nworkers, flush_policy = NoFlush(), poll_interval = 0.0)
end

function MNISTThreadedTrainer(
    layer,
    prototype_graph,
    params,
    opt_state,
    worker_graphs,
    workers,
    validation_graph,
    validation_worker,
    optimiser,
)
    return MNISTThreadedTrainer(
        layer,
        prototype_graph,
        params,
        opt_state,
        worker_graphs,
        workers,
        validation_graph,
        validation_worker,
        optimiser,
        _mnist_training_manager(workers),
    )
end

function _mnist_manager_output!(manager, graph)
    output = manager.state.output[]
    output .= vec(state(graph[end]))
    manager.state.prediction[] = argmax(output) - 1
    return output
end

"""
    init_mnist_process_manager(layer; graph = layer.model_graph)

Build a one-sample forward `ProcessManager` for MNIST debugging. The manager
uses the same `ForwardDynamics` process path as validation, but exposes
`dispatch!`/`poll!` style control for interactive demos.
"""
function init_mnist_process_manager(
    layer::LayeredIsingGraphLayer;
    graph = layer.model_graph,
    poll_interval::Real = 0.0,
)
    worker = _validation_process(layer, graph)
    output_dim = length(layer.output_layer)
    T = eltype(graph)

    recipe = (;
        initstate = config -> (;
            output = Ref(zeros(T, output_dim)),
            prediction = Ref(-1),
            label = Ref(-1),
        ),

        prepare! = (slot, job, manager) -> begin
            resetworker!(slot)
            _write_input!(slot.worker, job.x)
            manager.state.label[] = get(job, :label, -1)
            return nothing
        end,

        consume! = (slot, job, manager) -> begin
            ctx = Processes.context(slot.worker)
            _mnist_manager_output!(manager, ctx.dynamics.model)
            return nothing
        end,

        close! = (slot, manager) -> begin
            close(slot.worker)
            return nothing
        end,
    )

    manager = ProcessManager(
        recipe;
        workers = (worker,),
        flush_policy = NoFlush(),
        poll_interval,
    )
    return MNISTForwardManager(layer, graph, manager)
end

mnist_forward_closed(forward::MNISTForwardManager) = forward.manager.closed

function Base.close(forward::MNISTForwardManager)
    close(forward.manager)
    return nothing
end

function mnist_forward_active(forward::MNISTForwardManager)
    mnist_forward_closed(forward) && return false
    return any(slot -> slot.active, slots(forward.manager))
end

function dispatch_mnist_forward!(forward::MNISTForwardManager, x; label::Integer = -1)
    dispatch!(forward.manager, (; x, label = Int(label)))
    return forward
end

function poll_mnist_process_manager!(forward::MNISTForwardManager)
    mnist_forward_closed(forward) && return forward
    poll!(forward.manager)
    return forward
end

function run_mnist_forward!(forward::MNISTForwardManager, x; label::Integer = -1)
    run!(forward.manager, ((; x, label = Int(label)),))
    return forward.manager.state.output[]
end

function init_mnist_trainer(
    layer::LayeredIsingGraphLayer;
    graph = layer.model_graph,
    numthreads::Integer = Threads.nthreads(),
    optimiser = Optimisers.Descent(1f-3),
)
    numthreads > 0 || throw(ArgumentError("numthreads must be positive"))

    params = read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    manager = _mnist_training_manager(layer, graph, params, Int(numthreads))
    workers = collect(Processes.workers(manager))
    worker_graphs = [Processes.context(worker).dynamics.model for worker in workers]

    validation_template_graph = _worker_graph(graph, params)
    validation_worker = _validation_process(layer, validation_template_graph)
    # println("[threaded-mnist] built validation process id=", validation_worker.id)
    validation_graph = Processes.context(validation_worker).dynamics.model

    return MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        worker_graphs,
        workers,
        validation_graph,
        validation_worker,
        optimiser,
        manager,
    )
end

function close_trainer!(trainer::MNISTThreadedTrainer)
    close(trainer.manager)

    for worker in trainer.workers
        if !isnothing(worker.task)
            close(worker)
        end
    end

    if !isnothing(trainer.validation_worker.task)
        close(trainer.validation_worker)
    end

    return trainer
end

function _write_example!(worker, x, y)
    context = Processes.context(worker)
    context._state.x .= x
    context._state.y .= y
    return context
end

function _write_input!(worker, x)
    context = Processes.context(worker)
    context._state.x .= x
    return context
end

function _reset_batch_buffers!(trainer)
    workers = trainer.workers
    if length(workers) > 1
        Threads.@threads for idx in eachindex(workers)
            zero_buffer!(Processes.context(workers[idx])._state.buffers)
        end
    else
        zero_buffer_threaded!(Processes.context(only(workers))._state.buffers)
    end
    return trainer
end

function _collect_batch_gradient!(trainer, dest, batchsize)
    β = trainer.layer.β
    T = eltype(dest.w)
    scale = inv(T(2) * T(β) * T(batchsize))
    buffers = [Processes.context(worker)._state.buffers for worker in trainer.workers]

    _threaded_sum_scale!(dest.w, map(buffer -> buffer.w, buffers), scale)
    _sum_scale_small!(dest.b, map(buffer -> buffer.b, buffers), scale)
    hasproperty(dest, :α) && _sum_scale_small!(dest.α, map(buffer -> buffer.α, buffers), scale)
    return dest
end

function _broadcast_params!(trainer)
    sync_graph_params!(trainer.prototype_graph, trainer.params)
    for worker_graph in trainer.worker_graphs
        sync_graph_params!(worker_graph, trainer.params)
    end
    sync_graph_params!(trainer.validation_graph, trainer.params)
    return trainer
end

function _validation_output(trainer)
    equilibrium_state = Processes.context(trainer.validation_worker)._state.equilibrium_state
    return @view equilibrium_state[trainer.layer.output_layer]
end

function evaluate_mnist!(
    trainer::MNISTThreadedTrainer,
    x::AbstractMatrix,
    y::AbstractMatrix;
    show_progress::Bool = true,
    desc::AbstractString = "MNIST evaluation",
)
    nsamples = size(x, 2)
    progress = show_progress ? Progress(nsamples; desc = desc) : nothing
    ncorrect = 0
    total_squared_error = zero(eltype(trainer.params.w))

    for sample_idx in 1:nsamples
        worker = trainer.validation_worker
        _write_input!(worker, view(x, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)

        output = _validation_output(trainer)
        target = view(y, :, sample_idx)
        total_squared_error += sum(abs2, output .- target)
        ncorrect += argmax(output) == argmax(target)
        progress === nothing || next!(progress; showvalues = [(:sample, sample_idx)])
    end

    progress === nothing || finish!(progress)
    return (
        accuracy = ncorrect / nsamples,
        classification_error = 1 - ncorrect / nsamples,
        mean_squared_error = total_squared_error / nsamples,
        nsamples = nsamples,
    )
end

function _log_epoch_metrics(epoch, split, metrics)
    println(
        "Epoch ", epoch, " ", split,
        ": classification_error = ", metrics.classification_error,
        ", mean_squared_error = ", metrics.mean_squared_error,
        ", accuracy = ", metrics.accuracy,
    )
    return nothing
end

function _run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
    _reset_batch_buffers!(trainer)

    batchsize = size(xbatch, 2)
    jobs = [(; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)) for sample_idx in 1:batchsize]
    run!(trainer.manager, jobs)

    _collect_batch_gradient!(trainer, batch_gradient, batchsize)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    _broadcast_params!(trainer)
    # println("[threaded-mnist] minibatch update applied")
    return nothing
end

function fit_mnist_threaded!(
    trainer::MNISTThreadedTrainer;
    epochs::Integer = 1,
    batchsize::Integer = 128,
    split::Symbol = :train,
    validation_split::Union{Nothing,Symbol} = :test,
    shuffle::Bool = true,
    rng = Random.default_rng(),
    limit::Union{Nothing,Int} = nothing,
    validation_limit::Union{Nothing,Int} = nothing,
    show_progress::Bool = true,
    show_validation_progress::Bool = show_progress,
    log_metrics::Bool = true,
    train_eval_limit::Union{Nothing,Int} = batchsize,
    full_train_eval_every::Union{Nothing,Int} = nothing,
)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))

    x, y = load_mnist_arrays(trainer.layer; split, limit)
    xvalidation = nothing
    yvalidation = nothing
    if !isnothing(validation_split)
        xvalidation, yvalidation = load_mnist_arrays(trainer.layer; split = validation_split, limit = validation_limit)
    end

    batch_gradient = gradient_buffer(trainer.prototype_graph)
    stats = NamedTuple[]

    for epoch in 1:epochs
        loader = MNISTDataLoader(x, y; batchsize = batchsize, shuffle = shuffle, rng = rng)
        progress = show_progress ? Progress(length(loader); desc = "MNIST epoch $(epoch)") : nothing
        nbatches = 0

        for (xbatch, ybatch) in loader
            # println("[threaded-mnist] epoch ", epoch, " starting minibatch ", nbatches + 1)
            batch_elapsed = @elapsed _run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
            nbatches += 1
            # println("[threaded-mnist] epoch ", epoch, " finished minibatch ", nbatches, " in ", batch_elapsed, " seconds")
            progress === nothing || next!(progress; showvalues = [(:batch, nbatches)])
        end

        progress === nothing || finish!(progress)

        train_eval_count = if !isnothing(full_train_eval_every) && full_train_eval_every > 0 && mod(epoch, full_train_eval_every) == 0
            size(x, 2)
        elseif isnothing(train_eval_limit)
            0
        else
            min(train_eval_limit, size(x, 2))
        end

        train = nothing
        if train_eval_count > 0
            xtrain_eval = @view x[:, 1:train_eval_count]
            ytrain_eval = @view y[:, 1:train_eval_count]
            train = evaluate_mnist!(
                trainer,
                xtrain_eval,
                ytrain_eval;
                show_progress = false,
                desc = "MNIST train $(epoch)",
            )
        end

        validation = nothing
        if !isnothing(xvalidation)
            validation = evaluate_mnist!(
                trainer,
                xvalidation,
                yvalidation;
                show_progress = show_validation_progress,
                desc = "MNIST validation $(epoch)",
            )
        end
        if !isnothing(train)
            log_metrics && _log_epoch_metrics(epoch, "train", train)
        end
        if !isnothing(validation)
            log_metrics && _log_epoch_metrics(epoch, "validation", validation)
        end
        push!(stats, (; epoch, nbatches, train, validation))
    end

    return trainer.params, stats
end

function fit_mnist_threaded!(
    layer::LayeredIsingGraphLayer;
    graph = layer.model_graph,
    numthreads::Integer = Threads.nthreads(),
    optimiser = Optimisers.Descent(1f-3),
    close_workers::Bool = true,
    kwargs...,
)
    trainer = init_mnist_trainer(
        layer;
        graph = graph,
        numthreads = numthreads,
        optimiser = optimiser,
    )

    try
        return fit_mnist_threaded!(trainer; kwargs...)
    finally
        close_workers && close_trainer!(trainer)
    end
end
