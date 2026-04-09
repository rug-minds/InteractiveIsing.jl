export MNISTDataLoader,
       init_mnist_trainer,
       fit_mnist_threaded!,
       close_trainer!

using Optimisers
using ProgressMeter: Progress, next!, finish!
using LinearAlgebra: diag
import MLDatasets
import Random

struct MNISTDataLoader{TX<:AbstractMatrix, TY<:AbstractMatrix, TI<:AbstractVector{Int}}
    x::TX
    y::TY
    batchsize::Int
    indices::TI
end

mutable struct MNISTThreadedTrainer{L,G,P,S,W<:Process,V<:Process,O}
    layer::L
    prototype_graph::G
    params::P
    opt_state::S
    worker_graphs::Vector{G}
    workers::Vector{W}
    validation_graph::G
    validation_worker::V
    optimiser::O
end

gradient_buffer(graph) = (;
    w = zeros(eltype(graph), length(SparseArrays.getnzval(adj(graph)))),
    b = zeros(eltype(graph), nstates(graph)),
    α = zeros(eltype(graph), nstates(graph)),
)

function read_graph_params(graph)
    return (;
        w = copy(SparseArrays.getnzval(adj(graph))),
        b = copy(InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)),
        α = copy(diag(adj(graph))),
    )
end

function sync_graph_params!(graph, params)
    SparseArrays.getnzval(adj(graph)) .= params.w
    diag(adj(graph)) .= params.α
    InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b) .= params.b
    return graph
end

function zero_buffer!(buffer)
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

function add_buffer!(dest, src)
    dest.w .+= src.w
    dest.b .+= src.b
    dest.α .+= src.α
    return dest
end

function scale_buffer!(buffer, scale)
    buffer.w .*= scale
    buffer.b .*= scale
    buffer.α .*= scale
    return buffer
end

function _layer_state_bounds(layer)
    layer_stateset = InteractiveIsing.stateset(layer)
    return Float32(first(layer_stateset)), Float32(last(layer_stateset))
end

function _normalize_mnist_images(images, lo::Float32, hi::Float32)
    x = Float32.(images) ./ 255f0
    x .*= hi - lo
    x .+= lo
    return reshape(x, :, size(images, ndims(images)))
end

function _onehot(labels, nclasses::Integer; off_value::Float32, on_value::Float32)
    y = fill(off_value, nclasses, length(labels))
    @inbounds for (col, label) in enumerate(labels)
        y[Int(label) + 1, col] = on_value
    end
    return y
end

function load_mnist_arrays(layer::LayeredIsingGraphLayer; split::Symbol = :train, limit::Union{Nothing,Int} = nothing)
    images, labels =
        split === :train ? MLDatasets.MNIST.traindata() :
        split === :test ? MLDatasets.MNIST.testdata() :
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
    InteractiveIsing.temp!(worker_graph, Float32(1e-4))
    return worker_graph
end

function _worker_process(layer, worker_graph)
    algo = resolve(Forward_and_Nudged(layer).algorithm)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = gradient_buffer(worker_graph)

    Process(
        algo,
        Input(:_state;
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(state(worker_graph)),
        ),
        Input(:dynamics, state = worker_graph),
        Input(:plus_capture, state = worker_graph),
        Input(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

function _validation_process(layer, worker_graph)
    algo = resolve(ForwardDynamics(layer).algorithm)
    xdim = length(layer.input_layer)

    Process(
        algo,
        Input(:_state;
            x = zeros(eltype(worker_graph), xdim),
            equilibrium_state = copy(state(worker_graph)),
        ),
        Input(:dynamics, state = worker_graph);
        repeat = 1,
    )
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
    worker_template_graph = _worker_graph(graph, params)
    worker_template = _worker_process(layer, worker_template_graph)
    # println("[threaded-mnist] built worker template process id=", worker_template.id)
    workers = [
        idx == 1 ? worker_template :
        Processes.copyprocess(worker_template; context = deepcopy(worker_template.context))
        for idx in 1:numthreads
    ]
    for (idx, worker) in enumerate(workers)
        # println("[threaded-mnist] worker slot ", idx, " uses process id=", worker.id)
    end
    worker_graphs = [worker.context.dynamics.state for worker in workers]

    validation_template_graph = _worker_graph(graph, params)
    validation_worker = _validation_process(layer, validation_template_graph)
    # println("[threaded-mnist] built validation process id=", validation_worker.id)
    validation_graph = validation_worker.context.dynamics.state

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
    )
end

function close_trainer!(trainer::MNISTThreadedTrainer)
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
    context = worker.context
    context._state.x .= x
    context._state.y .= y
    return context
end

function _write_input!(worker, x)
    context = worker.context
    context._state.x .= x
    return context
end

function _reset_batch_buffers!(trainer)
    for worker in trainer.workers
        zero_buffer!(worker.context._state.buffers)
    end
    return trainer
end

function _collect_batch_gradient!(trainer, dest, batchsize)
    zero_buffer!(dest)
    for worker in trainer.workers
        add_buffer!(dest, worker.context._state.buffers)
    end
    β = trainer.layer.β
    scale_buffer!(dest, inv(Float32(2β * batchsize)))
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
    equilibrium_state = trainer.validation_worker.context._state.equilibrium_state
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
    workers = trainer.workers
    # println("[threaded-mnist] starting minibatch with ", batchsize, " samples on ", length(workers), " workers")

    for sample_idx in 1:batchsize
        while true
            worker_idx = findfirst(worker -> isnothing(worker.task) || Processes.isdone(worker), workers)

            if isnothing(worker_idx)
                # println("[threaded-mnist] sample ", sample_idx, " waiting for free worker")
                yield()
                continue
            end

            worker = workers[worker_idx]
            if Processes.isdone(worker)
                # println("[threaded-mnist] closing finished worker slot ", worker_idx, " process id=", worker.id, " before reusing for sample ", sample_idx)
                close(worker)
            end
            _write_example!(worker, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx))
            Processes.reset!(worker)
            # println("[threaded-mnist] dispatch sample ", sample_idx, " to worker slot ", worker_idx, " process id=", worker.id)
            run(worker)
            break
        end
    end

    for worker in workers
        if !isnothing(worker.task)
            # println("[threaded-mnist] draining worker process id=", worker.id)
            wait(worker)
            # println("[threaded-mnist] closing drained worker process id=", worker.id)
            close(worker)
        end
    end

    # println("[threaded-mnist] minibatch finished, collecting gradients")

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
