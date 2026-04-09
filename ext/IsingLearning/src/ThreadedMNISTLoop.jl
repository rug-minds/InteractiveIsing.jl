export SGD,
       ParameterBuffer,
       MNISTDataLoader,
       init_mnist_trainer,
       fit_mnist_threaded!,
       close_trainer!

using ProgressMeter: Progress, next!, finish!
using LinearAlgebra: diag
import Random

const HAS_MLDATASETS = let available = false
    try
        @eval import MLDatasets
        available = true
    catch
        available = false
    end
    available
end

struct ParameterBuffer{TW<:AbstractVector, TB<:AbstractVector, TA<:AbstractVector}
    w::TW
    b::TB
    α::TA
end

struct SGD{T}
    η::T
end

SGD(; η = 1f-3) = SGD(η)

struct WorkerScratch{G,P,B,S,R}
    graph::G
    params::P
    scratch::B
    equilibrium_state::S
    plus_state::S
    minus_state::S
    relaxer::R
end

struct ThreadedExampleGradient{R} <: Processes.ProcessAlgorithm
    input_layer_idx::Int
    target_idxs::R
    β::Float32
end

struct MNISTDataLoader{TX<:AbstractMatrix, TY<:AbstractMatrix, TI<:AbstractVector{Int}}
    x::TX
    y::TY
    batchsize::Int
    indices::TI
end

struct MNISTThreadedTrainer{G,P,O,W<:Process}
    graph::G
    params::P
    workers::Vector{W}
    optimiser::O
end

function ParameterBuffer(graph)
    biases = copy(InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b))
    weights = copy(SparseArrays.getnzval(adj(graph)))
    self_energies = copy(diag(adj(graph)))
    return ParameterBuffer(weights, biases, self_energies)
end

function gradient_buffer(graph)
    T = eltype(graph)
    return ParameterBuffer(
        zeros(T, length(SparseArrays.getnzval(adj(graph)))),
        zeros(T, nstates(graph)),
        zeros(T, nstates(graph)),
    )
end

function zero_buffer!(buffer::ParameterBuffer)
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

function add_buffer!(dest::ParameterBuffer, src::ParameterBuffer)
    dest.w .+= src.w
    dest.b .+= src.b
    dest.α .+= src.α
    return dest
end

function scale_buffer!(buffer::ParameterBuffer, scale)
    buffer.w .*= scale
    buffer.b .*= scale
    buffer.α .*= scale
    return buffer
end

function apply_sgd!(params::ParameterBuffer, grads::ParameterBuffer, optimiser::SGD)
    η = optimiser.η
    params.w .+= η .* grads.w
    params.b .+= η .* grads.b
    params.α .+= η .* grads.α
    return params
end

function _copy_if_writable!(dest, src)
    try
        dest .= src
    catch err
        if err isa ArgumentError || err isa MethodError
            return false
        end
        rethrow(err)
    end
    return true
end

function apply_params!(graph, params::ParameterBuffer)
    SparseArrays.getnzval(adj(graph)) .= params.w
    _copy_if_writable!(diag(adj(graph)), params.α)
    _copy_if_writable!(InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b), params.b)
    return graph
end

function _set_free_phase!(graph)
    clamping = graph.hamiltonian[InteractiveIsing.Clamping]
    clamping.β[] = zero(eltype(graph))
    fill!(clamping.y, zero(eltype(clamping.y)))
    return graph
end

function _set_clamped_phase!(graph, target_idxs, y, β)
    clamping = graph.hamiltonian[InteractiveIsing.Clamping]
    clamping.β[] = β
    fill!(clamping.y, zero(eltype(clamping.y)))
    @views clamping.y[target_idxs] .= y
    return graph
end

function _apply_input!(graph, input_layer_idx::Int, x)
    InteractiveIsing.off!(graph.index_set, input_layer_idx)
    state(graph[input_layer_idx]) .= x
    return graph
end

function make_worker_scratch(prototype_graph, shared_params; fullsweeps::Integer)
    graph = deepcopy(prototype_graph)
    nsteps = fullsweeps * nstates(graph)
    metropolis = Metropolis()
    relaxer = InlineProcess(metropolis, Input(metropolis, state = graph); repeats = nsteps)
    return WorkerScratch(
        graph,
        shared_params,
        gradient_buffer(graph),
        copy(state(graph)),
        copy(state(graph)),
        copy(state(graph)),
        relaxer,
    )
end

function _run_example_gradient!(
    algo::ThreadedExampleGradient,
    scratch::WorkerScratch,
    buffer::ParameterBuffer,
    x,
    y,
)
    graph = scratch.graph
    apply_params!(graph, scratch.params)

    resetstate!(graph)
    _set_free_phase!(graph)
    _apply_input!(graph, algo.input_layer_idx, x)
    run(scratch.relaxer)
    copyvector!(scratch.equilibrium_state, state(graph))

    state(graph) .= scratch.equilibrium_state
    _set_clamped_phase!(graph, algo.target_idxs, y, algo.β)
    _apply_input!(graph, algo.input_layer_idx, x)
    run(scratch.relaxer)
    copyvector!(scratch.plus_state, state(graph))

    state(graph) .= scratch.equilibrium_state
    _set_clamped_phase!(graph, algo.target_idxs, y, -algo.β)
    _apply_input!(graph, algo.input_layer_idx, x)
    run(scratch.relaxer)
    copyvector!(scratch.minus_state, state(graph))

    contrastive_gradient(
        graph,
        scratch.plus_state,
        scratch.minus_state,
        algo.β;
        buffers = scratch.scratch,
    )
    add_buffer!(buffer, scratch.scratch)
    return nothing
end

function Processes.step!(algo::ThreadedExampleGradient, context)
    _run_example_gradient!(algo, context.scratch, context.buffer, context.x, context.y)
    return nothing
end

function worker_algorithm(
    prototype_graph,
    shared_params;
    β::Real,
    fullsweeps::Integer,
    input_layer_idx::Integer = 1,
    target_layer_idx::Integer = length(prototype_graph),
)
    T = eltype(prototype_graph)
    input_dim = length(state(prototype_graph[input_layer_idx]))
    target_idxs = collect(InteractiveIsing.layerrange(prototype_graph[target_layer_idx]))
    target_dim = length(target_idxs)
    input_dim == 28 * 28 || throw(ArgumentError("MNIST expects 784 input units, got $(input_dim)"))
    target_dim == 10 || throw(ArgumentError("MNIST expects 10 output units, got $(target_dim)"))
    example_step = ThreadedExampleGradient(input_layer_idx, target_idxs, Float32(β))

    algo = @CompositeAlgorithm begin
        @state x = zeros(T, input_dim)
        @state y = zeros(T, target_dim)
        @state buffer = gradient_buffer(prototype_graph)
        @state scratch = make_worker_scratch(prototype_graph, shared_params; fullsweeps = fullsweeps)
        @alias example_step = example_step

        example_step(x = x, y = y, buffer = buffer, scratch = scratch)
    end

    return resolve(algo)
end

function _normalize_mnist_images(images)
    x = Float32.(images) ./ 255f0
    return reshape(x, :, size(images, ndims(images)))
end

function _onehot(labels, nclasses::Integer)
    y = zeros(Float32, nclasses, length(labels))
    @inbounds for (col, label) in enumerate(labels)
        y[Int(label) + 1, col] = 1f0
    end
    return y
end

function load_mnist_arrays(; split::Symbol = :train, limit::Union{Nothing,Int} = nothing)
    HAS_MLDATASETS || error(
        "MLDatasets is required to load MNIST. Add it to the IsingLearning environment and instantiate the project first.",
    )

    images, labels =
        split === :train ? MLDatasets.MNIST.traindata() :
        split === :test ? MLDatasets.MNIST.testdata() :
        throw(ArgumentError("split must be :train or :test, got $(split)"))

    x = _normalize_mnist_images(images)
    y = _onehot(labels, 10)

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

function init_mnist_trainer(
    graph;
    numthreads::Integer = Threads.nthreads(),
    β::Real = 0.1f0,
    fullsweeps::Integer = 10,
    optimiser = SGD(),
    input_layer_idx::Integer = 1,
    target_layer_idx::Integer = length(graph),
)
    numthreads > 0 || throw(ArgumentError("numthreads must be positive"))

    params = ParameterBuffer(graph)
    algo = worker_algorithm(
        graph,
        params;
        β = β,
        fullsweeps = fullsweeps,
        input_layer_idx = input_layer_idx,
        target_layer_idx = target_layer_idx,
    )
    workers = [Process(algo; repeat = 1) for _ in 1:numthreads]
    apply_params!(graph, params)
    return MNISTThreadedTrainer(graph, params, workers, optimiser)
end

function _write_example!(worker::Process, x, y)
    context = getcontext(worker)
    context._state.x .= x
    context._state.y .= y
    return context
end

function _close_workers!(workers)
    for worker in workers
        if !isnothing(worker.task)
            close(worker)
        end
    end
    return workers
end

function close_trainer!(trainer::MNISTThreadedTrainer)
    _close_workers!(trainer.workers)
    return trainer
end

function _reset_worker_buffers!(trainer::MNISTThreadedTrainer)
    for worker in trainer.workers
        zero_buffer!(getcontext(worker)._state.buffer)
    end
    return trainer
end

function _collect_batch_gradient!(dest::ParameterBuffer, trainer::MNISTThreadedTrainer, batchsize::Integer)
    zero_buffer!(dest)
    for worker in trainer.workers
        add_buffer!(dest, getcontext(worker)._state.buffer)
    end
    scale = inv(Float32(batchsize))
    scale_buffer!(dest, scale)
    return dest
end

function _run_minibatch!(trainer::MNISTThreadedTrainer, xbatch, ybatch, batch_gradient::ParameterBuffer)
    _reset_worker_buffers!(trainer)

    nworkers = length(trainer.workers)
    batchsize = size(xbatch, 2)

    for offset in 1:nworkers:batchsize
        active = min(nworkers, batchsize - offset + 1)
        for worker_idx in 1:active
            sample_idx = offset + worker_idx - 1
            worker = trainer.workers[worker_idx]
            _write_example!(worker, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx))
            Processes.reset!(worker)
            run(worker)
        end

        for worker_idx in 1:active
            wait(trainer.workers[worker_idx])
        end

        for worker_idx in 1:active
            close(trainer.workers[worker_idx])
        end
    end

    _collect_batch_gradient!(batch_gradient, trainer, batchsize)
    apply_sgd!(trainer.params, batch_gradient, trainer.optimiser)
    apply_params!(trainer.graph, trainer.params)
    return nothing
end

function fit_mnist_threaded!(
    trainer::MNISTThreadedTrainer;
    epochs::Integer = 1,
    batchsize::Integer = 128,
    split::Symbol = :train,
    shuffle::Bool = true,
    rng = Random.default_rng(),
    limit::Union{Nothing,Int} = nothing,
    show_progress::Bool = true,
)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))

    x, y = load_mnist_arrays(; split, limit)
    batch_gradient = gradient_buffer(trainer.graph)
    stats = NamedTuple[]

    for epoch in 1:epochs
        loader = MNISTDataLoader(x, y; batchsize = batchsize, shuffle = shuffle, rng = rng)
        progress = show_progress ? Progress(length(loader); desc = "MNIST epoch $(epoch)") : nothing
        nbatches = 0

        for (xbatch, ybatch) in loader
            _run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
            nbatches += 1
            progress === nothing || next!(progress; showvalues = [(:batch, nbatches)])
        end

        progress === nothing || finish!(progress)
        push!(stats, (; epoch, nbatches))
    end

    return trainer.params, stats
end

function fit_mnist_threaded!(
    graph;
    numthreads::Integer = Threads.nthreads(),
    β::Real = 0.1f0,
    fullsweeps::Integer = 10,
    optimiser = SGD(),
    close_workers::Bool = true,
    kwargs...,
)
    trainer = init_mnist_trainer(
        graph;
        numthreads = numthreads,
        β = β,
        fullsweeps = fullsweeps,
        optimiser = optimiser,
    )

    try
        return fit_mnist_threaded!(trainer; kwargs...)
    finally
        close_workers && close_trainer!(trainer)
    end
end
