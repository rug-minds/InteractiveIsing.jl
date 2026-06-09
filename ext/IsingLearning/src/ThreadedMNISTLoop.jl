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

mutable struct MNISTThreadedTrainer{L,G,P,S,WG,W<:Process,VG,V<:Process,O,M}
    layer::L
    prototype_graph::G
    params::P
    opt_state::S
    worker_graphs::Vector{WG}
    workers::Vector{W}
    validation_graph::VG
    validation_worker::V
    optimiser::O
    manager::M
    share_static_model_data::Bool
end

mutable struct MNISTForwardManager{L,G,M}
    layer::L
    graph::G
    manager::M
end

"""
    MNISTContrastiveStep(layer)

Single MNIST equilibrium-propagation training step implemented as a plain
`ProcessAlgorithm`. It avoids rebuilding the nested `Forward_and_Nudged`
routine for every worker while preserving the manager-facing `_state` context
fields used for input, target, and gradient buffers.
"""
struct MNISTContrastiveStep{D,N,T} <: ProcessAlgorithm
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    β::T
    input_dim::Int
    output_dim::Int
    free_relaxation_steps::Int
    nudged_relaxation_steps::Int
end

function MNISTContrastiveStep(layer::LayeredIsingGraphLayer)
    return MNISTContrastiveStep(
        deepcopy(layer.dynamics_algorithm),
        deepcopy(layer.nudged_dynamics_algorithm),
        layer.β,
        length(layer.input_layer),
        length(layer.output_layer),
        layer.free_relaxation_steps,
        layer.nudged_relaxation_steps,
    )
end

"""
    _relax_mnist_context!(algorithm, context, nsteps)

Run `nsteps` local dynamics updates against an already initialized dynamics
context. The graph state is mutated in place and the context buffers are reused.
"""
function _relax_mnist_context!(algorithm::A, context::C, nsteps::Integer) where {A,C}
    for _ in 1:nsteps
        StatefulAlgorithms.step!(algorithm, context)
    end
    return context
end

"""
    StatefulAlgorithms.init(step::MNISTContrastiveStep, context)

Create the persistent worker context for a custom MNIST contrastive step.
The returned fields live under the algorithm context key, normally `:_state`.
"""
function StatefulAlgorithms.init(step::MNISTContrastiveStep, context)
    model = context.model
    T = eltype(model)
    x = get(context, :x, zeros(T, step.input_dim))
    y = get(context, :y, zeros(T, step.output_dim))
    buffers = get(context, :buffers, gradient_buffer(model))
    equilibrium_state = get(context, :equilibrium_state, copy(state(model)))
    plus_state = get(context, :plus_state, similar(equilibrium_state))
    minus_state = get(context, :minus_state, similar(equilibrium_state))
    input_pattern = get(context, :input_pattern, isnothing(_mnist_input_magfield(model)) ? nothing : zeros(T, nstates(model)))

    free_context = StatefulAlgorithms.init(step.dynamics_algorithm, (; model))
    nudged_context = StatefulAlgorithms.init(step.nudged_dynamics_algorithm, (; model))

    return (; model, x, y, buffers, equilibrium_state, plus_state, minus_state, input_pattern, free_context, nudged_context)
end

"""Return the object inside a `Ref`, or the object itself for direct values."""
@inline _mnist_ref_value(x::Base.RefValue) = x[]
@inline _mnist_ref_value(x) = x

"""Return the current MNIST input vector from direct state or a job-buffer reference."""
function _mnist_context_x(context::C) where {C}
    return _mnist_ref_value(context.x)
end

"""Return the current MNIST target vector from direct state or a job-buffer reference."""
function _mnist_context_y(context::C) where {C}
    return _mnist_ref_value(context.y)
end

"""
    _apply_mnist_context_input!(model, context)

Apply the input stored in an MNIST worker context. When the graph owns a second
`MagField`, the context stores the already-precomputed field pattern and this
only installs that pattern for the current phase.
"""
function _apply_mnist_context_input!(model::G, context::C) where {G,C}
    if hasproperty(context, :input_pattern) && !isnothing(context.input_pattern)
        apply_input_pattern!(model, context.input_pattern)
    else
        apply_input(model, context.x)
    end
    return model
end

"""
    _prepare_mnist_context_input_pattern!(model, context)

Compute the sample-local MNIST input field inside the worker task. The manager
only writes `context.x`; this keeps the input projection parallel with the rest
of the per-sample worker work.
"""
function _prepare_mnist_context_input_pattern!(model::G, context::C, x) where {G,C}
    if hasproperty(context, :input_pattern) && !isnothing(context.input_pattern)
        precompute_mnist_input_pattern!(model, context.input_pattern, x)
    end
    return model
end

"""
    _accumulate_input_field_edge_gradient!(buffers, model, x, plus_state, minus_state)

Add the bilinear contrastive-gradient terms for edges touching the input layer
when MNIST input is represented as a precomputed local field. In that mode the
input layer state is cleared during dynamics to avoid double-counting the image,
so the generic bilinear derivative cannot see the fixed input values.
"""
function _accumulate_input_field_edge_gradient!(
    buffers::B,
    model::G,
    x::X,
    plus_state::S,
    minus_state::S,
) where {B,G,X<:AbstractVector,S<:AbstractVector}
    input_idxs = InteractiveIsing.layerrange(model[1])
    input_first = first(input_idxs)
    input_last = last(input_idxs)

    adjacency = adj(model)
    colptrs = SparseArrays.getcolptr(adjacency)
    rowvals = SparseArrays.rowvals(adjacency)

    # Buffers are in the same CSC pointer order as `SparseArrays.nonzeros(adj)`.
    @inbounds for col_idx in 1:size(adjacency, 2)
        col_is_input = input_first <= col_idx <= input_last
        for ptr in colptrs[col_idx]:(colptrs[col_idx + 1] - 1)
            row_idx = rowvals[ptr]
            row_is_input = input_first <= row_idx <= input_last
            col_is_input == row_is_input && continue

            if col_is_input
                xval = x[col_idx - input_first + 1]
                other_idx = row_idx
            else
                xval = x[row_idx - input_first + 1]
                other_idx = col_idx
            end

            buffers.w[ptr] += -oftype(buffers.w[ptr], 0.5) * xval * plus_state[other_idx]
            buffers.w[ptr] -= -oftype(buffers.w[ptr], 0.5) * xval * minus_state[other_idx]
        end
    end
    return buffers
end

"""
    StatefulAlgorithms.step!(step::MNISTContrastiveStep, context)

Run free, positive nudged, and negative nudged phases for one MNIST sample and
accumulate the symmetric contrastive gradient into `context.buffers`.
"""
function StatefulAlgorithms.step!(step::MNISTContrastiveStep, context)
    model = context.model
    β = step.β
    x = _mnist_context_x(context)
    y = _mnist_context_y(context)
    _prepare_mnist_context_input_pattern!(model, context, x)

    resetstate!(model)
    _apply_mnist_context_input!(model, context)
    _relax_mnist_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
    context.equilibrium_state .= state(model)

    state(model) .= context.equilibrium_state
    _apply_mnist_context_input!(model, context)
    apply_targets(model, y)
    set_clamping_beta!(model, β)
    _relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.plus_state .= state(model)

    state(model) .= context.equilibrium_state
    _apply_mnist_context_input!(model, context)
    apply_targets(model, y)
    set_clamping_beta!(model, -β)
    _relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.minus_state .= state(model)

    set_clamping_beta!(model, zero(β))
    contrastive_gradient(model, context.plus_state, context.minus_state, β; buffers = context.buffers)
    if hasproperty(context, :input_pattern) && !isnothing(context.input_pattern)
        _accumulate_input_field_edge_gradient!(
            context.buffers,
            model,
            x,
            context.plus_state,
            context.minus_state,
        )
    end
    return nothing
end

"""
    StatefulAlgorithms.cleanup(step::MNISTContrastiveStep, context)

Finalize an MNIST contrastive worker step. The algorithm owns no external
resources, so cleanup intentionally leaves the persistent buffers untouched.
"""
function StatefulAlgorithms.cleanup(step::MNISTContrastiveStep, context)
    return nothing
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
        b = copy(_mnist_base_magfield(graph).b),
    )
    isnothing(hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)) && return base
    return merge(base, (; α = copy(diag(adj(graph)))))
end

function sync_graph_params!(graph, params)
    SparseArrays.getnzval(adj(graph)) .= params.w
    _mnist_base_magfield(graph).b .= params.b
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

"""
    _mnist_output_replicas(layer)

Return how many output spins encode one MNIST digit class. A plain `10`-spin
output layer has one replica per class; a `40`-spin layer has four.
"""
function _mnist_output_replicas(layer::L) where {L<:LayeredIsingGraphLayer}
    noutputs = length(layer.output_layer)
    replicas, remainder = divrem(noutputs, MNIST_NCLASSES)
    remainder == 0 || throw(DimensionMismatch("MNIST output length must be a multiple of $(MNIST_NCLASSES), got $(noutputs)"))
    replicas > 0 || throw(DimensionMismatch("MNIST output layer must contain at least $(MNIST_NCLASSES) spins"))
    return replicas
end

"""
    _mnist_repeated_targets(labels, replicas; off_value, on_value)

Build MNIST targets with `replicas` adjacent output spins per digit class.
This supports the `784 -> 120 -> 40` layout from the Ising-machine paper while
preserving the original one-spin-per-class behavior when `replicas == 1`.
"""
function _mnist_repeated_targets(labels, replicas::Integer; off_value::T, on_value::T) where {T<:AbstractFloat}
    replicas > 0 || throw(ArgumentError("replicas must be positive"))
    y = fill(off_value, MNIST_NCLASSES * replicas, length(labels))
    @inbounds for (col, label) in enumerate(labels)
        first_idx = Int(label) * replicas + 1
        last_idx = first_idx + replicas - 1
        y[first_idx:last_idx, col] .= on_value
    end
    return y
end

"""
    _mnist_class_scores(output)

Reduce a `10 * replicas` MNIST output vector to one score per digit by summing
the replica spins for each class.
"""
function _mnist_class_scores(output::V) where {V<:AbstractVector}
    replicas, remainder = divrem(length(output), MNIST_NCLASSES)
    remainder == 0 || throw(DimensionMismatch("MNIST output length must be a multiple of $(MNIST_NCLASSES), got $(length(output))"))
    scores = zeros(eltype(output), MNIST_NCLASSES)
    @inbounds for digit_idx in 1:MNIST_NCLASSES
        first_idx = (digit_idx - 1) * replicas + 1
        last_idx = first_idx + replicas - 1
        scores[digit_idx] = sum(view(output, first_idx:last_idx))
    end
    return scores
end

function load_mnist_arrays(layer::LayeredIsingGraphLayer; split::Symbol = :train, limit::Union{Nothing,Int} = nothing)
    images, labels =
        split === :train ? MLDatasets.MNIST(split = :train)[:] :
        split === :test ? MLDatasets.MNIST(split = :test)[:] :
        throw(ArgumentError("split must be :train or :test, got $(split)"))

    input_lo, input_hi = _layer_state_bounds(layer.model_graph[1])
    output_lo, output_hi = _layer_state_bounds(layer.model_graph[end])

    x = _normalize_mnist_images(images, input_lo, input_hi)
    y = _mnist_repeated_targets(labels, _mnist_output_replicas(layer); off_value = output_lo, on_value = output_hi)

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

"""
    _shared_mnist_learning_term(prototype_graph)

Return a fresh worker-local learning nudging term matching the prototype graph's
target-loss term while keeping target buffers worker-local.
"""
function _shared_mnist_learning_term(prototype_graph::G) where {G}
    softplus_margin = hamiltonian_or_nothing(
        prototype_graph.hamiltonian,
        InteractiveIsing.SoftplusMarginNudging,
    )
    if !isnothing(softplus_margin)
        return InteractiveIsing.SoftplusMarginNudging(
            β = InteractiveIsing.UniformArray(zero(eltype(prototype_graph))),
            y = g -> InteractiveIsing.filltype(Vector, zero(eltype(prototype_graph)), statelen(g)),
            τ = InteractiveIsing.UniformArray(eltype(prototype_graph)(softplus_margin.τ[])),
        )
    end

    return Clamping(
        β = InteractiveIsing.UniformArray(zero(eltype(prototype_graph))),
        y = g -> InteractiveIsing.filltype(Vector, zero(eltype(prototype_graph)), statelen(g)),
    )
end

"""
    _shared_mnist_worker_graph(prototype_graph; input_mode = :state)

Create a worker graph with the same layer layout as `prototype_graph`, fresh
state, shared adjacency, shared base bias, and optional worker-local input
field. The prototype graph remains the only graph that receives parameter
writes after optimizer updates.
"""
function _shared_mnist_worker_graph(prototype_graph::G; input_mode::Symbol = :state) where {G}
    input_b = input_mode === :field ? zeros(eltype(prototype_graph), nstates(prototype_graph)) :
        input_mode === :state ? nothing :
        throw(ArgumentError("input_mode must be :state or :field, got $(repr(input_mode))"))

    base_bias = InteractiveIsing.Force(_mnist_base_magfield(prototype_graph).b)
    hamiltonian = Bilinear() + MagField(b = base_bias)
    if !isnothing(input_b)
        hamiltonian = hamiltonian + MagField(b = input_b)
    end
    hamiltonian = hamiltonian + _shared_mnist_learning_term(prototype_graph)

    worker_graph = IsingGraph(
        getfield(prototype_graph, :layers)...,
        hamiltonian;
        precision = eltype(prototype_graph),
        adj = adj(prototype_graph),
        index_set = g -> ToggledIndexSet(g),
    )
    InteractiveIsing.temp!(worker_graph, prototype_graph.temp)
    return worker_graph
end

"""
    _worker_graph(prototype_graph, params; share_static_model_data = false, input_mode = :state)

Create the graph owned by one MNIST worker. In shared mode this supports any
MNIST-like layered graph with input first and output last; copied mode preserves
the previous deep-copy behavior.
"""
function _worker_graph(prototype_graph, params; share_static_model_data::Bool = false, input_mode::Symbol = :state)
    if share_static_model_data
        return _shared_mnist_worker_graph(prototype_graph; input_mode)
    end

    worker_graph = deepcopy(prototype_graph)
    sync_graph_params!(worker_graph, params)
    InteractiveIsing.temp!(worker_graph, prototype_graph.temp)
    return worker_graph
end

"""
    _sync_worker_graph_params!(worker_graph, prototype_graph, params)

Synchronize copied worker parameter arrays after a minibatch. Arrays shared
with the prototype graph are skipped because `sync_graph_params!` on the
prototype has already updated them.
"""
function _sync_worker_graph_params!(worker_graph::WG, prototype_graph::PG, params) where {WG,PG}
    SparseArrays.getnzval(adj(worker_graph)) === SparseArrays.getnzval(adj(prototype_graph)) ||
        (SparseArrays.getnzval(adj(worker_graph)) .= params.w)

    worker_bias = _mnist_base_magfield(worker_graph).b
    prototype_bias = _mnist_base_magfield(prototype_graph).b
    worker_bias === prototype_bias || (worker_bias .= params.b)

    quadratic = hamiltonian_or_nothing(worker_graph.hamiltonian, InteractiveIsing.Quadratic)
    if !isnothing(quadratic)
        hasproperty(params, :α) || error("graph has Quadratic local potential but params has no α")
        diag(adj(worker_graph)) .= params.α
        InteractiveIsing.getparam(worker_graph.hamiltonian, InteractiveIsing.Quadratic, :lp) .= params.α
    end
    return worker_graph
end

"""
    _mnist_worker_state(worker)

Return the mutable subcontext that stores MNIST sample state and gradient
buffers. The legacy DSL worker names it `:_state`; the custom contrastive worker
has one process subcontext, so that subcontext is used directly.
"""
function _mnist_worker_state(worker::W) where {W}
    return StatefulAlgorithms.context(worker)._state
end

function _worker_process(layer, worker_graph)
    algo = :_state => MNISTContrastiveStep(layer)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = gradient_buffer(worker_graph)

    Process(
        algo,
        Init(:_state;
            model = worker_graph,
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(state(worker_graph)),
            plus_state = similar(state(worker_graph)),
            minus_state = similar(state(worker_graph)),
            input_pattern = isnothing(_mnist_input_magfield(worker_graph)) ? nothing : zeros(eltype(worker_graph), nstates(worker_graph)),
        );
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
        loadjob! = (slot, job, manager) -> begin
            _write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(recipe; workers, sync_policy = NoSync(), poll_interval = 0.0)
end

function _mnist_training_manager(layer, graph, params, nworkers::Integer)
    recipe = (;
        makeworker = (idx, manager) -> _worker_process(layer, _worker_graph(graph, params)),
        loadjob! = (slot, job, manager) -> begin
            _write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(recipe; nworkers, sync_policy = NoSync(), poll_interval = 0.0)
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
    share_static_model_data::Bool = false,
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
        share_static_model_data,
    )
end

function _mnist_manager_output!(manager, graph)
    output = manager.state.output[]
    output .= vec(state(graph[end]))
    manager.state.prediction[] = argmax(_mnist_class_scores(output)) - 1
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

        loadjob! = (slot, job, manager) -> begin
            resetworker!(slot)
            _write_input!(slot.worker, job.x)
            manager.state.label[] = get(job, :label, -1)
            return nothing
        end,

        afterjob! = (slot, job, manager) -> begin
            ctx = StatefulAlgorithms.context(slot.worker)
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
        sync_policy = NoSync(),
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
    share_static_model_data::Bool = false,
    input_mode::Symbol = :state,
)
    numthreads > 0 || throw(ArgumentError("numthreads must be positive"))

    params = read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    recipe = (;
        makeworker = (idx, manager) -> _worker_process(
            layer,
            _worker_graph(
                graph,
                params;
                share_static_model_data = manager.config.share_static_model_data,
                input_mode = manager.config.input_mode,
            ),
        ),
        loadjob! = (slot, job, manager) -> begin
            _write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    manager = ProcessManager(
        recipe;
        nworkers = Int(numthreads),
        config = (; share_static_model_data, input_mode),
        worker_init = share_static_model_data ? StatefulAlgorithms.MakeEachWorker() : StatefulAlgorithms.CopyFirstWorker(),
        sync_policy = NoSync(),
        poll_interval = 0.0,
    )
    workers = collect(StatefulAlgorithms.workers(manager))
    worker_graphs = [_mnist_worker_state(worker).model for worker in workers]

    validation_template_graph = _worker_graph(graph, params; share_static_model_data, input_mode)
    validation_worker = _validation_process(layer, validation_template_graph)
    # println("[threaded-mnist] built validation process id=", validation_worker.id)
    validation_graph = StatefulAlgorithms.context(validation_worker).dynamics.model

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
        share_static_model_data,
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
    state_context = _mnist_worker_state(worker)
    state_context.x .= x
    state_context.y .= y
    return state_context
end

function _write_input!(worker, x)
    state_context = _mnist_worker_state(worker)
    state_context.x .= x
    return state_context
end

function _reset_batch_buffers!(trainer)
    workers = trainer.workers
    if length(workers) > 1
        Threads.@threads for idx in eachindex(workers)
            zero_buffer!(_mnist_worker_state(workers[idx]).buffers)
        end
    else
        zero_buffer_threaded!(_mnist_worker_state(only(workers)).buffers)
    end
    return trainer
end

function _collect_batch_gradient!(trainer, dest, batchsize)
    β = trainer.layer.β
    T = eltype(dest.w)
    scale = inv(T(2) * T(β) * T(batchsize))
    buffers = [_mnist_worker_state(worker).buffers for worker in trainer.workers]

    _threaded_sum_scale!(dest.w, map(buffer -> buffer.w, buffers), scale)
    _sum_scale_small!(dest.b, map(buffer -> buffer.b, buffers), scale)
    hasproperty(dest, :α) && _sum_scale_small!(dest.α, map(buffer -> buffer.α, buffers), scale)
    return dest
end

"""
    _broadcast_params!(trainer)

Copy the latest optimizer-owned parameter arrays into every graph that can be
used after a minibatch update. Worker graph copies are independent, so they can
be synchronized in parallel once the manager has drained the current batch.
"""
function _broadcast_params!(trainer)
    sync_graph_params!(trainer.prototype_graph, trainer.params)

    worker_graphs = trainer.worker_graphs
    if length(worker_graphs) > 1 && Threads.nthreads() > 1
        Threads.@threads for idx in eachindex(worker_graphs)
            _sync_worker_graph_params!(worker_graphs[idx], trainer.prototype_graph, trainer.params)
        end
    else
        for worker_graph in worker_graphs
            _sync_worker_graph_params!(worker_graph, trainer.prototype_graph, trainer.params)
        end
    end

    _sync_worker_graph_params!(trainer.validation_graph, trainer.prototype_graph, trainer.params)
    return trainer
end

function _validation_output(trainer)
    equilibrium_state = StatefulAlgorithms.context(trainer.validation_worker)._state.equilibrium_state
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
        StatefulAlgorithms.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)

        output = _validation_output(trainer)
        target = view(y, :, sample_idx)
        total_squared_error += sum(abs2, output .- target)
        ncorrect += argmax(_mnist_class_scores(output)) == argmax(_mnist_class_scores(target))
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
