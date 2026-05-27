using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CairoMakie
using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using MLDatasets
using Optimisers
using Random
using Serialization
using SparseArrays
using Statistics

CairoMakie.activate!()

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float32
const INPUT_DIM = IsingLearning.MNIST_INPUT_DIM
const NCLASSES = IsingLearning.MNIST_NCLASSES

Base.@kwdef struct InputFieldMNISTConfig{T<:AbstractFloat,S<:AbstractString}
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_IF_WORKERS", "32"))
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EPOCHS", "200"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_IF_BATCHSIZE", "128"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_PER_CLASS", "5421"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TEST_PER_CLASS", "892"))
    train_eval_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS", "100"))
    eval_every::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EVAL_EVERY", "5"))
    hidden::Int = parse(Int, get(ENV, "ISING_MNIST_IF_HIDDEN", "120"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS", "4"))
    sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_SWEEPS", "500"))
    β::T = parse(FT, get(ENV, "ISING_MNIST_IF_BETA", "5.0"))
    lr::T = parse(FT, get(ENV, "ISING_MNIST_IF_LR", "0.003"))
    temp::T = parse(FT, get(ENV, "ISING_MNIST_IF_TEMP", "0.001"))
    stepsize::T = parse(FT, get(ENV, "ISING_MNIST_IF_STEPSIZE", "0.5"))
    weight_scale::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_SCALE", "0.005"))
    weight_decay::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_DECAY", "0.0"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_IF_SEED", "20260526"))
    outdir::S = get(
        ENV,
        "ISING_MNIST_IF_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "mnist_784_120_40_adam_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

struct InputFieldMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

mutable struct InputFieldMNISTManagerState{L,G,P,B,O}
    layer::L
    source_graph::G
    params::Base.RefValue{P}
    batch_gradient::B
    nsamples::Base.RefValue{Int}
    opt_state::O
end

"""Append one named-tuple row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the number of non-input spins stepped by the MNIST dynamics."""
function active_units(graph::G) where {G}
    total = 0
    for layer_idx in 2:length(graph)
        total += length(II.layerrange(graph[layer_idx]))
    end
    return total
end

"""Set every parameter-gradient array to zero."""
function clear_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Add one gradient buffer into another buffer with matching fields."""
function add_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

"""Scale a gradient buffer in place."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Write one runtime beta value into a worker-local reference."""
function set_beta_ref!(beta_ref::R, beta::T) where {R<:Base.RefValue,T<:Real}
    beta_ref[] = beta
    return nothing
end

"""Normalize raw MNIST images to `[0, 1]` input-field intensities."""
function normalize_images(images::A, ::Type{T}) where {A,T<:AbstractFloat}
    x = T.(images)
    maximum(x) > one(T) && (x ./= T(255))
    return reshape(x, :, size(images, ndims(images)))
end

"""Build four-replica output targets with `-1` off spins and `+1` on spins."""
function repeated_targets(labels::L, replicas::I, ::Type{T}) where {L,I<:Integer,T<:AbstractFloat}
    y = fill(-one(T), NCLASSES * Int(replicas), length(labels))
    @inbounds for (col, label) in enumerate(labels)
        first_idx = Int(label) * Int(replicas) + 1
        y[first_idx:(first_idx + Int(replicas) - 1), col] .= one(T)
    end
    return y
end

"""Load a balanced MNIST split with the same number of examples per class."""
function balanced_mnist(split::Symbol, per_class::I, config::C) where {I<:Integer,C<:InputFieldMNISTConfig}
    dataset = split === :train ? MLDatasets.MNIST(split = :train) :
        split === :test ? MLDatasets.MNIST(split = :test) :
        throw(ArgumentError("split must be :train or :test"))
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end

    keep = Int[]
    rng = Random.MersenneTwister(hash((config.seed, split, Int(per_class))))
    for digit in 1:NCLASSES
        length(buckets[digit]) >= Int(per_class) ||
            throw(ArgumentError("split $(split) has only $(length(buckets[digit])) samples for digit $(digit - 1)"))

        # Shuffle class buckets once so reduced runs are still representative.
        digit_indices = copy(buckets[digit])
        Random.shuffle!(rng, digit_indices)
        append!(keep, @view digit_indices[1:Int(per_class)])
    end

    xall = normalize_images(images, FT)
    yall = repeated_targets(labels, config.output_replicas, FT)
    return Matrix{FT}(xall[:, keep]), Matrix{FT}(yall[:, keep])
end

"""Average output replicas into one score per digit."""
function class_scores(output::V, replicas::I) where {V<:AbstractVector,I<:Integer}
    scores = zeros(eltype(output), NCLASSES)
    @inbounds for digit in 1:NCLASSES
        first_idx = (digit - 1) * Int(replicas) + 1
        scores[digit] = mean(@view output[first_idx:(first_idx + Int(replicas) - 1)])
    end
    return scores
end

"""Create the source graph and graph-backed layer for the baseline."""
function build_layer(config::C) where {C<:InputFieldMNISTConfig}
    graph = IsingLearning.MNISTArchitecture(
        hidden = config.hidden,
        output_replicas = config.output_replicas,
        precision = FT,
        weight_scale = config.weight_scale,
        rng = Random.MersenneTwister(config.seed),
    )
    II.temp!(graph, config.temp)

    relaxation_steps = max(1, round(Int, config.sweeps * active_units(graph)))
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = FT(0.15),
        adjusted = false,
        order = :cyclic,
    )
    layer = IsingLearning.MNISTLayer(
        graph = graph,
        β = config.β,
        free_relaxation_steps = relaxation_steps,
        nudged_relaxation_steps = relaxation_steps,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return (; graph, layer, relaxation_steps)
end

# Accumulate the one-sided MNIST input-field gradient into one worker buffer.
function accumulate_input_field_gradient!(
    isinggraph::G,
    nudged_state,
    equilibrium_state,
    x,
    buffers,
    β::T,
) where {G,T<:Real}
    IsingLearning.contrastive_gradient(isinggraph, nudged_state, equilibrium_state, β; buffers = buffers)
    IsingLearning._accumulate_input_field_edge_gradient!(buffers, isinggraph, x, nudged_state, equilibrium_state)
    return
end

"""Build the free-phase input-field routine for one MNIST sample."""
function input_field_free_phase_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    free_steps = layer.free_relaxation_steps
    n_units = layer.nunits

    return Processes.@Routine begin
        @state x
        @state equilibrium_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Fold the image into the worker-local input field, then relax once.
        II.resetstate!(dynamics.model)
        IsingLearning.apply_input(dynamics.model, x)
        model = @repeat free_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(graph -> II.state(graph), model))
    end
end

"""Build the positive-nudged input-field routine for one MNIST sample."""
function input_field_nudged_phase_algorithm(layer::L, beta_ref::R) where {L<:IsingLearning.LayeredIsingGraphLayer,R<:Base.RefValue}
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits

    return Processes.@Routine begin
        @state x
        @state y
        @state equilibrium_state
        @state nudged_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Restart from the free equilibrium, add the target nudge, then relax.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, beta_ref[])
        model = @repeat nudged_steps dynamics()
        IsingLearning.copyvector!(nudged_state, @transform(graph -> II.state(graph), model))

        # Reset clamping so the worker can be rerun without rebuilding state.
        IsingLearning.set_clamping_beta!(dynamics.model, zero(beta_ref[]))
    end
end

"""Build the composite single-sample input-field equilibrium-propagation step."""
function input_field_contrastive_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    default_β = layer.β
    beta_ref = Ref(default_β)
    free_phase = input_field_free_phase_algorithm(layer)
    nudged_phase = input_field_nudged_phase_algorithm(layer, beta_ref)

    return Processes.@CompositeAlgorithm begin
        @state x
        @state y
        @state buffers
        @input beta::FT = default_β

        @context free_context = free_phase()
        set_beta_ref!(beta_ref, beta)
        @context nudged_context = nudged_phase()

        accumulate_input_field_gradient!(
            nudged_context.dynamics.model,
            nudged_context.nudged_state,
            free_context.equilibrium_state,
            x,
            buffers,
            beta,
        )
    end
end

"""Return the mutable sample/buffer context stored in one worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Return the graph owned by a worker dynamics context."""
function worker_graph(worker::W) where {W}
    return Processes.context(worker).dynamics.model
end

"""Build a worker graph whose adjacency is the source graph adjacency object."""
function shared_worker_graph(source::G) where {G}
    graph = IsingLearning._shared_mnist_worker_graph(source; input_mode = :field)
    II.adj(graph) === II.adj(source) || error("worker graph does not share source adjacency")
    SparseArrays.nonzeros(II.adj(graph)) === SparseArrays.nonzeros(II.adj(source)) ||
        error("worker graph J storage is not pointer-shared with source graph")
    return graph
end

"""Build one ProcessManager worker from the already resolved baseline LoopAlgorithm."""
function input_field_worker(algorithm::A, layer::L, graph::G) where {A,L<:IsingLearning.LayeredIsingGraphLayer,G}
    state = II.state(graph)
    return Processes.Process(
        algorithm,
        Processes.Init(:_state;
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            buffers = IsingLearning.gradient_buffer(graph),
            equilibrium_state = copy(state),
            nudged_state = similar(state),
        ),
        Processes.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Clear the manager batch buffer and all worker-local gradient buffers."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    manager.state.nsamples[] = 0
    for worker in Processes.workers(manager)
        clear_buffer!(worker_context(worker).buffers)
    end
    return manager
end

"""Flush worker gradients into one Adam-ready minibatch gradient."""
function flush_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        add_buffer!(manager.state.batch_gradient, worker_context(worker).buffers)
        clear_buffer!(worker_context(worker).buffers)
    end

    nsamples = manager.state.nsamples[]
    nsamples > 0 || throw(ArgumentError("cannot flush an empty MNIST minibatch"))
    scale_buffer!(manager.state.batch_gradient, inv(FT(manager.config.β) * FT(nsamples)))
    manager.config.weight_decay > zero(FT) && (manager.state.batch_gradient.w .+= manager.config.weight_decay .* manager.state.params[].w)
    return manager.state.batch_gradient
end

"""Synchronize the source graph and shared worker fields after an Adam update."""
function sync_after_update!(manager::M, params::P) where {M<:Processes.ProcessManager,P}
    IsingLearning.sync_graph_params!(manager.state.source_graph, params)
    for worker in Processes.workers(manager)
        IsingLearning._sync_worker_graph_params!(worker_graph(worker), manager.state.source_graph, params)
    end
    return manager
end

"""Construct the ProcessManager that owns persistent input-field workers."""
function input_field_manager(layer::L, source::G, config::C) where {L<:IsingLearning.LayeredIsingGraphLayer,G,C<:InputFieldMNISTConfig}
    params = IsingLearning.read_graph_params(source)
    optimiser = Optimisers.Adam(config.lr)
    algorithm = Processes.resolve(input_field_contrastive_algorithm(layer))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        IsingLearning.gradient_buffer(source),
        Ref(0),
        Optimisers.setup(optimiser, params),
    )
    recipe = (;
        makeworker = (idx, manager) -> input_field_worker(algorithm, manager.state.layer, shared_worker_graph(manager.state.source_graph)),
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x .= job.x
            ctx.y .= job.y
            manager.state.nsamples[] += 1
            Processes.resetworker!(slot)
            return nothing
        end,
        runarguments = (slot, job, manager) -> (; beta = manager.config.β),
        flush! = flush_manager_buffers!,
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = InputFieldMNISTJob{Vector{FT},Vector{FT}},
    )
end

"""Split a minibatch index view into concrete manager jobs."""
function batch_jobs(x::X, y::Y, indices::V) where {X<:AbstractMatrix,Y<:AbstractMatrix,V<:AbstractVector{Int}}
    jobs = InputFieldMNISTJob{Vector{FT},Vector{FT}}[]
    for sample_idx in indices
        push!(jobs, InputFieldMNISTJob(copy(view(x, :, sample_idx)), copy(view(y, :, sample_idx))))
    end
    return jobs
end

"""Run one minibatch through the manager and apply one Adam update."""
function run_minibatch!(manager::M, jobs::J) where {M<:Processes.ProcessManager,J<:AbstractVector}
    clear_manager_buffers!(manager)
    Processes.run!(manager, jobs, Processes.Dynamic())
    manager.state.opt_state, params = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = params
    sync_after_update!(manager, params)
    return params
end

"""Build the reusable validation process with input-field graph semantics."""
function validation_worker(layer::L, source::G) where {L<:IsingLearning.LayeredIsingGraphLayer,G}
    graph = shared_worker_graph(source)
    algorithm = Processes.resolve(IsingLearning.ForwardDynamics(layer; dynamics_algorithm = layer.validation_algorithm).algorithm)
    return Processes.Process(
        algorithm,
        Processes.Init(:_state;
            x = zeros(eltype(graph), length(layer.input_layer)),
            equilibrium_state = copy(II.state(graph)),
        ),
        Processes.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Evaluate accuracy and squared error using free-phase input-field dynamics."""
function evaluate(worker::W, x::X, y::Y, config::C) where {W,X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig}
    correct = 0
    squared_error = zero(FT)
    pred_counts = zeros(Int, NCLASSES)
    ctx = Processes.context(worker)
    output = ctx._state.equilibrium_state
    output_idxs = II.layerrange(ctx.dynamics.model[end])

    for sample_idx in axes(x, 2)
        ctx._state.x .= view(x, :, sample_idx)
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)

        out = @view output[output_idxs]
        target = view(y, :, sample_idx)
        pred = argmax(class_scores(out, config.output_replicas))
        truth = argmax(class_scores(target, config.output_replicas))
        pred_counts[pred] += 1
        correct += pred == truth
        squared_error += sum(abs2, out .- target)
    end
    nsamples = size(x, 2)
    return (; accuracy = correct / nsamples, loss = squared_error / nsamples, pred_counts)
end

"""Serialize optimizer-facing parameters and run metadata."""
function save_checkpoint(path::P, manager::M, config::C, rows::R) where {P<:AbstractString,M<:Processes.ProcessManager,C<:InputFieldMNISTConfig,R<:AbstractVector}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (;
            architecture = "784-$(config.hidden)-$(NCLASSES * config.output_replicas)",
            params = manager.state.params[],
            opt_state = manager.state.opt_state,
            rows,
            config,
        ))
    end
    return path
end

"""Plot train/test accuracy and loss curves for a retained run."""
function plot_metrics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    fig = Figure(size = (1200, 760))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "MNIST 784-120-40 baseline accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Mean squared output error")
    train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
    lines!(ax_acc, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], label = "train", color = :steelblue)
    test_rows = [row for row in rows if !ismissing(row.test_accuracy)]
    lines!(ax_acc, [row.epoch for row in test_rows], [row.test_accuracy for row in test_rows], label = "test", color = :orange)
    lines!(ax_loss, [row.epoch for row in train_rows], [row.train_loss for row in train_rows], label = "train", color = :steelblue)
    lines!(ax_loss, [row.epoch for row in test_rows], [row.test_loss for row in test_rows], label = "test", color = :orange)
    axislegend(ax_acc, position = :rb)
    save(path, fig)
    return path
end

"""Write the run settings needed to reproduce the baseline."""
function write_settings!(path::P, config::C, relaxation_steps::I) where {P<:AbstractString,C<:InputFieldMNISTConfig,I<:Integer}
    open(path, "w") do io
        println(io, "# MNIST 784-120-40 Adam Baseline")
        println(io)
        println(io, "- paper: https://arxiv.org/pdf/2305.18321")
        println(io, "- architecture: `784 -> $(config.hidden) -> $(NCLASSES * config.output_replicas)`")
        println(io, "- input handling: MNIST pixels in `[0, 1]` are folded into a worker-local second `MagField`")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- worker graph adjacency: pointer-shared with source graph")
        println(io, "- learning step: one-sided free/nudged Process `LoopAlgorithm` from `@Routine`")
        println(io, "- optimiser: `Optimisers.Adam($(config.lr))`")
        println(io, "- epochs/batchsize: `$(config.epochs)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- train eval per class: `$(config.train_eval_per_class)`")
        println(io, "- sweeps/relaxation steps: `$(config.sweeps)` / `$(relaxation_steps)`")
        println(io, "- beta/temp/stepsize: `$(config.β)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- weight scale/decay: `$(config.weight_scale)` / `$(config.weight_decay)`")
    end
    return path
end

"""Run the full baseline experiment."""
function run_config!(config::C) where {C<:InputFieldMNISTConfig}
    config.workers > 0 || throw(ArgumentError("ISING_MNIST_IF_WORKERS must be positive"))
    config.batchsize > 0 || throw(ArgumentError("ISING_MNIST_IF_BATCHSIZE must be positive"))
    config.epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EPOCHS must be nonnegative"))
    config.hidden == 120 || @warn "baseline paper hidden count is 120" hidden = config.hidden
    config.output_replicas == 4 || @warn "baseline paper output count is 40, i.e. four replicas per digit" output_replicas = config.output_replicas
    config.train_per_class < 5421 && @warn "this run uses a subsampled balanced training split, so it is not the full balanced MNIST baseline" train_per_class = config.train_per_class
    config.test_per_class < 892 && @warn "this run uses a subsampled balanced test split, so reported accuracy will be noisy" test_per_class = config.test_per_class
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers

    mkpath(config.outdir)
    println("building MNIST 784-120-40 layer")
    flush(stdout)
    setup = build_layer(config)
    write_settings!(joinpath(config.outdir, "settings.md"), config, setup.relaxation_steps)
    println("loading training split")
    flush(stdout)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    println("loading train-eval split")
    flush(stdout)
    xtrain_eval, ytrain_eval = config.train_eval_per_class > 0 ?
        balanced_mnist(:train, config.train_eval_per_class, config) :
        (zeros(FT, INPUT_DIM, 0), zeros(FT, NCLASSES * config.output_replicas, 0))
    println("loading test split")
    flush(stdout)
    xtest, ytest = balanced_mnist(:test, config.test_per_class, config)
    println("constructing ", config.workers, "-worker Adam manager")
    flush(stdout)
    manager = input_field_manager(setup.layer, setup.graph, config)
    println("constructing validation worker")
    flush(stdout)
    validator = validation_worker(setup.layer, setup.graph)
    println("starting epochs")
    flush(stdout)

    csv_path = joinpath(config.outdir, "mnist_784_120_40_adam.csv")
    best_path = joinpath(config.outdir, "best_checkpoint.bin")
    final_path = joinpath(config.outdir, "final_checkpoint.bin")
    best_accuracy = Ref(-Inf)
    rows = NamedTuple[]

    try
        for epoch in 0:config.epochs
            seconds = 0.0
            train_accuracy = missing
            train_loss = missing
            test_accuracy = missing
            test_loss = missing
            pred_counts = missing
            if epoch > 0
                println("epoch ", epoch, " training")
                flush(stdout)
                order = Random.shuffle(Random.MersenneTwister(config.seed + epoch), collect(axes(xtrain, 2)))
                seconds = @elapsed begin
                    for first_idx in 1:config.batchsize:length(order)
                        last_idx = min(first_idx + config.batchsize - 1, length(order))
                        run_minibatch!(manager, batch_jobs(xtrain, ytrain, @view order[first_idx:last_idx]))
                    end
                end
            end

            # Validation is intentionally not run every epoch; on larger held-out sets it can dominate training time.
            should_eval_train = config.train_eval_per_class > 0 &&
                (epoch == 0 || epoch == config.epochs || (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0))
            if should_eval_train
                train = evaluate(validator, xtrain_eval, ytrain_eval, config)
                train_accuracy = train.accuracy
                train_loss = train.loss
            end
            should_eval_test = epoch == 0 || epoch == config.epochs ||
                (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0)
            if should_eval_test
                println("epoch ", epoch, " evaluating")
                flush(stdout)
                test = evaluate(validator, xtest, ytest, config)
                test_accuracy = test.accuracy
                test_loss = test.loss
                pred_counts = join(test.pred_counts, "-")
                if test.accuracy > best_accuracy[]
                    best_accuracy[] = test.accuracy
                    save_checkpoint(best_path, manager, config, rows)
                end
            end

            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                epoch,
                seconds,
                train_accuracy,
                train_loss,
                test_accuracy,
                test_loss,
                pred_counts,
                best_accuracy = best_accuracy[],
                best_path,
                final_path = epoch == config.epochs ? final_path : "",
            )
            append_row!(csv_path, row)
            push!(rows, row)
            println(row)
            flush(stdout)
        end

        save_checkpoint(final_path, manager, config, rows)
        plot_metrics(joinpath(config.outdir, "learning_summary.png"), rows)
        println("saved MNIST 784-120-40 baseline in ", config.outdir)
        return (; config, rows, best_accuracy = best_accuracy[])
    finally
        close(manager)
        close(validator)
    end
end

"""Entry point for command-line runs."""
function main()
    return run_config!(InputFieldMNISTConfig())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
