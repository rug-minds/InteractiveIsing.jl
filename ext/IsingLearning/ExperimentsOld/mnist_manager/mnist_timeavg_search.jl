using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Optimisers
using Random
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_TAVG_WORKERS", "15"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_TAVG_HIDDEN", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_TAVG_OUTPUT_REPLICAS", "4"))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_TAVG_EPOCHS", "4"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_TAVG_BATCHSIZE", "64"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_TAVG_TRAIN_LIMIT", "1024"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_TAVG_VALIDATION_LIMIT", "256"))
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_TAVG_TRAIN_EVAL_LIMIT", "256"))
const FREE_SWEEPS = parse.(Float64, split(get(ENV, "ISING_MNIST_TAVG_FREE_SWEEPS", "150,200"), ","))
const NUDGED_BURNIN_SWEEPS = parse.(Float64, split(get(ENV, "ISING_MNIST_TAVG_NUDGED_BURNIN_SWEEPS", "50,100"), ","))
const NUDGED_SAMPLE_INTERVAL_SWEEPS = parse.(Float64, split(get(ENV, "ISING_MNIST_TAVG_NUDGED_INTERVAL_SWEEPS", "25"), ","))
const NUDGED_SAMPLES = parse.(Int, split(get(ENV, "ISING_MNIST_TAVG_NUDGED_SAMPLES", "1,3"), ","))
const STEPSIZES = parse.(Float32, split(get(ENV, "ISING_MNIST_TAVG_STEPSIZES", "0.05"), ","))
const BETAS = parse.(Float32, split(get(ENV, "ISING_MNIST_TAVG_BETAS", "0.05,0.1"), ","))
const LRS = parse.(Float32, split(get(ENV, "ISING_MNIST_TAVG_LRS", "0.0005,0.001"), ","))
const TARGET_SCALES = parse.(Float32, split(get(ENV, "ISING_MNIST_TAVG_TARGET_SCALES", "0.2,0.5"), ","))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_TAVG_WEIGHT_SCALE", "0.005"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_TAVG_BIAS_SCALE", "0.02"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "ISING_MNIST_TAVG_WEIGHT_DECAY", "1e-4"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_TAVG_TEMP", "0.001"))
const MINIT = parse(Int, get(ENV, "ISING_MNIST_TAVG_MINIT", "1"))
const OUTDIR = get(ENV, "ISING_MNIST_TAVG_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_timeavg")))

mkpath(OUTDIR)

"""
    MNISTTimeAveragedStep(layer; nudged_burnin_steps, nudged_sample_interval_steps, nudged_samples)

Local experiment worker algorithm for MNIST EP. It keeps the usual free phase,
then estimates the plus/minus nudged gradient from several time-separated
nudged states instead of only the final state.
"""
struct MNISTTimeAveragedStep{D,N,T} <: ProcessAlgorithm
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    β::T
    input_dim::Int
    output_dim::Int
    free_relaxation_steps::Int
    nudged_burnin_steps::Int
    nudged_sample_interval_steps::Int
    nudged_samples::Int
end

function MNISTTimeAveragedStep(
    layer::LayeredIsingGraphLayer;
    nudged_burnin_steps::Integer,
    nudged_sample_interval_steps::Integer,
    nudged_samples::Integer,
)
    nudged_samples > 0 || throw(ArgumentError("nudged_samples must be positive"))
    return MNISTTimeAveragedStep(
        deepcopy(layer.dynamics_algorithm),
        deepcopy(layer.nudged_dynamics_algorithm),
        layer.β,
        length(layer.input_layer),
        length(layer.output_layer),
        layer.free_relaxation_steps,
        Int(nudged_burnin_steps),
        Int(nudged_sample_interval_steps),
        Int(nudged_samples),
    )
end

"""
    StatefulAlgorithms.init(step::MNISTTimeAveragedStep, context)

Allocate persistent worker state for the time-averaged MNIST worker.
"""
function StatefulAlgorithms.init(step::MNISTTimeAveragedStep, context)
    model = context.model
    T = eltype(model)
    nstate = length(state(model))
    x = get(context, :x, zeros(T, step.input_dim))
    y = get(context, :y, zeros(T, step.output_dim))
    buffers = get(context, :buffers, IsingLearning.gradient_buffer(model))
    local_buffers = get(context, :local_buffers, IsingLearning.gradient_buffer(model))
    equilibrium_state = get(context, :equilibrium_state, copy(state(model)))
    plus_samples = get(context, :plus_samples, zeros(T, nstate, step.nudged_samples))
    minus_samples = get(context, :minus_samples, zeros(T, nstate, step.nudged_samples))
    free_context = StatefulAlgorithms.init(step.dynamics_algorithm, (; model))
    nudged_context = StatefulAlgorithms.init(step.nudged_dynamics_algorithm, (; model))
    return (; model, x, y, buffers, local_buffers, equilibrium_state, plus_samples, minus_samples, free_context, nudged_context)
end

"""
    relax_steps!(algorithm, context, nsteps)

Run a fixed number of single-spin process steps.
"""
function relax_steps!(algorithm::A, context::C, nsteps::Integer) where {A,C}
    for _ in 1:nsteps
        StatefulAlgorithms.step!(algorithm, context)
    end
    return context
end

"""
    record_nudged_samples!(dest, step, context, signed_beta)

Start from `context.equilibrium_state`, apply a signed nudge, and record nudged
state samples into columns of `dest`.
"""
function record_nudged_samples!(dest, step::MNISTTimeAveragedStep, context, signed_beta)
    model = context.model
    state(model) .= context.equilibrium_state
    IsingLearning.apply_input(model, context.x)
    IsingLearning.apply_targets(model, context.y)
    IsingLearning.set_clamping_beta!(model, signed_beta)
    relax_steps!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_burnin_steps)
    for sample_idx in 1:step.nudged_samples
        relax_steps!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_sample_interval_steps)
        @views dest[:, sample_idx] .= state(model)
    end
    return dest
end

"""
    add_scaled_buffer!(dest, src, scale)

Add `scale * src` into `dest` for all parameter buffers present in the graph.
"""
function add_scaled_buffer!(dest, src, scale)
    dest.w .+= scale .* src.w
    dest.b .+= scale .* src.b
    hasproperty(dest, :α) && (dest.α .+= scale .* src.α)
    return dest
end

"""
    StatefulAlgorithms.step!(step::MNISTTimeAveragedStep, context)

Accumulate one sample's time-averaged symmetric EP gradient into the persistent
worker buffer.
"""
function StatefulAlgorithms.step!(step::MNISTTimeAveragedStep, context)
    model = context.model
    β = step.β

    resetstate!(model)
    IsingLearning.apply_input(model, context.x)
    IsingLearning.set_clamping_beta!(model, zero(β))
    relax_steps!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
    context.equilibrium_state .= state(model)

    record_nudged_samples!(context.plus_samples, step, context, β)
    record_nudged_samples!(context.minus_samples, step, context, -β)
    IsingLearning.set_clamping_beta!(model, zero(β))

    IsingLearning.zero_buffer!(context.local_buffers)
    for sample_idx in 1:step.nudged_samples
        IsingLearning.contrastive_gradient(
            model,
            view(context.plus_samples, :, sample_idx),
            view(context.minus_samples, :, sample_idx),
            β;
            buffers = context.local_buffers,
        )
    end
    add_scaled_buffer!(context.buffers, context.local_buffers, inv(eltype(context.local_buffers.w)(step.nudged_samples)))
    return nothing
end

function StatefulAlgorithms.cleanup(::MNISTTimeAveragedStep, context)
    return nothing
end

mutable struct LocalMNISTTrainer{L,G,P,S,W<:Process,O,M}
    layer::L
    prototype_graph::G
    params::P
    opt_state::S
    worker_graphs::Vector{G}
    workers::Vector{W}
    validation_graph::G
    optimiser::O
    manager::M
end

function append_row!(path::AbstractString, row)
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

function active_units(graph)
    return length(II.layerrange(graph[2])) + length(II.layerrange(graph[end]))
end

function init_biases!(graph, seed::Integer)
    rng = Random.MersenneTwister(seed)
    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

function make_jobs(xbatch, ybatch)
    first_job = (; x = view(xbatch, :, first(axes(xbatch, 2))), y = view(ybatch, :, first(axes(ybatch, 2))))
    jobs = typeof(first_job)[]
    sizehint!(jobs, size(xbatch, 2) * MINIT)
    for sample_idx in axes(xbatch, 2)
        for _ in 1:MINIT
            push!(jobs, (; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)))
        end
    end
    return jobs
end

function worker_graph(prototype_graph, params)
    graph = deepcopy(prototype_graph)
    IsingLearning.sync_graph_params!(graph, params)
    II.temp!(graph, II.temp(prototype_graph))
    return graph
end

function worker_process(layer, graph, nudged_burnin_steps, nudged_sample_interval_steps, nudged_samples)
    step = :_state => MNISTTimeAveragedStep(
        layer;
        nudged_burnin_steps,
        nudged_sample_interval_steps,
        nudged_samples,
    )
    Process(
        step,
        Init(:_state;
            model = graph,
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            buffers = IsingLearning.gradient_buffer(graph),
            local_buffers = IsingLearning.gradient_buffer(graph),
            equilibrium_state = copy(state(graph)),
        );
        repeat = 1,
    )
end

worker_state(worker) = StatefulAlgorithms.context(worker)._state

function write_example!(worker, x, y)
    st = worker_state(worker)
    st.x .= x
    st.y .= y
    return st
end

function training_manager(layer, graph, params, nworkers, nudged_burnin_steps, nudged_sample_interval_steps, nudged_samples)
    recipe = (;
        makeworker = (idx, manager) -> worker_process(
            layer,
            worker_graph(graph, params),
            nudged_burnin_steps,
            nudged_sample_interval_steps,
            nudged_samples,
        ),
        prepare! = (slot, job, manager) -> begin
            write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(recipe; nworkers, flush_policy = NoFlush(), poll_interval = 0.0)
end

function zero_worker_buffers!(trainer)
    Threads.@threads for idx in eachindex(trainer.workers)
        IsingLearning.zero_buffer!(worker_state(trainer.workers[idx]).buffers)
    end
    return trainer
end

function collect_gradient!(trainer, dest, njobs::Integer)
    β = trainer.layer.β
    scale = inv(eltype(dest.w)(2) * eltype(dest.w)(β) * eltype(dest.w)(njobs))
    buffers = [worker_state(worker).buffers for worker in trainer.workers]
    fill!(dest.w, zero(eltype(dest.w)))
    for buffer in buffers
        dest.w .+= buffer.w
    end
    dest.w .*= scale
    fill!(dest.b, zero(eltype(dest.b)))
    for buffer in buffers
        dest.b .+= buffer.b
    end
    dest.b .*= scale
    if hasproperty(dest, :α)
        fill!(dest.α, zero(eltype(dest.α)))
        for buffer in buffers
            dest.α .+= buffer.α
        end
        dest.α .*= scale
    end
    return dest
end

function add_weight_decay!(gradient, params)
    WEIGHT_DECAY > 0 || return gradient
    gradient.w .+= WEIGHT_DECAY .* params.w
    return gradient
end

function broadcast_params!(trainer)
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    Threads.@threads for idx in eachindex(trainer.worker_graphs)
        IsingLearning.sync_graph_params!(trainer.worker_graphs[idx], trainer.params)
    end
    IsingLearning.sync_graph_params!(trainer.validation_graph, trainer.params)
    return trainer
end

function run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
    zero_worker_buffers!(trainer)
    jobs = make_jobs(xbatch, ybatch)
    run!(trainer.manager, jobs)
    collect_gradient!(trainer, batch_gradient, length(jobs))
    add_weight_decay!(batch_gradient, trainer.params)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    broadcast_params!(trainer)
    return batch_gradient
end

function scale_targets!(y, target_scale::Float32)
    y .= ifelse.(y .> 0, target_scale, -target_scale)
    return y
end

function load_scaled_mnist(layer, target_scale; split, limit)
    xraw, yraw = load_mnist_arrays(layer; split, limit = nothing)
    last_idx = min(limit, size(xraw, 2))
    x = copy(xraw[:, 1:last_idx])
    y = copy(yraw[:, 1:last_idx])
    return x, scale_targets!(y, target_scale)
end

function evaluate!(trainer, x, y)
    nsamples = size(x, 2)
    ncorrect = 0
    total_squared_error = zero(eltype(x))
    dynamics = deepcopy(trainer.layer.validation_algorithm)
    context = StatefulAlgorithms.init(dynamics, (; model = trainer.validation_graph))
    for sample_idx in axes(x, 2)
        resetstate!(trainer.validation_graph)
        IsingLearning.apply_input(trainer.validation_graph, view(x, :, sample_idx))
        IsingLearning.set_clamping_beta!(trainer.validation_graph, zero(trainer.layer.β))
        relax_steps!(dynamics, context, trainer.layer.free_relaxation_steps)
        output = vec(state(trainer.validation_graph[end]))
        target = view(y, :, sample_idx)
        output_scores = IsingLearning._mnist_class_scores(output)
        target_scores = IsingLearning._mnist_class_scores(target)
        ncorrect += argmax(output_scores) == argmax(target_scores)
        total_squared_error += sum(abs2, output .- target)
    end
    accuracy = ncorrect / nsamples
    return (; accuracy, classification_error = 1 - accuracy, mean_squared_error = total_squared_error / nsamples, nsamples)
end

function strip_weight_generators!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

function save_graph(path, graph)
    mkpath(dirname(path))
    return II.save_isinggraph(path, strip_weight_generators!(deepcopy(graph)))
end

function build_trainer(config_id, free_sweeps, nudged_burnin_sweeps, nudged_interval_sweeps, nudged_samples, stepsize, beta, lr)
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(10_000 + config_id),
    )
    init_biases!(graph, 20_000 + config_id)
    II.temp!(graph, TEMP)
    nactive = active_units(graph)
    free_steps = max(1, round(Int, free_sweeps * nactive))
    burnin_steps = max(1, round(Int, nudged_burnin_sweeps * nactive))
    interval_steps = max(1, round(Int, nudged_interval_sweeps * nactive))
    dynamics = LocalLangevin(stepsize = stepsize, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = beta,
        free_relaxation_steps = free_steps,
        nudged_relaxation_steps = burnin_steps + nudged_samples * interval_steps,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), params)
    manager = training_manager(layer, graph, params, WORKERS, burnin_steps, interval_steps, nudged_samples)
    workers = collect(StatefulAlgorithms.workers(manager))
    worker_graphs = [worker_state(worker).model for worker in workers]
    validation_graph = worker_graph(graph, params)
    trainer = LocalMNISTTrainer(layer, graph, params, opt_state, worker_graphs, workers, validation_graph, Optimisers.Adam(lr), manager)
    return trainer, free_steps, burnin_steps, interval_steps
end

function close_trainer!(trainer)
    close(trainer.manager)
    return nothing
end

function run_config(config_id, free_sweeps, nudged_burnin_sweeps, nudged_interval_sweeps, nudged_samples, stepsize, beta, lr, target_scale)
    trainer, free_steps, burnin_steps, interval_steps = build_trainer(
        config_id,
        free_sweeps,
        nudged_burnin_sweeps,
        nudged_interval_sweeps,
        nudged_samples,
        stepsize,
        beta,
        lr,
    )
    csv_path = joinpath(OUTDIR, "mnist_timeavg_search.csv")
    checkpoint_dir = joinpath(OUTDIR, "checkpoints", "config_$(config_id)")
    initial_graph_path = save_graph(joinpath(checkpoint_dir, "initial_graph.jld2"), trainer.prototype_graph)
    best_graph_path = joinpath(checkpoint_dir, "best_graph.jld2")
    final_graph_path = joinpath(checkpoint_dir, "final_graph.jld2")
    best_validation_mse = Ref(Inf)
    best_validation_accuracy = Ref(0.0)
    best_epoch = Ref(0)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)

    xtrain, ytrain = load_scaled_mnist(trainer.layer, target_scale; split = :train, limit = TRAIN_LIMIT)
    xval, yval = load_scaled_mnist(trainer.layer, target_scale; split = :test, limit = VALIDATION_LIMIT)
    xtrain_eval = @view xtrain[:, 1:min(TRAIN_EVAL_LIMIT, size(xtrain, 2))]
    ytrain_eval = @view ytrain[:, 1:min(TRAIN_EVAL_LIMIT, size(ytrain, 2))]

    try
        for epoch in 0:EPOCHS
            seconds = 0.0
            if epoch > 0
                loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(100_000 * config_id + epoch))
                seconds = @elapsed begin
                    for (xbatch, ybatch) in loader
                        run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
                    end
                end
            end

            train_metrics = evaluate!(trainer, xtrain_eval, ytrain_eval)
            validation_metrics = evaluate!(trainer, xval, yval)
            if Float64(validation_metrics.mean_squared_error) < best_validation_mse[]
                best_validation_mse[] = Float64(validation_metrics.mean_squared_error)
                best_validation_accuracy[] = Float64(validation_metrics.accuracy)
                best_epoch[] = epoch
                save_graph(best_graph_path, trainer.prototype_graph)
            end
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                config_id,
                epoch,
                hidden = HIDDEN,
                output_replicas = OUTPUT_REPLICAS,
                workers = WORKERS,
                train_limit = TRAIN_LIMIT,
                validation_limit = VALIDATION_LIMIT,
                batchsize = BATCHSIZE,
                free_sweeps,
                nudged_burnin_sweeps,
                nudged_interval_sweeps,
                nudged_samples,
                free_steps,
                burnin_steps,
                interval_steps,
                stepsize,
                beta,
                lr,
                target_scale,
                minit = MINIT,
                temp = TEMP,
                weight_scale = WEIGHT_SCALE,
                weight_decay = WEIGHT_DECAY,
                seconds,
                train_accuracy = train_metrics.accuracy,
                train_mse = train_metrics.mean_squared_error,
                validation_accuracy = validation_metrics.accuracy,
                validation_mse = validation_metrics.mean_squared_error,
                best_epoch = best_epoch[],
                best_validation_mse = best_validation_mse[],
                best_validation_accuracy = best_validation_accuracy[],
                initial_graph_path,
                best_graph_path,
                final_graph_path = epoch == EPOCHS ? final_graph_path : "",
            )
            append_row!(csv_path, row)
            println(row)
            flush(stdout)
        end
        save_graph(final_graph_path, trainer.prototype_graph)
    finally
        close_trainer!(trainer)
    end
    return csv_path
end

function main()
    println(
        "MNIST timeavg search workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " output_replicas=", OUTPUT_REPLICAS,
        " epochs=", EPOCHS,
        " train_limit=", TRAIN_LIMIT,
        " validation_limit=", VALIDATION_LIMIT,
        " batchsize=", BATCHSIZE,
    )
    flush(stdout)

    config_id = 0
    for free_sweeps in FREE_SWEEPS,
        nudged_burnin_sweeps in NUDGED_BURNIN_SWEEPS,
        nudged_interval_sweeps in NUDGED_SAMPLE_INTERVAL_SWEEPS,
        nudged_samples in NUDGED_SAMPLES,
        stepsize in STEPSIZES,
        beta in BETAS,
        lr in LRS,
        target_scale in TARGET_SCALES

        config_id += 1
        run_config(config_id, free_sweeps, nudged_burnin_sweeps, nudged_interval_sweeps, nudged_samples, stepsize, beta, lr, target_scale)
    end
    println("Saved MNIST time-average search in ", OUTDIR)
end

main()
