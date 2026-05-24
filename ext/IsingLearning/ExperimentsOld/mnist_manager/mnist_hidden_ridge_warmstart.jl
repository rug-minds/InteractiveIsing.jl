using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using LinearAlgebra
using Optimisers
using Random
using Statistics

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_WORKERS", "15"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_HIDDEN", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_OUTPUT_REPLICAS", "4"))
const TRAIN_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_TRAIN_PER_CLASS", "100"))
const VALIDATION_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_VALIDATION_PER_CLASS", "50"))
const EP_EPOCHS = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_EP_EPOCHS", "0"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_BATCHSIZE", "64"))
const MINIT = parse(Int, get(ENV, "ISING_MNIST_RIDGE_WARM_MINIT", "1"))
const FREE_SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_RIDGE_WARM_FREE_SWEEPS", "100"))
const NUDGED_SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_RIDGE_WARM_NUDGED_SWEEPS", "50"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_STEPSIZE", "0.05"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_BETA", "0.1"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_LR", "0.0005"))
const RIDGE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_LAMBDA", "1e-2"))
const TARGET_ON = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_TARGET_ON", "0.5"))
const TARGET_OFF = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_TARGET_OFF", "0.0"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_WEIGHT_SCALE", "0.005"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_BIAS_SCALE", "0.02"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_WARM_TEMP", "0.001"))
const OUTDIR = get(ENV, "ISING_MNIST_RIDGE_WARM_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_hidden_ridge_warm")))

mkpath(OUTDIR)

"""
    balanced_mnist_arrays(layer; split, per_class, rng)

Return balanced MNIST arrays using the target coding requested for this
warm-start experiment.
"""
function balanced_mnist_arrays(layer::L; split::Symbol, per_class::Integer, rng = Random.MersenneTwister(1)) where {L}
    xraw, yraw = load_mnist_arrays(layer; split, limit = nothing)
    buckets = [Int[] for _ in 1:10]
    for sample_idx in axes(yraw, 2)
        digit = argmax(IsingLearning._mnist_class_scores(view(yraw, :, sample_idx)))
        push!(buckets[digit], sample_idx)
    end

    keep = Int[]
    sizehint!(keep, 10 * Int(per_class))
    for digit in 1:10
        Random.shuffle!(rng, buckets[digit])
        append!(keep, @view buckets[digit][1:min(Int(per_class), length(buckets[digit]))])
    end
    Random.shuffle!(rng, keep)

    x = copy(xraw[:, keep])
    y = copy(yraw[:, keep])
    y .= ifelse.(y .> 0, TARGET_ON, TARGET_OFF)
    return x, y
end

"""
    active_units(graph)

Return the active non-input spin count used to convert sweeps to steps.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    init_biases!(graph, seed)

Initialize graph biases with a deterministic small Gaussian perturbation.
"""
function init_biases!(graph::G, seed::Integer) where {G}
    rng = Random.MersenneTwister(seed)
    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

"""
    build_trainer()

Build the 784 -> hidden -> 40 MNIST trainer used by the warm-start probe.
"""
function build_trainer()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(11),
    )
    init_biases!(graph, 12)
    temp!(graph, TEMP)
    free_relaxation = max(1, round(Int, FREE_SWEEPS * active_units(graph)))
    nudged_relaxation = max(1, round(Int, NUDGED_SWEEPS * active_units(graph)))
    dynamics = LocalLangevin(
        stepsize = STEPSIZE,
        max_drift_fraction = 0.15f0,
        adjusted = false,
        order = :cyclic,
    )
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        free_relaxation_steps = free_relaxation,
        nudged_relaxation_steps = nudged_relaxation,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = WORKERS,
        optimiser = Optimisers.Adam(LR),
        share_static_model_data = true,
        input_mode = :field,
    )
    return graph, layer, trainer
end

"""
    collect_hidden_states!(trainer, x)

Run free inference for each sample and collect the relaxed hidden-layer states.
"""
function collect_hidden_states!(trainer::T, x::X) where {T,X<:AbstractMatrix}
    hidden_idxs = collect(InteractiveIsing.layerrange(trainer.validation_graph[2]))
    features = Matrix{Float32}(undef, length(hidden_idxs), size(x, 2))
    for sample_idx in axes(x, 2)
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        features[:, sample_idx] .= @view state(trainer.validation_graph)[hidden_idxs]
    end
    return features
end

"""
    fit_ridge(features, targets)

Fit a linear readout from hidden features to repeated MNIST targets.
"""
function fit_ridge(features::F, targets::Y) where {F<:AbstractMatrix,Y<:AbstractMatrix}
    xb = vcat(features, ones(Float32, 1, size(features, 2)))
    gram = xb * xb' + RIDGE * I
    coeff = (targets * xb') / gram
    return Matrix{Float32}(coeff[:, 1:end-1]), Vector{Float32}(coeff[:, end])
end

"""
    install_readout!(graph, W, b)

Write the fitted hidden-to-output readout into the Ising graph adjacency and
output magnetic fields.
"""
function install_readout!(graph::G, W::WMat, b::BVec) where {G,WMat<:AbstractMatrix,BVec<:AbstractVector}
    hidden_idxs = collect(InteractiveIsing.layerrange(graph[2]))
    output_idxs = collect(InteractiveIsing.layerrange(graph[end]))
    A = InteractiveIsing.adj(graph)
    for output_pos in eachindex(output_idxs)
        output_idx = output_idxs[output_pos]
        for hidden_pos in eachindex(hidden_idxs)
            hidden_idx = hidden_idxs[hidden_pos]
            A[output_idx, hidden_idx] = W[output_pos, hidden_pos]
            A[hidden_idx, output_idx] = W[output_pos, hidden_pos]
        end
    end
    bias = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    bias[output_idxs] .= b
    return graph
end

"""
    evaluate!(trainer, x, y)

Evaluate accuracy, MSE, and prediction histogram using the validation worker.
"""
function evaluate!(trainer::T, x::X, y::Y) where {T,X<:AbstractMatrix,Y<:AbstractMatrix}
    ncorrect = 0
    total_mse = 0f0
    pred_counts = zeros(Int, 10)
    target_counts = zeros(Int, 10)
    for sample_idx in axes(x, 2)
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        output = IsingLearning._validation_output(trainer)
        target = view(y, :, sample_idx)
        pred_digit = argmax(IsingLearning._mnist_class_scores(output)) - 1
        target_digit = argmax(IsingLearning._mnist_class_scores(target)) - 1
        pred_counts[pred_digit + 1] += 1
        target_counts[target_digit + 1] += 1
        ncorrect += pred_digit == target_digit
        total_mse += sum(abs2, output .- target)
    end
    return (accuracy = ncorrect / size(x, 2), mse = total_mse / size(x, 2), pred_counts, target_counts)
end

"""
    run_minibatch!(trainer, xbatch, ybatch, batch_gradient)

Run one stock MNIST contrastive minibatch from the warm-started trainer.
"""
function run_minibatch!(trainer::T, xbatch::X, ybatch::Y, batch_gradient::G) where {T,X<:AbstractMatrix,Y<:AbstractMatrix,G}
    IsingLearning._reset_batch_buffers!(trainer)
    jobs = NamedTuple[]
    sizehint!(jobs, size(xbatch, 2) * MINIT)
    for sample_idx in axes(xbatch, 2)
        for _ in 1:MINIT
            push!(jobs, (; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)))
        end
    end
    run!(trainer.manager, jobs)
    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, length(jobs))
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    return nothing
end

"""
    run_ep_epochs!(trainer, xtrain, ytrain, xval, yval, io)

Continue training from the ridge warm-start using the stock process-manager EP
minibatch loop.
"""
function run_ep_epochs!(
    trainer::T,
    xtrain::X,
    ytrain::Y,
    xval::VX,
    yval::VY,
    io::IO,
) where {T,X<:AbstractMatrix,Y<:AbstractMatrix,VX<:AbstractMatrix,VY<:AbstractMatrix}
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    for epoch in 1:EP_EPOCHS
        loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(60_000 + epoch))
        seconds = @elapsed begin
            for (xbatch, ybatch) in loader
                run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
            end
        end
        validation = evaluate!(trainer, xval, yval)
        println(io, "ep_epoch_", epoch, "=", validation, " seconds=", seconds)
        println("ep_epoch=", epoch, " validation=", validation, " seconds=", seconds)
        flush(stdout)
    end
    return nothing
end

"""
    main()

Run the hidden-state ridge warm-start diagnostic and optionally continue with
the stock EP trainer for a few epochs.
"""
function main()
    graph, layer, trainer = build_trainer()
    xtrain, ytrain = balanced_mnist_arrays(layer; split = :train, per_class = TRAIN_PER_CLASS, rng = Random.MersenneTwister(21))
    xval, yval = balanced_mnist_arrays(layer; split = :test, per_class = VALIDATION_PER_CLASS, rng = Random.MersenneTwister(22))

    try
        before = evaluate!(trainer, xval, yval)
        features = collect_hidden_states!(trainer, xtrain)
        W, b = fit_ridge(features, ytrain)
        install_readout!(graph, W, b)
        IsingLearning.sync_graph_params!(trainer.prototype_graph, IsingLearning.read_graph_params(graph))
        trainer.params = IsingLearning.read_graph_params(graph)
        trainer.opt_state = Optimisers.setup(trainer.optimiser, trainer.params)
        IsingLearning._broadcast_params!(trainer)
        after = evaluate!(trainer, xval, yval)

        summary_path = joinpath(OUTDIR, "hidden_ridge_warmstart_summary.txt")
        open(summary_path, "w") do io
            println(io, "hidden=", HIDDEN)
            println(io, "output_replicas=", OUTPUT_REPLICAS)
            println(io, "train_per_class=", TRAIN_PER_CLASS)
            println(io, "validation_per_class=", VALIDATION_PER_CLASS)
            println(io, "free_sweeps=", FREE_SWEEPS)
            println(io, "nudged_sweeps=", NUDGED_SWEEPS)
            println(io, "stepsize=", STEPSIZE)
            println(io, "beta=", BETA)
            println(io, "ridge=", RIDGE)
            println(io, "target_on=", TARGET_ON)
            println(io, "target_off=", TARGET_OFF)
            println(io, "before=", before)
            println(io, "after_ridge=", after)
            println(io, "ep_epochs=", EP_EPOCHS)
            EP_EPOCHS > 0 && run_ep_epochs!(trainer, xtrain, ytrain, xval, yval, io)
        end
        IsingLearning.InteractiveIsing.save_isinggraph(joinpath(OUTDIR, "final_graph.jld2"), trainer.prototype_graph)
        println(read(summary_path, String))
        println("Saved hidden ridge warm-start in ", OUTDIR)
    finally
        close_trainer!(trainer)
    end
end

main()
