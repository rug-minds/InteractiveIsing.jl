using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using LinearAlgebra
using Random
using Statistics

const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_RIDGE_HIDDEN", string(10 * 28^2)))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_RIDGE_TRAIN_LIMIT", "10000"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_RIDGE_VALIDATION_LIMIT", "2000"))
const RIDGE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_LAMBDA", "1e-2"))
const COPY_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_COPY_SCALE", "1.0"))
const OUTPUT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_OUTPUT_SCALE", "0.2"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_TEMP", "0.0"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_RIDGE_STEPSIZE", "0.1"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_RIDGE_SWEEPS", "1.0"))
const OUTDIR = get(ENV, "ISING_MNIST_RIDGE_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_ridge_init")))

mkpath(OUTDIR)

function load_xy(layer; split, limit)
    x, y = load_mnist_arrays(layer; split, limit)
    return Matrix{Float32}(x), Matrix{Float32}(y)
end

function fit_ridge(x, y)
    xb = vcat(x, ones(Float32, 1, size(x, 2)))
    target = ifelse.(y .> 0, 1f0, -1f0)
    gram = xb * xb' + RIDGE * I
    coeff = (target * xb') / gram
    return Matrix{Float32}(coeff[:, 1:end-1]), Vector{Float32}(coeff[:, end])
end

function init_pixel_copy_hidden!(graph)
    input_idxs = collect(InteractiveIsing.layerrange(graph[1]))
    hidden_idxs = collect(InteractiveIsing.layerrange(graph[2]))
    A = InteractiveIsing.adj(graph)
    for hidden_idx in hidden_idxs, input_idx in input_idxs
        A[hidden_idx, input_idx] = zero(eltype(graph))
    end
    for (hidden_pos, hidden_idx) in enumerate(hidden_idxs)
        input_idx = input_idxs[mod1(hidden_pos, length(input_idxs))]
        A[hidden_idx, input_idx] = COPY_SCALE
    end
    return graph
end

function init_ridge_readout!(graph, W, b)
    input_dim = size(W, 2)
    hidden_idxs = collect(InteractiveIsing.layerrange(graph[2]))
    output_idxs = collect(InteractiveIsing.layerrange(graph[end]))
    A = InteractiveIsing.adj(graph)
    copies_per_pixel = zeros(Int, input_dim)
    for hidden_pos in eachindex(hidden_idxs)
        copies_per_pixel[mod1(hidden_pos, input_dim)] += 1
    end
    for output_idx in output_idxs, hidden_idx in hidden_idxs
        A[output_idx, hidden_idx] = zero(eltype(graph))
    end
    for (hidden_pos, hidden_idx) in enumerate(hidden_idxs)
        pixel = mod1(hidden_pos, input_dim)
        scale = OUTPUT_SCALE / copies_per_pixel[pixel]
        for (out_pos, output_idx) in enumerate(output_idxs)
            A[output_idx, hidden_idx] = scale * W[out_pos, pixel]
        end
    end
    bias = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    fill!(bias, zero(eltype(bias)))
    bias[output_idxs] .= OUTPUT_SCALE .* b
    return graph
end

function evaluate_forward!(trainer, x, y)
    ncorrect = 0
    total_mse = 0f0
    outputs = zeros(Float32, size(y, 1), size(y, 2))
    for sample_idx in axes(x, 2)
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        StatefulAlgorithms.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        output = IsingLearning._validation_output(trainer)
        outputs[:, sample_idx] .= output
        total_mse += sum(abs2, output .- view(y, :, sample_idx))
        ncorrect += argmax(output) == argmax(view(y, :, sample_idx))
    end
    return (accuracy = ncorrect / size(x, 2), mse = total_mse / size(x, 2), outputs = outputs)
end

function main()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        precision = Float32,
        weight_scale = 0f0,
        rng = Random.MersenneTwister(1234),
    )
    temp!(graph, TEMP)
    init_pixel_copy_hidden!(graph)

    relaxation = max(1, round(Int, SWEEPS * (length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end])))))
    dynamics = LocalLangevin(stepsize = STEPSIZE, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = 0.1f0,
        free_relaxation_steps = relaxation,
        nudged_relaxation_steps = relaxation,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )

    xtrain, ytrain = load_xy(layer; split = :train, limit = TRAIN_LIMIT)
    xval, yval = load_xy(layer; split = :test, limit = VALIDATION_LIMIT)
    W, b = fit_ridge(xtrain, ytrain)
    init_ridge_readout!(graph, W, b)

    trainer = init_mnist_trainer(layer; graph, numthreads = 1)
    try
        train = evaluate_forward!(trainer, xtrain[:, 1:min(1000, size(xtrain, 2))], ytrain[:, 1:min(1000, size(ytrain, 2))])
        validation = evaluate_forward!(trainer, xval, yval)
        summary_path = joinpath(OUTDIR, "mnist_ridge_init_summary.txt")
        open(summary_path, "w") do io
            println(io, "hidden=", HIDDEN)
            println(io, "train_limit=", TRAIN_LIMIT)
            println(io, "validation_limit=", VALIDATION_LIMIT)
            println(io, "ridge=", RIDGE)
            println(io, "copy_scale=", COPY_SCALE)
            println(io, "output_scale=", OUTPUT_SCALE)
            println(io, "relaxation=", relaxation)
            println(io, "train_accuracy=", train.accuracy)
            println(io, "train_mse=", train.mse)
            println(io, "validation_accuracy=", validation.accuracy)
            println(io, "validation_mse=", validation.mse)
            println(io, "validation_output_mean=", mean(validation.outputs))
            println(io, "validation_output_std=", std(validation.outputs))
        end
        println(read(summary_path, String))
        println("Saved ridge init smoke in ", OUTDIR)
    finally
        close_trainer!(trainer)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
