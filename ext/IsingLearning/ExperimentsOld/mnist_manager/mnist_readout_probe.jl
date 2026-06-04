using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using LinearAlgebra
using Random
using Statistics

const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_READOUT_HIDDEN", "512"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_READOUT_TRAIN_LIMIT", "2048"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_READOUT_VALIDATION_LIMIT", "1024"))
const DIGITS_RAW = get(ENV, "ISING_MNIST_READOUT_DIGITS", "0,1")
const DIGITS = parse.(Int, split(DIGITS_RAW, ","))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_READOUT_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_READOUT_STEPSIZE", "0.05"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_READOUT_SWEEPS", "2.0"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_READOUT_WEIGHT_SCALE", "0.01"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_READOUT_BIAS_SCALE", "0.02"))
const RIDGE = parse(Float64, get(ENV, "ISING_MNIST_READOUT_RIDGE", "1e-3"))
const OUTDIR = get(ENV, "ISING_MNIST_READOUT_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_readout")))

mkpath(OUTDIR)

function init_biases!(graph, seed)
    rng = Random.MersenneTwister(seed)
    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

function filtered_data(layer; split, limit)
    xraw, yraw = load_mnist_arrays(layer; split, limit = nothing)
    keep = Int[]
    for idx in axes(yraw, 2)
        digit = argmax(view(yraw, :, idx)) - 1
        digit in DIGITS && push!(keep, idx)
        length(keep) >= limit && break
    end
    return xraw[:, keep], yraw[:, keep]
end

function build_trainer()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(1234),
    )
    init_biases!(graph, 4321)
    temp!(graph, TEMP)
    relaxation = max(1, round(Int, SWEEPS * (length(layerrange(graph[2])) + length(layerrange(graph[end])))))
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
    trainer = init_mnist_trainer(layer; graph, numthreads = 1)
    return graph, layer, trainer, relaxation
end

function collect_features(trainer, x, y; feature)
    hidden_idxs = layerrange(trainer.layer.model_graph[2])
    output_idxs = trainer.layer.output_layer
    input_idxs = trainer.layer.input_layer
    nfeat = feature === :input ? length(input_idxs) :
        feature === :hidden ? length(hidden_idxs) :
        feature === :hidden_output ? length(hidden_idxs) + length(output_idxs) :
        feature === :input_hidden ? length(input_idxs) + length(hidden_idxs) :
        error("unknown feature $(feature)")

    X = Matrix{Float64}(undef, nfeat + 1, size(x, 2))
    labels = Vector{Int}(undef, size(x, 2))
    worker = trainer.validation_worker
    for sample_idx in axes(x, 2)
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        StatefulAlgorithms.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        s = StatefulAlgorithms.context(worker)._state.equilibrium_state
        if feature === :input
            X[1:nfeat, sample_idx] .= Float64.(view(s, input_idxs))
        elseif feature === :hidden
            X[1:nfeat, sample_idx] .= Float64.(view(s, hidden_idxs))
        elseif feature === :hidden_output
            X[1:length(hidden_idxs), sample_idx] .= Float64.(view(s, hidden_idxs))
            X[(length(hidden_idxs) + 1):nfeat, sample_idx] .= Float64.(view(s, output_idxs))
        elseif feature === :input_hidden
            X[1:length(input_idxs), sample_idx] .= Float64.(view(s, input_idxs))
            X[(length(input_idxs) + 1):nfeat, sample_idx] .= Float64.(view(s, hidden_idxs))
        end
        X[end, sample_idx] = 1.0
        labels[sample_idx] = argmax(view(y, :, sample_idx)) - 1
    end
    return X, labels
end

function onehot_labels(labels)
    Y = fill(-1.0, length(DIGITS), length(labels))
    for (idx, label) in pairs(labels)
        row = findfirst(==(label), DIGITS)
        Y[row, idx] = 1.0
    end
    return Y
end

function fit_ridge(X, labels)
    Y = onehot_labels(labels)
    gram = X * X' + RIDGE * I
    return (Y * X') / gram
end

function accuracy(W, X, labels)
    scores = W * X
    correct = 0
    for idx in axes(scores, 2)
        pred = DIGITS[argmax(view(scores, :, idx))]
        correct += pred == labels[idx]
    end
    return correct / length(labels)
end

graph, layer, trainer, relaxation = build_trainer()
xtrain, ytrain = filtered_data(layer; split = :train, limit = TRAIN_LIMIT)
xval, yval = filtered_data(layer; split = :test, limit = VALIDATION_LIMIT)

println(
    "readout probe hidden=", HIDDEN,
    " digits=", DIGITS,
    " train=", size(xtrain, 2),
    " val=", size(xval, 2),
    " relaxation=", relaxation,
    " temp=", TEMP,
)
flush(stdout)

rows = NamedTuple[]
for feature in (:input, :hidden, :hidden_output, :input_hidden)
    t = @elapsed begin
        Xtr, ytr = collect_features(trainer, xtrain, ytrain; feature)
        Xva, yva = collect_features(trainer, xval, yval; feature)
        W = fit_ridge(Xtr, ytr)
        train_acc = accuracy(W, Xtr, ytr)
        val_acc = accuracy(W, Xva, yva)
        push!(rows, (; feature, train_acc, val_acc, nfeat = size(Xtr, 1) - 1))
    end
    println(last(rows), " seconds=", round(t; digits = 3))
    flush(stdout)
end

open(joinpath(OUTDIR, "mnist_readout_probe.csv"), "w") do io
    names = propertynames(first(rows))
    println(io, join(names, ","))
    for row in rows
        println(io, join((getproperty(row, name) for name in names), ","))
    end
end

close_trainer!(trainer)
println("Saved readout probe in ", OUTDIR)
