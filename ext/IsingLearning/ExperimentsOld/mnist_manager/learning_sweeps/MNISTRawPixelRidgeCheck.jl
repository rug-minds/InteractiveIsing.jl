using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using LinearAlgebra
using Random

const FT = Float32
const TRAIN_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_RAW_RIDGE_TRAIN_PER_CLASS", "100"))
const VALIDATION_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_RAW_RIDGE_VALIDATION_PER_CLASS", "30"))
const RIDGE = parse(FT, get(ENV, "ISING_MNIST_RAW_RIDGE_LAMBDA", "1e-2"))

"""Return a class-balanced slice with the requested number of samples per digit."""
function balanced_slice(x::X, y::Y, per_class::Integer, rng::R) where {X<:AbstractMatrix,Y<:AbstractMatrix,R<:AbstractRNG}
    buckets = [Int[] for _ in 1:10]
    for sample_idx in axes(y, 2)
        digit = argmax(IsingLearning._mnist_class_scores(view(y, :, sample_idx)))
        push!(buckets[digit], sample_idx)
    end

    keep = Int[]
    sizehint!(keep, 10 * Int(per_class))
    for digit in 1:10
        Random.shuffle!(rng, buckets[digit])
        append!(keep, @view buckets[digit][1:Int(per_class)])
    end
    Random.shuffle!(rng, keep)

    targets = copy(y[:, keep])
    targets .= ifelse.(targets .> 0, one(FT), zero(FT))
    return copy(x[:, keep]), targets
end

"""Fit a direct raw-pixel ridge readout and report balanced validation accuracy."""
function main()
    graph = MNISTArchitecture(hidden = 120, output_replicas = 4, precision = FT)
    layer = MNISTLayer(graph = graph)
    xtrain_raw, ytrain_raw = load_mnist_arrays(layer; split = :train)
    xval_raw, yval_raw = load_mnist_arrays(layer; split = :test)
    xtrain, ytrain = balanced_slice(xtrain_raw, ytrain_raw, TRAIN_PER_CLASS, Random.MersenneTwister(1))
    xval, yval = balanced_slice(xval_raw, yval_raw, VALIDATION_PER_CLASS, Random.MersenneTwister(2))

    xtrain_aug = vcat(xtrain, ones(FT, 1, size(xtrain, 2)))
    weights = (ytrain * transpose(xtrain_aug)) / (xtrain_aug * transpose(xtrain_aug) + RIDGE * I)
    scores = weights * vcat(xval, ones(FT, 1, size(xval, 2)))

    ncorrect = 0
    for sample_idx in axes(xval, 2)
        pred = argmax(IsingLearning._mnist_class_scores(view(scores, :, sample_idx)))
        target = argmax(IsingLearning._mnist_class_scores(view(yval, :, sample_idx)))
        ncorrect += pred == target
    end
    println("raw_pixel_ridge_accuracy=", ncorrect / size(xval, 2))
    println("train_per_class=", TRAIN_PER_CLASS, " validation_per_class=", VALIDATION_PER_CLASS, " ridge=", RIDGE)
    return nothing
end

main()
