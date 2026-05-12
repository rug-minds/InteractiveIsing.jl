using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random
using Base.Threads

Random.seed!(1234)

function float_type_from_env(name::AbstractString, default::AbstractString)
    value = lowercase(get(ENV, name, default))
    value in ("float64", "64", "double") && return Float64
    value in ("float32", "32", "single") && return Float32
    error("$name must be Float64 or Float32, got `$value`")
end

const FT = float_type_from_env("ISING_TINY_MNIST_FLOAT_TYPE", "Float64")
const EPOCHS = parse(Int, get(ENV, "ISING_TINY_MNIST_EPOCHS", "5"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_TINY_MNIST_LIMIT", "128"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_TINY_MNIST_VALIDATION_LIMIT", "64"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_TINY_MNIST_BATCHSIZE", "32"))
const RELAXATION_STEPS = parse(Int, get(ENV, "ISING_TINY_MNIST_RELAXATION_STEPS", "20"))
const LEARNING_RATE = parse(FT, get(ENV, "ISING_TINY_MNIST_LEARNING_RATE", "1e-3"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_TINY_MNIST_WEIGHT_SCALE", "0.01"))
const REQUESTED_THREADS = parse(Int, get(ENV, "ISING_TINY_MNIST_WORKER_THREADS", "2"))
const WORKER_THREADS = max(1, min(REQUESTED_THREADS, max(1, nthreads() - 1)))

function small_weight_generator(seed::Integer = 4321)
    rng = Random.MersenneTwister(seed)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, FT))
end

function assert_finite_params(params)
    all(isfinite, params.w) || error("non-finite weights detected")
    all(isfinite, params.b) || error("non-finite biases detected")
    all(isfinite, params.α) || error("non-finite local potentials detected")
    return nothing
end

function assert_finite_metrics(stats)
    for epoch_stats in stats
        for split_stats in (epoch_stats.train, epoch_stats.validation)
            isnothing(split_stats) && continue
            isfinite(split_stats.accuracy) || error("non-finite accuracy detected")
            isfinite(split_stats.classification_error) || error("non-finite classification error detected")
            isfinite(split_stats.mean_squared_error) || error("non-finite MSE detected")
        end
    end
    return nothing
end

graph = ReducedBoltzmannArchitecture(784, 32, 10; precision = FT, weight_generator = small_weight_generator())
dynamics = GlobalLangevin(stepsize = FT(1e-3), adjusted = false, group_steps = 1)

layer = LayeredIsingGraphLayer(
    () -> ReducedBoltzmannArchitecture(784, 32, 10; precision = FT, weight_generator = small_weight_generator());
    input_idxs = layerrange(graph[1]),
    output_idxs = layerrange(graph[end]),
    β = FT(0.1),
    fullsweeps = 1,
    relaxation_steps = RELAXATION_STEPS,
    dynamics_algorithm = dynamics,
    validation_algorithm = deepcopy(dynamics),
)

println(
    "Running tiny MNIST smoke test: ",
    (epochs = EPOCHS, limit = TRAIN_LIMIT, validation_limit = VALIDATION_LIMIT,
     batchsize = BATCHSIZE, worker_threads = WORKER_THREADS,
     relaxation_steps = RELAXATION_STEPS, learning_rate = LEARNING_RATE,
     weight_scale = WEIGHT_SCALE, float_type = FT),
)

params, stats = fit_mnist_threaded!(
    layer;
    graph,
    numthreads = WORKER_THREADS,
    optimiser = Optimisers.Adam(LEARNING_RATE),
    epochs = EPOCHS,
    batchsize = BATCHSIZE,
    shuffle = true,
    show_progress = true,
    limit = TRAIN_LIMIT,
    validation_split = :test,
    validation_limit = VALIDATION_LIMIT,
    show_validation_progress = true,
    train_eval_limit = min(BATCHSIZE, TRAIN_LIMIT),
    full_train_eval_every = nothing,
)

assert_finite_params(params)
assert_finite_metrics(stats)
all(isfinite, state(graph)) || error("non-finite graph state detected")

println("Finished tiny MNIST smoke test.")
println("Final epoch: ", isempty(stats) ? nothing : stats[end])
println("Parameter summary: ", (
    w_norm = sqrt(sum(abs2, params.w)),
    b_norm = sqrt(sum(abs2, params.b)),
    α_norm = sqrt(sum(abs2, params.α)),
))
println("Gradient summary: unavailable from fit_mnist_threaded! return value")
