using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random
using Base.Threads

Random.seed!(1234)

const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_EPOCHS", "200"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_LIMIT", "1024"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_VALIDATION_LIMIT", "256"))
const VALIDATION_SPLIT = Symbol(get(ENV, "ISING_MNIST_VALIDATION_SPLIT", "test"))
const REQUESTED_WORKER_THREADS = parse(Int, get(ENV, "ISING_MNIST_WORKER_THREADS", "16"))
const WORKER_THREADS = max(1, min(REQUESTED_WORKER_THREADS, nthreads()))
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_TRAIN_EVAL_LIMIT", "128"))
const MNIST_HIDDEN = parse(Int, get(ENV, "ISING_MNIST_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_BATCHSIZE", "256"))
const MNIST_TEMP = parse(Float32, get(ENV, "ISING_MNIST_TEMP", "0.005"))
const MNIST_STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_STEPSIZE", "0.4"))
const MNIST_BETA = parse(Float32, get(ENV, "ISING_MNIST_BETA", "2.0"))
const MNIST_FREE_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_FREE_RELAXATION", "600"))
const MNIST_NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_NUDGED_RELAXATION", "600"))
const MNIST_LR = parse(Float32, get(ENV, "ISING_MNIST_LR", "0.003"))

function plot_error_curves(stats; validation_label::AbstractString = string(VALIDATION_SPLIT))
    epochs = [entry.epoch for entry in stats]
    train_error = [entry.train.classification_error for entry in stats]
    train_mse = [entry.train.mean_squared_error for entry in stats]

    validation = [entry.validation for entry in stats if !isnothing(entry.validation)]
    validation_epochs = [stats[idx].epoch for idx in eachindex(stats) if !isnothing(stats[idx].validation)]
    validation_error = [entry.classification_error for entry in validation]
    validation_mse = [entry.mean_squared_error for entry in validation]

    fig = Figure(size = (900, 700))

    ax_error = Axis(
        fig[1, 1];
        title = "MNIST Classification Error",
        xlabel = "Epoch",
        ylabel = "Error",
    )
    lines!(ax_error, epochs, train_error; label = "train", linewidth = 3)
    isempty(validation_error) || lines!(ax_error, validation_epochs, validation_error; label = validation_label, linewidth = 3)
    axislegend(ax_error; position = :rt)

    ax_mse = Axis(
        fig[2, 1];
        title = "MNIST Mean Squared Error",
        xlabel = "Epoch",
        ylabel = "MSE",
    )
    lines!(ax_mse, epochs, train_mse; label = "train", linewidth = 3)
    isempty(validation_mse) || lines!(ax_mse, validation_epochs, validation_mse; label = validation_label, linewidth = 3)
    axislegend(ax_mse; position = :rt)

    return fig
end

rbm = MNISTArchitecture(hidden = MNIST_HIDDEN, precision = Float32)
temp!(rbm, MNIST_TEMP)

dynamics = LocalLangevin(stepsize = MNIST_STEPSIZE, adjusted = false)
layer = MNISTLayer(
    graph = rbm,
    β = MNIST_BETA,
    free_relaxation_steps = MNIST_FREE_RELAXATION,
    nudged_relaxation_steps = MNIST_NUDGED_RELAXATION,
    dynamics_algorithm = dynamics,
    nudged_dynamics_algorithm = deepcopy(dynamics),
    validation_algorithm = deepcopy(dynamics),
)

optimiser = Optimisers.Descent(MNIST_LR)

println("Running manager MNIST example $(MNIST_INPUT_DIM) -> $(MNIST_HIDDEN) -> 10 with $(WORKER_THREADS) worker slots on $(nthreads()) Julia threads for $(EPOCHS) epochs")
println("Dynamics: LocalLangevin(T=$(MNIST_TEMP), stepsize=$(MNIST_STEPSIZE), β=$(MNIST_BETA), free/nudged=$(MNIST_FREE_RELAXATION)/$(MNIST_NUDGED_RELAXATION), lr=$(MNIST_LR))")

params, stats = fit_mnist_threaded!(
    layer;
    graph = rbm,
    numthreads = WORKER_THREADS,
    optimiser = optimiser,
    epochs = EPOCHS,
    batchsize = BATCHSIZE,
    shuffle = true,
    show_progress = true,
    limit = TRAIN_LIMIT,
    validation_split = VALIDATION_SPLIT,
    validation_limit = VALIDATION_LIMIT,
    show_validation_progress = true,
    train_eval_limit = TRAIN_EVAL_LIMIT,
    full_train_eval_every = nothing,
)

fig = plot_error_curves(stats)
plot_path = joinpath(@__DIR__, "threaded_mnist_errors.png")
save(plot_path, fig)

println("Finished.")
println("Epoch stats: ", stats)
println("Saved plot: ", plot_path)
println("Parameter sizes: ",
    (w = length(params.w), b = length(params.b), α = length(params.α)))
