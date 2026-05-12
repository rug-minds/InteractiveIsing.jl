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
const WORKER_THREADS = min(REQUESTED_WORKER_THREADS, nthreads()-1)
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_TRAIN_EVAL_LIMIT", "128"))

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

rbm = ReducedBoltzmannArchitecture(784, 100, 10; precision = Float32)
rbm_copy = GraphFromSource(rbm)

layer = LayeredIsingGraphLayer(
    () -> ReducedBoltzmannArchitecture(784, 100, 10; precision = Float32);
    input_idxs = layerrange(rbm_copy[1]),
    output_idxs = layerrange(rbm_copy[end]),
)

optimiser = Optimisers.Descent(1f-3)

println("Running threaded MNIST example with $(WORKER_THREADS) worker threads on $(nthreads()) Julia threads for $(EPOCHS) epochs")

params, stats = fit_mnist_threaded!(
    layer;
    graph = rbm,
    numthreads = WORKER_THREADS,
    optimiser = optimiser,
    epochs = EPOCHS,
    batchsize = 128,
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
