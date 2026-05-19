using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random

const WORKER_COUNTS = parse.(Int, split(get(ENV, "ISING_MNIST_OVERNIGHT_WORKERS", "8,12,16"), ","))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_EPOCHS", "5"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_BATCHSIZE", "256"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const TRAIN_LIMIT_RAW = get(ENV, "ISING_MNIST_OVERNIGHT_LIMIT", "")
const TRAIN_LIMIT = isempty(TRAIN_LIMIT_RAW) ? nothing : parse(Int, TRAIN_LIMIT_RAW)
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_VALIDATION_LIMIT", "1024"))
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_TRAIN_EVAL_LIMIT", "1024"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_OVERNIGHT_TEMP", "0.005"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_OVERNIGHT_STEPSIZE", "0.4"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_OVERNIGHT_BETA", "2.0"))
const FREE_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_FREE", "600"))
const NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_OVERNIGHT_NUDGED", "600"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_OVERNIGHT_LR", "0.003"))
const OUTDIR = get(ENV, "ISING_MNIST_OVERNIGHT_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS")))

mkpath(OUTDIR)

function mnist_run_layer()
    graph = MNISTArchitecture(hidden = HIDDEN, precision = Float32)
    temp!(graph, TEMP)
    dynamics = LocalLangevin(stepsize = STEPSIZE, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        free_relaxation_steps = FREE_RELAXATION,
        nudged_relaxation_steps = NUDGED_RELAXATION,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return graph, layer
end

function append_row!(path, row)
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

function run_config(nworkers::Integer)
    graph, layer = mnist_run_layer()
    trainer = init_mnist_trainer(layer; graph, numthreads = nworkers, optimiser = Optimisers.Descent(LR))
    csv_path = joinpath(OUTDIR, "mnist_overnight_metrics.csv")

    println("Starting MNIST manager run workers=", nworkers,
        " epochs=", EPOCHS,
        " train_limit=", TRAIN_LIMIT,
        " validation_limit=", VALIDATION_LIMIT,
        " hidden=", HIDDEN)
    flush(stdout)

    try
        for epoch in 1:EPOCHS
            result_ref = Ref{Any}(nothing)
            t = @elapsed begin
                _, stats = fit_mnist_threaded!(
                    trainer;
                    epochs = 1,
                    batchsize = BATCHSIZE,
                    split = :train,
                    validation_split = :test,
                    shuffle = true,
                    rng = Random.MersenneTwister(10_000 * nworkers + epoch),
                    limit = TRAIN_LIMIT,
                    validation_limit = VALIDATION_LIMIT,
                    show_progress = true,
                    show_validation_progress = false,
                    log_metrics = true,
                    train_eval_limit = TRAIN_EVAL_LIMIT,
                    full_train_eval_every = nothing,
                )
                result_ref[] = only(stats)
            end
            result = result_ref[]
            train = result.train
            validation = result.validation
            append_row!(csv_path, (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                workers = nworkers,
                epoch,
                seconds = t,
                train_accuracy = isnothing(train) ? missing : train.accuracy,
                train_mse = isnothing(train) ? missing : train.mean_squared_error,
                validation_accuracy = isnothing(validation) ? missing : validation.accuracy,
                validation_mse = isnothing(validation) ? missing : validation.mean_squared_error,
            ))
            println("workers=", nworkers, " epoch=", epoch, " seconds=", round(t, digits = 3))
            flush(stdout)
        end
    finally
        close_trainer!(trainer)
    end

    return csv_path
end

for nworkers in WORKER_COUNTS
    run_config(nworkers)
end

println("Saved overnight metrics in ", OUTDIR)
println("settings batchsize=", BATCHSIZE,
    " hidden=", HIDDEN,
    " temp=", TEMP,
    " stepsize=", STEPSIZE,
    " beta=", BETA,
    " free/nudged=", FREE_RELAXATION, "/", NUDGED_RELAXATION,
    " lr=", LR,
    " threads=", Threads.nthreads())
