using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random

const II = IsingLearning.InteractiveIsing

function _parse_list(::Type{T}, value::AbstractString) where {T}
    return parse.(T, split(value, ","))
end

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_GRID_WORKERS", "16"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_GRID_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_GRID_BATCHSIZE", "256"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_GRID_TRAIN_LIMIT", "2048"))
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_GRID_TRAIN_EVAL_LIMIT", "128"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_GRID_VALIDATION_LIMIT", "128"))
const FULL_EPOCHS = parse(Int, get(ENV, "ISING_MNIST_GRID_FULL_EPOCHS", "0"))
const FULL_TRAIN_LIMIT_RAW = get(ENV, "ISING_MNIST_GRID_FULL_TRAIN_LIMIT", "")
const FULL_TRAIN_LIMIT = isempty(FULL_TRAIN_LIMIT_RAW) ? nothing : parse(Int, FULL_TRAIN_LIMIT_RAW)
const FULL_VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_GRID_FULL_VALIDATION_LIMIT", "1024"))
const FULL_TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_GRID_FULL_TRAIN_EVAL_LIMIT", "1024"))
const FULL_CHECKPOINT_EVERY = parse(Int, get(ENV, "ISING_MNIST_GRID_FULL_CHECKPOINT_EVERY", "1"))
const RELAXATIONS = _parse_list(Int, get(ENV, "ISING_MNIST_GRID_RELAXATIONS", "150,300,600"))
const STEPSIZES = _parse_list(Float32, get(ENV, "ISING_MNIST_GRID_STEPSIZES", "0.2,0.4,0.6"))
const TEMPS = _parse_list(Float32, get(ENV, "ISING_MNIST_GRID_TEMPS", "0.005"))
const BETAS = _parse_list(Float32, get(ENV, "ISING_MNIST_GRID_BETAS", "2.0"))
const LRS = _parse_list(Float32, get(ENV, "ISING_MNIST_GRID_LRS", "0.003"))
const OUTDIR = get(ENV, "ISING_MNIST_GRID_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_grid")))

mkpath(OUTDIR)

function mnist_grid_layer(; hidden, temp, stepsize, beta, relaxation)
    graph = MNISTArchitecture(hidden = hidden, precision = Float32)
    temp!(graph, temp)
    dynamics = LocalLangevin(stepsize = stepsize, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = beta,
        free_relaxation_steps = relaxation,
        nudged_relaxation_steps = relaxation,
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

function strip_weight_generators!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

function save_mnist_graph(path, graph)
    return II.save_isinggraph(path, strip_weight_generators!(deepcopy(graph)))
end

function run_config(; relaxation, stepsize, temp, beta, lr, config_id)
    graph, layer = mnist_grid_layer(; hidden = HIDDEN, temp, stepsize, beta, relaxation)
    trainer = init_mnist_trainer(layer; graph, numthreads = WORKERS, optimiser = Optimisers.Descent(lr))
    csv_path = joinpath(OUTDIR, "mnist_param_grid.csv")

    try
        xtrain, ytrain = load_mnist_arrays(layer; split = :train, limit = TRAIN_LIMIT)
        xtrain_eval = @view xtrain[:, 1:min(TRAIN_EVAL_LIMIT, size(xtrain, 2))]
        ytrain_eval = @view ytrain[:, 1:min(TRAIN_EVAL_LIMIT, size(ytrain, 2))]
        xval, yval = load_mnist_arrays(layer; split = :test, limit = VALIDATION_LIMIT)

        before_train = IsingLearning.evaluate_mnist!(trainer, xtrain_eval, ytrain_eval; show_progress = false)
        before_val = IsingLearning.evaluate_mnist!(trainer, xval, yval; show_progress = false)

        batch_gradient = IsingLearning.gradient_buffer(graph)
        loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(1000 + config_id))
        nbatches = 0
        train_seconds = @elapsed begin
            for (xbatch, ybatch) in loader
                IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
                nbatches += 1
            end
        end

        after_train = IsingLearning.evaluate_mnist!(trainer, xtrain_eval, ytrain_eval; show_progress = false)
        after_val = IsingLearning.evaluate_mnist!(trainer, xval, yval; show_progress = false)

        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            config_id,
            workers = WORKERS,
            hidden = HIDDEN,
            train_limit = TRAIN_LIMIT,
            batchsize = BATCHSIZE,
            nbatches,
            relaxation,
            stepsize,
            temp,
            beta,
            lr,
            train_seconds,
            before_train_error = before_train.classification_error,
            after_train_error = after_train.classification_error,
            delta_train_error = after_train.classification_error - before_train.classification_error,
            before_train_mse = before_train.mean_squared_error,
            after_train_mse = after_train.mean_squared_error,
            delta_train_mse = after_train.mean_squared_error - before_train.mean_squared_error,
            before_val_error = before_val.classification_error,
            after_val_error = after_val.classification_error,
            delta_val_error = after_val.classification_error - before_val.classification_error,
            before_val_mse = before_val.mean_squared_error,
            after_val_mse = after_val.mean_squared_error,
            delta_val_mse = after_val.mean_squared_error - before_val.mean_squared_error,
        )
        append_row!(csv_path, row)
        println(row)
        flush(stdout)
        return row
    finally
        close_trainer!(trainer)
    end
end

config_id = 0
rows = NamedTuple[]
for relaxation in RELAXATIONS, stepsize in STEPSIZES, temp in TEMPS, beta in BETAS, lr in LRS
    global config_id += 1
    row = run_config(; relaxation, stepsize, temp, beta, lr, config_id)
    push!(rows, row)
end

function run_full_from_best(best)
    FULL_EPOCHS > 0 || return nothing

    graph, layer = mnist_grid_layer(;
        hidden = HIDDEN,
        temp = best.temp,
        stepsize = best.stepsize,
        beta = best.beta,
        relaxation = best.relaxation,
    )
    trainer = init_mnist_trainer(layer; graph, numthreads = WORKERS, optimiser = Optimisers.Descent(best.lr))
    csv_path = joinpath(OUTDIR, "mnist_best_full.csv")
    checkpoint_dir = joinpath(OUTDIR, "checkpoints")
    mkpath(checkpoint_dir)
    initial_graph_path = save_mnist_graph(joinpath(checkpoint_dir, "initial_graph.jld2"), trainer.prototype_graph)
    best_graph_path = joinpath(checkpoint_dir, "best_graph.jld2")
    final_graph_path = joinpath(checkpoint_dir, "final_graph.jld2")
    best_validation_mse = Ref(Inf)
    best_epoch = Ref(0)

    try
        for epoch in 1:FULL_EPOCHS
            result_ref = Ref{Any}(nothing)
            seconds = @elapsed begin
                _, stats = fit_mnist_threaded!(
                    trainer;
                    epochs = 1,
                    batchsize = BATCHSIZE,
                    split = :train,
                    validation_split = :test,
                    shuffle = true,
                    rng = Random.MersenneTwister(20_000 + epoch),
                    limit = FULL_TRAIN_LIMIT,
                    validation_limit = FULL_VALIDATION_LIMIT,
                    show_progress = true,
                    show_validation_progress = false,
                    log_metrics = true,
                    train_eval_limit = FULL_TRAIN_EVAL_LIMIT,
                    full_train_eval_every = nothing,
                )
                result_ref[] = only(stats)
            end
            result = result_ref[]
            train = result.train
            validation = result.validation
            validation_mse = isnothing(validation) ? Inf : Float64(validation.mean_squared_error)
            epoch_graph_path = ""
            if FULL_CHECKPOINT_EVERY > 0 && (epoch == 1 || epoch == FULL_EPOCHS || epoch % FULL_CHECKPOINT_EVERY == 0)
                epoch_graph_path = save_mnist_graph(joinpath(checkpoint_dir, "epoch_$(lpad(epoch, 4, '0')).jld2"), trainer.prototype_graph)
            end
            if validation_mse < best_validation_mse[]
                best_validation_mse[] = validation_mse
                best_epoch[] = epoch
                save_mnist_graph(best_graph_path, trainer.prototype_graph)
            end
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                source_config_id = best.config_id,
                workers = WORKERS,
                epoch,
                seconds,
                hidden = HIDDEN,
                train_limit = FULL_TRAIN_LIMIT,
                batchsize = BATCHSIZE,
                relaxation = best.relaxation,
                stepsize = best.stepsize,
                temp = best.temp,
                beta = best.beta,
                lr = best.lr,
                train_accuracy = isnothing(train) ? missing : train.accuracy,
                train_mse = isnothing(train) ? missing : train.mean_squared_error,
                validation_accuracy = isnothing(validation) ? missing : validation.accuracy,
                validation_mse = isnothing(validation) ? missing : validation.mean_squared_error,
                best_epoch = best_epoch[],
                best_validation_mse = best_validation_mse[],
                epoch_graph_path,
                best_graph_path,
            )
            append_row!(csv_path, row)
            println(row)
            flush(stdout)
        end
        save_mnist_graph(final_graph_path, trainer.prototype_graph)
        open(joinpath(OUTDIR, "README.md"), "w") do io
            println(io, "# MNIST Manager Full Run")
            println(io)
            println(io, "- Initial graph: `$(initial_graph_path)`")
            println(io, "- Best graph: `$(best_graph_path)`")
            println(io, "- Final graph: `$(final_graph_path)`")
            println(io, "- Metrics: `$(csv_path)`")
            println(io, "- Best epoch: `$(best_epoch[])`")
            println(io, "- Best validation MSE: `$(best_validation_mse[])`")
            println(io)
            println(io, "Settings:")
            println(io, "- workers: `$(WORKERS)`")
            println(io, "- hidden: `$(HIDDEN)`")
            println(io, "- batchsize: `$(BATCHSIZE)`")
            println(io, "- full epochs: `$(FULL_EPOCHS)`")
            println(io, "- full train limit: `$(FULL_TRAIN_LIMIT)`")
            println(io, "- full validation limit: `$(FULL_VALIDATION_LIMIT)`")
            println(io, "- relaxation: `$(best.relaxation)`")
            println(io, "- stepsize: `$(best.stepsize)`")
            println(io, "- T: `$(best.temp)`")
            println(io, "- beta: `$(best.beta)`")
            println(io, "- lr: `$(best.lr)`")
        end
    finally
        close_trainer!(trainer)
    end
    return csv_path
end

if !isempty(rows)
    best = rows[argmin(map(row -> row.after_val_mse, rows))]
    println("Best grid row by after_val_mse: ", best)
    flush(stdout)
    full_path = run_full_from_best(best)
    isnothing(full_path) || println("Saved best full run: ", full_path)
end

println("Saved MNIST grid: ", joinpath(OUTDIR, "mnist_param_grid.csv"))
println("settings workers=", WORKERS,
    " hidden=", HIDDEN,
    " train_limit=", TRAIN_LIMIT,
    " batchsize=", BATCHSIZE,
    " train_eval_limit=", TRAIN_EVAL_LIMIT,
    " validation_limit=", VALIDATION_LIMIT,
    " full_epochs=", FULL_EPOCHS,
    " threads=", Threads.nthreads())
