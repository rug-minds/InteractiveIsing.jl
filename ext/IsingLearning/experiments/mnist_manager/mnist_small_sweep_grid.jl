using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using SparseArrays
using Statistics

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_SMALL_WORKERS", "16"))
const HIDDENS = parse.(Int, split(get(ENV, "ISING_MNIST_SMALL_HIDDENS", "128,512"), ","))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_SMALL_EPOCHS", "5"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_SMALL_BATCHSIZE", "128"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_SMALL_TRAIN_LIMIT", "2048"))
const VALIDATION_LIMIT = parse(Int, get(ENV, "ISING_MNIST_SMALL_VALIDATION_LIMIT", "512"))
const TRAIN_EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_SMALL_TRAIN_EVAL_LIMIT", "512"))
const DIGITS_RAW = get(ENV, "ISING_MNIST_SMALL_DIGITS", "")
const DIGITS = isempty(DIGITS_RAW) ? collect(0:9) : parse.(Int, split(DIGITS_RAW, ","))
const MINIT = parse(Int, get(ENV, "ISING_MNIST_SMALL_MINIT", "2"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_SMALL_SWEEPS", "2.0"))
const TARGET_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_TARGET_SCALE", "0.2"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_WEIGHT_SCALE", "0.01"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_BIAS_SCALE", "0.02"))
const SKIP_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_SKIP_SCALE", "0.0"))
const PIXEL_COPY_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_PIXEL_COPY_SCALE", "0.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_SMALL_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_SMALL_STEPSIZE", "0.05"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_SMALL_BETA", "0.1"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_SMALL_LR", "0.003"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "ISING_MNIST_SMALL_WEIGHT_DECAY", "1e-4"))
const TARGET_OFF = parse(Float32, get(ENV, "ISING_MNIST_SMALL_TARGET_OFF", string(-TARGET_SCALE)))
const TARGET_ON = parse(Float32, get(ENV, "ISING_MNIST_SMALL_TARGET_ON", string(TARGET_SCALE)))
const INPUT_LOW = parse(Float32, get(ENV, "ISING_MNIST_SMALL_INPUT_LOW", "-1.0"))
const INPUT_HIGH = parse(Float32, get(ENV, "ISING_MNIST_SMALL_INPUT_HIGH", "1.0"))
const OUTDIR = get(ENV, "ISING_MNIST_SMALL_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_small_sweep")))

mkpath(OUTDIR)

function append_row!(path, row)
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

function scaled_targets!(y)
    y .= ifelse.(y .> 0, TARGET_ON, TARGET_OFF)
    return y
end

function load_scaled_mnist(layer; split, limit)
    xraw, yraw = load_mnist_arrays(layer; split, limit = nothing)
    keep = Int[]
    for idx in axes(yraw, 2)
        digit = argmax(view(yraw, :, idx)) - 1
        digit in DIGITS && push!(keep, idx)
        !isnothing(limit) && length(keep) >= limit && break
    end
    x = copy(xraw[:, keep])
    if INPUT_LOW != -1f0 || INPUT_HIGH != 1f0
        x .= (x .+ 1f0) ./ 2f0
        x .*= INPUT_HIGH - INPUT_LOW
        x .+= INPUT_LOW
    end
    y = copy(yraw[:, keep])
    return x, scaled_targets!(y)
end

function init_biases!(graph, seed)
    rng = Random.MersenneTwister(seed)
    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

function init_input_output_skip!(graph, seed)
    SKIP_SCALE > 0 || return graph
    rng = Random.MersenneTwister(seed)
    input_idxs = collect(InteractiveIsing.layerrange(graph[1]))
    output_idxs = collect(InteractiveIsing.layerrange(graph[end]))
    for output_idx in output_idxs, input_idx in input_idxs
        InteractiveIsing.adj(graph)[output_idx, input_idx] = SKIP_SCALE * randn(rng, eltype(graph))
    end
    return graph
end

function init_pixel_copy_hidden!(graph)
    PIXEL_COPY_SCALE > 0 || return graph
    input_idxs = collect(InteractiveIsing.layerrange(graph[1]))
    hidden_idxs = collect(InteractiveIsing.layerrange(graph[2]))
    A = InteractiveIsing.adj(graph)

    for hidden_idx in hidden_idxs, input_idx in input_idxs
        A[hidden_idx, input_idx] = zero(eltype(graph))
    end

    for (hidden_pos, hidden_idx) in enumerate(hidden_idxs)
        input_idx = input_idxs[mod1(hidden_pos, length(input_idxs))]
        A[hidden_idx, input_idx] = PIXEL_COPY_SCALE
    end
    return graph
end

function active_units(graph)
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

function build_trainer(hidden, config_id)
    graph = MNISTArchitecture(
        hidden = hidden,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(10_000 + config_id),
    )
    init_biases!(graph, 20_000 + config_id)
    init_pixel_copy_hidden!(graph)
    init_input_output_skip!(graph, 30_000 + config_id)
    temp!(graph, TEMP)
    relaxation = max(1, round(Int, SWEEPS * active_units(graph)))
    dynamics = LocalLangevin(stepsize = STEPSIZE, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        free_relaxation_steps = relaxation,
        nudged_relaxation_steps = relaxation,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(layer; graph, numthreads = WORKERS, optimiser = Optimisers.Adam(LR))
    return graph, layer, trainer, relaxation
end

function add_weight_decay!(gradient, params)
    WEIGHT_DECAY > 0 || return gradient
    gradient.w .+= WEIGHT_DECAY .* params.w
    return gradient
end

function run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
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
    add_weight_decay!(batch_gradient, trainer.params)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    return nothing
end

function evaluate!(trainer, x, y)
    nsamples = size(x, 2)
    ncorrect = 0
    total_squared_error = zero(eltype(x))
    digit_idxs = DIGITS .+ 1
    for sample_idx in axes(x, 2)
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        output = IsingLearning._validation_output(trainer)
        target = view(y, :, sample_idx)
        total_squared_error += sum(abs2, output .- target)
        pred_digit = DIGITS[argmax(view(output, digit_idxs))]
        target_digit = argmax(target) - 1
        ncorrect += pred_digit == target_digit
    end
    accuracy = ncorrect / nsamples
    return (;
        accuracy,
        classification_error = 1 - accuracy,
        mean_squared_error = total_squared_error / nsamples,
        nsamples,
    )
end

function run_config(hidden, config_id)
    graph, layer, trainer, relaxation = build_trainer(hidden, config_id)
    csv_path = joinpath(OUTDIR, "mnist_small_sweep_grid.csv")
    batch_gradient = IsingLearning.gradient_buffer(graph)
    xtrain, ytrain = load_scaled_mnist(layer; split = :train, limit = TRAIN_LIMIT)
    xval, yval = load_scaled_mnist(layer; split = :test, limit = VALIDATION_LIMIT)
    xtrain_eval = @view xtrain[:, 1:min(TRAIN_EVAL_LIMIT, size(xtrain, 2))]
    ytrain_eval = @view ytrain[:, 1:min(TRAIN_EVAL_LIMIT, size(ytrain, 2))]

    try
        before_train = evaluate!(trainer, xtrain_eval, ytrain_eval)
        before_val = evaluate!(trainer, xval, yval)
        append_row!(csv_path, (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            config_id,
            hidden,
            epoch = 0,
            seconds = 0.0,
            relaxation,
            sweeps = SWEEPS,
            minit = MINIT,
            digits = join(DIGITS, "-"),
            target_scale = TARGET_SCALE,
            target_off = TARGET_OFF,
            target_on = TARGET_ON,
            input_low = INPUT_LOW,
            input_high = INPUT_HIGH,
            skip_scale = SKIP_SCALE,
            pixel_copy_scale = PIXEL_COPY_SCALE,
            beta = BETA,
            lr = LR,
            train_accuracy = before_train.accuracy,
            train_mse = before_train.mean_squared_error,
            validation_accuracy = before_val.accuracy,
            validation_mse = before_val.mean_squared_error,
        ))
        println("config=", config_id, " hidden=", hidden, " epoch=0 train=", before_train, " val=", before_val)
        flush(stdout)

        for epoch in 1:EPOCHS
            loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(100_000 * config_id + epoch))
            seconds = @elapsed begin
                for (xbatch, ybatch) in loader
                    run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
                end
            end
            train = evaluate!(trainer, xtrain_eval, ytrain_eval)
            validation = evaluate!(trainer, xval, yval)
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                config_id,
                hidden,
                epoch,
                seconds,
                relaxation,
                sweeps = SWEEPS,
                minit = MINIT,
                digits = join(DIGITS, "-"),
                target_scale = TARGET_SCALE,
                target_off = TARGET_OFF,
                target_on = TARGET_ON,
                input_low = INPUT_LOW,
                input_high = INPUT_HIGH,
                skip_scale = SKIP_SCALE,
                pixel_copy_scale = PIXEL_COPY_SCALE,
                beta = BETA,
                lr = LR,
                train_accuracy = train.accuracy,
                train_mse = train.mean_squared_error,
                validation_accuracy = validation.accuracy,
                validation_mse = validation.mean_squared_error,
            )
            append_row!(csv_path, row)
            println(row)
            flush(stdout)
        end
    finally
        close_trainer!(trainer)
    end
    return csv_path
end

println(
    "MNIST small sweep grid workers=", WORKERS,
    " threads=", Threads.nthreads(),
    " hiddens=", HIDDENS,
    " epochs=", EPOCHS,
    " train_limit=", TRAIN_LIMIT,
    " digits=", DIGITS,
    " batchsize=", BATCHSIZE,
    " minit=", MINIT,
    " sweeps=", SWEEPS,
    " target_off/on=", TARGET_OFF, "/", TARGET_ON,
    " input_low/high=", INPUT_LOW, "/", INPUT_HIGH,
    " skip_scale=", SKIP_SCALE,
    " pixel_copy_scale=", PIXEL_COPY_SCALE,
    " beta=", BETA,
    " lr=", LR,
)
flush(stdout)

for (config_id, hidden) in enumerate(HIDDENS)
    run_config(hidden, config_id)
end

println("Saved MNIST small sweep grid in ", OUTDIR)
