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

const II = IsingLearning.InteractiveIsing

_parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_WORKERS", "15"))
const HIDDENS = _parse_list(Int, get(ENV, "ISING_MNIST_BALANCED_HIDDENS", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_OUTPUT_REPLICAS", "4"))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_EPOCHS", "5"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_BALANCED_BATCHSIZE", "64"))
const TRAIN_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_TRAIN_PER_CLASS", "100"))
const VALIDATION_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_VALIDATION_PER_CLASS", "30"))
const TRAIN_EVAL_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_BALANCED_TRAIN_EVAL_PER_CLASS", "30"))
const MINITS = _parse_list(Int, get(ENV, "ISING_MNIST_BALANCED_MINITS", "1,4"))
const FREE_SWEEPS = _parse_list(Float64, get(ENV, "ISING_MNIST_BALANCED_FREE_SWEEPS", "200"))
const NUDGED_SWEEPS = _parse_list(Float64, get(ENV, "ISING_MNIST_BALANCED_NUDGED_SWEEPS", "100"))
const STEPSIZES = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_STEPSIZES", "0.05"))
const BETAS = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_BETAS", "0.05,0.1"))
const LRS = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_LRS", "0.0005,0.001"))
const TARGET_SCALES = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_TARGET_SCALES", "0.2,0.5"))
const TARGET_OFFS_RAW = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_TARGET_OFFS", ""))
const GRADIENT_SIGNS = _parse_list(Float32, get(ENV, "ISING_MNIST_BALANCED_GRADIENT_SIGNS", "1.0"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_WEIGHT_SCALE", "0.005"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_BIAS_SCALE", "0.02"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_TEMP", "0.001"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_WEIGHT_DECAY", "1e-4"))
const INPUT_LOW = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_INPUT_LOW", "-1.0"))
const INPUT_HIGH = parse(Float32, get(ENV, "ISING_MNIST_BALANCED_INPUT_HIGH", "1.0"))
const OUTDIR = get(ENV, "ISING_MNIST_BALANCED_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_balanced_grid")))

mkpath(OUTDIR)

"""
    append_row!(path, row)

Append one named-tuple row to a CSV file, writing the header on first use.
"""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    strip_weight_generators!(graph)

Remove generator closures before saving a graph checkpoint.
"""
function strip_weight_generators!(graph::G) where {G}
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

"""
    save_mnist_graph(path, graph)

Save a copy of an MNIST graph without mutating the live trainer graph.
"""
function save_mnist_graph(path::P, graph::G) where {P<:AbstractString,G}
    mkpath(dirname(path))
    return II.save_isinggraph(path, strip_weight_generators!(deepcopy(graph)))
end

"""
    balanced_mnist_arrays(layer; split, per_class, target_on, target_off, rng)

Load a class-balanced MNIST slice and rescale targets to the requested output
amplitude. This keeps small validation runs from being dominated by whichever
digits happen to occur early in the dataset.
"""
function balanced_mnist_arrays(
    layer::L;
    split::Symbol,
    per_class::Integer,
    target_on::T,
    target_off::U,
    rng = Random.MersenneTwister(1),
) where {L,T<:Real,U<:Real}
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
    if INPUT_LOW != -1f0 || INPUT_HIGH != 1f0
        x .= (x .+ 1f0) ./ 2f0
        x .*= INPUT_HIGH - INPUT_LOW
        x .+= INPUT_LOW
    end
    y = copy(yraw[:, keep])
    y .= ifelse.(y .> 0, Float32(target_on), Float32(target_off))
    return x, y
end

"""
    init_biases!(graph, seed)

Initialize magnetic fields with a small deterministic Gaussian perturbation.
"""
function init_biases!(graph::G, seed::Integer) where {G}
    rng = Random.MersenneTwister(seed)
    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

"""
    active_units(graph)

Return the number of non-input spins updated during MNIST relaxation.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    add_weight_decay!(gradient, params)

Apply decoupled L2-style shrinkage to the accumulated parameter gradient.
"""
function add_weight_decay!(gradient::G, params::P) where {G,P}
    WEIGHT_DECAY > 0 || return gradient
    gradient.w .+= WEIGHT_DECAY .* params.w
    return gradient
end

"""
    build_trainer(config)

Create one stock MNIST process-manager trainer for a grid configuration.
"""
function build_trainer(config::C) where {C<:NamedTuple}
    graph = MNISTArchitecture(
        hidden = config.hidden,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(10_000 + config.config_id),
    )
    init_biases!(graph, 20_000 + config.config_id)
    temp!(graph, TEMP)

    nactive = active_units(graph)
    free_relaxation = max(1, round(Int, config.free_sweeps * nactive))
    nudged_relaxation = max(1, round(Int, config.nudged_sweeps * nactive))
    dynamics = LocalLangevin(stepsize = config.stepsize, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = config.beta,
        free_relaxation_steps = free_relaxation,
        nudged_relaxation_steps = nudged_relaxation,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(layer; graph, numthreads = WORKERS, optimiser = Optimisers.Adam(config.lr))
    return graph, layer, trainer, free_relaxation, nudged_relaxation
end

"""
    run_minibatch!(trainer, xbatch, ybatch, batch_gradient, minit, gradient_sign)

Run one minibatch with optional repeated random initial states per example.
"""
function run_minibatch!(
    trainer::T,
    xbatch::X,
    ybatch::Y,
    batch_gradient::G,
    minit::Integer,
    gradient_sign::S,
) where {T,X<:AbstractMatrix,Y<:AbstractMatrix,G,S<:Real}
    IsingLearning._reset_batch_buffers!(trainer)
    jobs = NamedTuple[]
    sizehint!(jobs, size(xbatch, 2) * Int(minit))
    for sample_idx in axes(xbatch, 2)
        for _ in 1:Int(minit)
            push!(jobs, (; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)))
        end
    end
    run!(trainer.manager, jobs)
    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, length(jobs))
    batch_gradient.w .*= gradient_sign
    batch_gradient.b .*= gradient_sign
    hasproperty(batch_gradient, :α) && (batch_gradient.α .*= gradient_sign)
    add_weight_decay!(batch_gradient, trainer.params)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    return nothing
end

"""
    evaluate!(trainer, x, y)

Evaluate classifier accuracy and mean squared output error with the trainer's
validation worker.
"""
function evaluate!(trainer::T, x::X, y::Y) where {T,X<:AbstractMatrix,Y<:AbstractMatrix}
    nsamples = size(x, 2)
    ncorrect = 0
    total_squared_error = zero(eltype(x))
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
        pred_digit = argmax(IsingLearning._mnist_class_scores(output)) - 1
        target_digit = argmax(IsingLearning._mnist_class_scores(target)) - 1
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

"""
    run_config(config)

Train and evaluate one balanced MNIST grid configuration.
"""
function run_config(config::C) where {C<:NamedTuple}
    graph, layer, trainer, free_relaxation, nudged_relaxation = build_trainer(config)
    csv_path = joinpath(OUTDIR, "mnist_balanced_grid.csv")
    checkpoint_dir = joinpath(OUTDIR, "checkpoints", "config_$(config.config_id)")
    initial_graph_path = save_mnist_graph(joinpath(checkpoint_dir, "initial_graph.jld2"), trainer.prototype_graph)
    best_graph_path = joinpath(checkpoint_dir, "best_graph.jld2")
    final_graph_path = joinpath(checkpoint_dir, "final_graph.jld2")
    best_validation_accuracy = Ref(-Inf)
    best_validation_mse = Ref(Inf)
    best_epoch = Ref(0)
    batch_gradient = IsingLearning.gradient_buffer(graph)

    xtrain, ytrain = balanced_mnist_arrays(
        layer;
        split = :train,
        per_class = TRAIN_PER_CLASS,
        target_on = config.target_on,
        target_off = config.target_off,
        rng = Random.MersenneTwister(30_000 + config.config_id),
    )
    xval, yval = balanced_mnist_arrays(
        layer;
        split = :test,
        per_class = VALIDATION_PER_CLASS,
        target_on = config.target_on,
        target_off = config.target_off,
        rng = Random.MersenneTwister(40_000 + config.config_id),
    )
    xtrain_eval, ytrain_eval = balanced_mnist_arrays(
        layer;
        split = :train,
        per_class = TRAIN_EVAL_PER_CLASS,
        target_on = config.target_on,
        target_off = config.target_off,
        rng = Random.MersenneTwister(50_000 + config.config_id),
    )

    try
        before_train = evaluate!(trainer, xtrain_eval, ytrain_eval)
        before_val = evaluate!(trainer, xval, yval)
        best_validation_accuracy[] = Float64(before_val.accuracy)
        best_validation_mse[] = Float64(before_val.mean_squared_error)
        save_mnist_graph(best_graph_path, trainer.prototype_graph)

        println("config=", config.config_id, " epoch=0 train=", before_train, " val=", before_val)
        flush(stdout)

        for epoch in 0:EPOCHS
            seconds = 0.0
            train = epoch == 0 ? before_train : nothing
            validation = epoch == 0 ? before_val : nothing
            if epoch > 0
                loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(100_000 * config.config_id + epoch))
                seconds = @elapsed begin
                    for (xbatch, ybatch) in loader
                        run_minibatch!(trainer, xbatch, ybatch, batch_gradient, config.minit, config.gradient_sign)
                    end
                end
                train = evaluate!(trainer, xtrain_eval, ytrain_eval)
                validation = evaluate!(trainer, xval, yval)
            end

            if Float64(validation.accuracy) > best_validation_accuracy[] ||
               (Float64(validation.accuracy) == best_validation_accuracy[] && Float64(validation.mean_squared_error) < best_validation_mse[])
                best_validation_accuracy[] = Float64(validation.accuracy)
                best_validation_mse[] = Float64(validation.mean_squared_error)
                best_epoch[] = epoch
                save_mnist_graph(best_graph_path, trainer.prototype_graph)
            end

            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                config_id = config.config_id,
                hidden = config.hidden,
                output_replicas = OUTPUT_REPLICAS,
                epoch,
                seconds,
                train_per_class = TRAIN_PER_CLASS,
                validation_per_class = VALIDATION_PER_CLASS,
                batchsize = BATCHSIZE,
                free_relaxation,
                nudged_relaxation,
                free_sweeps = config.free_sweeps,
                nudged_sweeps = config.nudged_sweeps,
                minit = config.minit,
                target_on = config.target_on,
                target_off = config.target_off,
                beta = config.beta,
                lr = config.lr,
                stepsize = config.stepsize,
                weight_scale = WEIGHT_SCALE,
                bias_scale = BIAS_SCALE,
                temp = TEMP,
                weight_decay = WEIGHT_DECAY,
                gradient_sign = config.gradient_sign,
                train_accuracy = train.accuracy,
                train_mse = train.mean_squared_error,
                validation_accuracy = validation.accuracy,
                validation_mse = validation.mean_squared_error,
                best_epoch = best_epoch[],
                best_validation_accuracy = best_validation_accuracy[],
                best_validation_mse = best_validation_mse[],
                initial_graph_path,
                best_graph_path,
                final_graph_path = epoch == EPOCHS ? final_graph_path : "",
            )
            append_row!(csv_path, row)
            println(row)
            flush(stdout)
        end
        save_mnist_graph(final_graph_path, trainer.prototype_graph)
    finally
        close_trainer!(trainer)
    end
    return csv_path
end

configs = NamedTuple[]
config_id = 0
for hidden in HIDDENS, free_sweeps in FREE_SWEEPS, nudged_sweeps in NUDGED_SWEEPS,
    stepsize in STEPSIZES, beta in BETAS, lr in LRS, target_on in TARGET_SCALES,
    minit in MINITS, gradient_sign in GRADIENT_SIGNS
    target_offs = isempty(TARGET_OFFS_RAW) ? Float32[-target_on] : TARGET_OFFS_RAW
    for target_off in target_offs
        global config_id += 1
        push!(configs, (; config_id, hidden, free_sweeps, nudged_sweeps, stepsize, beta, lr, target_on, target_off, minit, gradient_sign))
    end
end

println(
    "MNIST balanced grid workers=", WORKERS,
    " threads=", Threads.nthreads(),
    " configs=", length(configs),
    " hiddens=", HIDDENS,
    " output_replicas=", OUTPUT_REPLICAS,
    " epochs=", EPOCHS,
    " train_per_class=", TRAIN_PER_CLASS,
    " validation_per_class=", VALIDATION_PER_CLASS,
    " batchsize=", BATCHSIZE,
    " minits=", MINITS,
    " free_sweeps=", FREE_SWEEPS,
    " nudged_sweeps=", NUDGED_SWEEPS,
    " stepsizes=", STEPSIZES,
    " betas=", BETAS,
    " lrs=", LRS,
    " target_ons=", TARGET_SCALES,
    " target_offs=", isempty(TARGET_OFFS_RAW) ? "negative target_on" : string(TARGET_OFFS_RAW),
    " gradient_signs=", GRADIENT_SIGNS,
)
flush(stdout)

for config in configs
    run_config(config)
end

println("Saved MNIST balanced grid in ", OUTDIR)
