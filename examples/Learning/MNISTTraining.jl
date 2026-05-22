using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "ext", "IsingLearning"))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Optimisers
using Random
using Serialization

const FT = Float32
const WORKERS = parse(Int, get(ENV, "ISING_MNIST_WORKERS", "32"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_HIDDEN", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_OUTPUT_REPLICAS", "4"))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_EPOCHS", "1"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_BATCHSIZE", "256"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_SWEEPS", "500"))
const TRAIN_LIMIT_TEXT = get(ENV, "ISING_MNIST_TRAIN_LIMIT", "")
const VALIDATION_LIMIT_TEXT = get(ENV, "ISING_MNIST_VALIDATION_LIMIT", "1000")
const TEMP = parse(FT, get(ENV, "ISING_MNIST_TEMP", "0.001"))
const STEPSIZE = parse(FT, get(ENV, "ISING_MNIST_STEPSIZE", "0.5"))
const BETA = parse(FT, get(ENV, "ISING_MNIST_BETA", "0.1"))
const LR = parse(FT, get(ENV, "ISING_MNIST_LR", "0.003"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_MNIST_WEIGHT_SCALE", "0.005"))
const OUTDIR = get(ENV, "ISING_MNIST_OUTDIR", joinpath(@__DIR__, "..", "..", "ext", "IsingLearning", "experiments", "mnist_manager", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_example_mnist")))

mkpath(OUTDIR)

"""Parse an optional integer environment value."""
function optional_int(text::T) where {T<:AbstractString}
    isempty(strip(text)) && return nothing
    return parse(Int, text)
end

"""Return the number of non-input units advanced by the MNIST dynamics."""
function active_units(graph::G) where {G}
    return sum(length(InteractiveIsing.layerrange(graph[layer_idx])) for layer_idx in 2:length(graph))
end

"""Append one named-tuple row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Construct the shared-data paper-like MNIST trainer."""
function build_trainer()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = FT,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(42),
    )
    temp!(graph, TEMP)

    relaxation_steps = max(1, round(Int, SWEEPS * active_units(graph)))
    dynamics = LocalLangevin(
        stepsize = STEPSIZE,
        max_drift_fraction = FT(0.15),
        adjusted = false,
        order = :cyclic,
    )
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        free_relaxation_steps = relaxation_steps,
        nudged_relaxation_steps = relaxation_steps,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = WORKERS,
        optimiser = Optimisers.Adam(LR),
        share_static_model_data = true,
        input_mode = :field,
    )
    return (; graph, layer, trainer, relaxation_steps)
end

"""Serialize the optimizer-facing parameters and run metadata."""
function save_checkpoint(path::P, setup, stats) where {P<:AbstractString}
    open(path, "w") do io
        serialize(io, (;
            architecture = "MNIST $(D_MNIST)^2 -> $(HIDDEN) -> $(10 * OUTPUT_REPLICAS)",
            params = setup.trainer.params,
            stats,
            settings = (;
                workers = WORKERS,
                epochs = EPOCHS,
                batchsize = BATCHSIZE,
                sweeps = SWEEPS,
                relaxation_steps = setup.relaxation_steps,
                temp = TEMP,
                stepsize = STEPSIZE,
                beta = BETA,
                lr = LR,
                weight_scale = WEIGHT_SCALE,
            ),
        ))
    end
    return path
end

train_limit = optional_int(TRAIN_LIMIT_TEXT)
validation_limit = optional_int(VALIDATION_LIMIT_TEXT)
setup = build_trainer()

println("MNIST training example")
println("architecture = $(D_MNIST)^2 -> $(HIDDEN) -> $(10 * OUTPUT_REPLICAS)")
println("threads = $(Threads.nthreads()), workers = $(WORKERS), batchsize = $(BATCHSIZE), epochs = $(EPOCHS)")
println("sweeps = $(SWEEPS), relaxation_steps = $(setup.relaxation_steps), shared_static_model_data = true")
flush(stdout)

stats = nothing
elapsed = @elapsed begin
    stats = fit_mnist_threaded!(
        setup.trainer;
        epochs = EPOCHS,
        batchsize = BATCHSIZE,
        split = :train,
        validation_split = isnothing(validation_limit) ? nothing : :test,
        shuffle = true,
        rng = Random.MersenneTwister(2026),
        limit = train_limit,
        validation_limit = validation_limit,
        show_progress = true,
        show_validation_progress = false,
        train_eval_limit = min(BATCHSIZE, isnothing(train_limit) ? BATCHSIZE : train_limit),
    )
end

checkpoint_path = save_checkpoint(joinpath(OUTDIR, "mnist_shared_training.bin"), setup, stats)
append_row!(
    joinpath(OUTDIR, "mnist_shared_training_summary.csv"),
    (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-$(HIDDEN)-$(10 * OUTPUT_REPLICAS)",
        workers = WORKERS,
        threads = Threads.nthreads(),
        epochs = EPOCHS,
        batchsize = BATCHSIZE,
        sweeps = SWEEPS,
        relaxation_steps = setup.relaxation_steps,
        train_limit = isnothing(train_limit) ? "full" : string(train_limit),
        validation_limit = isnothing(validation_limit) ? "none" : string(validation_limit),
        elapsed_seconds = elapsed,
        elapsed_minutes = elapsed / 60,
        checkpoint = checkpoint_path,
    ),
)

println("elapsed seconds = ", elapsed)
println("checkpoint = ", checkpoint_path)
println("outputs = ", OUTDIR)
close_trainer!(setup.trainer)
