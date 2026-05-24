using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using Statistics

const WORKER_COUNTS = parse.(Int, split(get(ENV, "ISING_MNIST_PROBE_WORKERS", "1,4,8,12,16"), ","))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_PROBE_BATCHSIZE", "256"))
const NBATCHES = parse(Int, get(ENV, "ISING_MNIST_PROBE_BATCHES", "3"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_PROBE_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_PROBE_LIMIT", string(BATCHSIZE * NBATCHES)))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_PROBE_TEMP", "0.005"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_PROBE_STEPSIZE", "0.4"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_PROBE_BETA", "2.0"))
const FREE_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_PROBE_FREE", "600"))
const NUDGED_RELAXATION = parse(Int, get(ENV, "ISING_MNIST_PROBE_NUDGED", "600"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_PROBE_LR", "0.003"))
const OUTDIR = get(ENV, "ISING_MNIST_PROBE_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS")))

mkpath(OUTDIR)

function mnist_probe_layer()
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

function buffer_norm(buffer)
    total = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :α) && (total += sum(abs2, buffer.α))
    return sqrt(total)
end

function gradient_norm(buffer)
    return buffer_norm(buffer)
end

function run_probe_for_workers(nworkers::Integer)
    graph, layer = mnist_probe_layer()
    trainer = init_mnist_trainer(layer; graph, numthreads = nworkers, optimiser = Optimisers.Descent(LR))
    println("loading MNIST arrays for workers=", nworkers, " limit=", TRAIN_LIMIT)
    flush(stdout)
    x, y = load_mnist_arrays(layer; split = :train, limit = TRAIN_LIMIT)
    loader = MNISTDataLoader(x, y; batchsize = BATCHSIZE, shuffle = true, rng = Random.MersenneTwister(1234))
    batch_gradient = IsingLearning.gradient_buffer(graph)

    println("workers=", nworkers,
        " slots=", length(slots(trainer.manager)),
        " owns_workers=", trainer.manager.owns_workers,
        " unique_worker_ids=", length(unique(map(worker -> worker.id, trainer.workers))),
        " samples=", size(x, 2),
        " batchsize=", BATCHSIZE)
    flush(stdout)

    rows = NamedTuple[]
    for (batch_idx, (xbatch, ybatch)) in enumerate(loader)
        batch_idx > NBATCHES && break
        IsingLearning._reset_batch_buffers!(trainer)
        jobs = [(; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)) for sample_idx in axes(xbatch, 2)]

        run_seconds = @elapsed run!(trainer.manager, jobs)
        worker_norms = [buffer_norm(Processes.context(worker)._state.buffers) for worker in trainer.workers]
        active_workers = count(>(0), worker_norms)

        collect_seconds = @elapsed IsingLearning._collect_batch_gradient!(trainer, batch_gradient, size(xbatch, 2))
        grad_norm = gradient_norm(batch_gradient)

        update_seconds = @elapsed begin
            trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
            IsingLearning._broadcast_params!(trainer)
        end

        row = (;
            workers = nworkers,
            batch = batch_idx,
            run_seconds,
            collect_seconds,
            update_seconds,
            total_seconds = run_seconds + collect_seconds + update_seconds,
            active_workers,
            min_worker_norm = minimum(worker_norms),
            mean_worker_norm = mean(worker_norms),
            max_worker_norm = maximum(worker_norms),
            grad_norm,
        )
        push!(rows, row)
        println(row)
        flush(stdout)
    end

    close_trainer!(trainer)
    return rows
end

function write_csv(path, rows)
    isempty(rows) && return path
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(names, ","))
        for row in rows
            println(io, join((getproperty(row, name) for name in names), ","))
        end
    end
    return path
end

all_rows = NamedTuple[]
for nworkers in WORKER_COUNTS
    append!(all_rows, run_probe_for_workers(nworkers))
end

csv_path = write_csv(joinpath(OUTDIR, "mnist_minibatch_probe.csv"), all_rows)
println("Saved probe CSV: ", csv_path)
println("settings hidden=", HIDDEN,
    " temp=", TEMP,
    " stepsize=", STEPSIZE,
    " beta=", BETA,
    " free/nudged=", FREE_RELAXATION, "/", NUDGED_RELAXATION,
    " threads=", Threads.nthreads())
