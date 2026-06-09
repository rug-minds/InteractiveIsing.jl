using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Optimisers
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_SPAWN_WORKERS", get(ENV, "ISING_MNIST_RUNTHREADED_WORKERS", "16,32")), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_SPAWN_HIDDEN", get(ENV, "ISING_MNIST_RUNTHREADED_HIDDEN", "7840")))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_SPAWN_OUTPUT_REPLICAS", get(ENV, "ISING_MNIST_RUNTHREADED_OUTPUT_REPLICAS", "4")))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_SPAWN_BATCHSIZE", get(ENV, "ISING_MNIST_RUNTHREADED_BATCHSIZE", "32")))
const NBATCHES = parse(Int, get(ENV, "ISING_MNIST_SPAWN_BATCHES", get(ENV, "ISING_MNIST_RUNTHREADED_BATCHES", "1")))
const WARMUP_BATCHES = parse(Int, get(ENV, "ISING_MNIST_SPAWN_WARMUP_BATCHES", get(ENV, "ISING_MNIST_RUNTHREADED_WARMUP_BATCHES", "1")))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_SPAWN_SWEEPS", get(ENV, "ISING_MNIST_RUNTHREADED_SWEEPS", "50.0")))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_SPAWN_LIMIT", get(ENV, "ISING_MNIST_RUNTHREADED_LIMIT", string(BATCHSIZE * (NBATCHES + WARMUP_BATCHES)))))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_SPAWN_TEMP", get(ENV, "ISING_MNIST_RUNTHREADED_TEMP", "0.001")))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_SPAWN_STEPSIZE", get(ENV, "ISING_MNIST_RUNTHREADED_STEPSIZE", "0.5")))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_SPAWN_BETA", get(ENV, "ISING_MNIST_RUNTHREADED_BETA", "0.1")))
const LR = parse(Float32, get(ENV, "ISING_MNIST_SPAWN_LR", get(ENV, "ISING_MNIST_RUNTHREADED_LR", "0.003")))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_SPAWN_WEIGHT_SCALE", get(ENV, "ISING_MNIST_RUNTHREADED_WEIGHT_SCALE", "0.005")))
const MEASURE_CPU = parse(Bool, get(ENV, "ISING_MNIST_SPAWN_CPU", get(ENV, "ISING_MNIST_RUNTHREADED_CPU", "true")))
const WORKER_INIT_MODE = Symbol(get(ENV, "ISING_MNIST_SPAWN_WORKER_INIT", "copy_first"))
const OUTDIR = get(ENV, "ISING_MNIST_SPAWN_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_spawn")))

mkpath(OUTDIR)

"""
    append_csv_row!(path, row)

Append a named-tuple row to a CSV file, writing a header if needed.
"""
function append_csv_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    process_cpu_seconds()

Return current Julia process CPU seconds on Windows when enabled.
"""
function process_cpu_seconds()
    MEASURE_CPU || return NaN
    if Sys.iswindows()
        cmd = `powershell -NoProfile -Command "(Get-Process -Id $(getpid())).CPU"`
        text = strip(read(cmd, String))
        isempty(text) && return NaN
        return parse(Float64, replace(text, "," => "."))
    end
    return NaN
end

"""
    timed_cpu(f)

Measure wall time and approximate process CPU-thread usage for a closure.
"""
function timed_cpu(f::F) where {F}
    cpu0 = process_cpu_seconds()
    wall = @elapsed f()
    cpu1 = process_cpu_seconds()
    cpu_seconds = (isnan(cpu0) || isnan(cpu1)) ? NaN : cpu1 - cpu0
    cpu_threads = (isnan(cpu_seconds) || wall < 0.02) ? NaN : cpu_seconds / wall
    return (; wall, cpu_seconds, cpu_threads)
end

"""
    active_units(graph)

Return the hidden plus output units used to convert sweeps to raw steps.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    buffer_norm(buffer)

Return an L2 norm over all gradient buffer arrays.
"""
function buffer_norm(buffer::B) where {B}
    total = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :α) && (total += sum(abs2, buffer.α))
    return sqrt(total)
end

"""
    fresh_mnist_graph(seed)

Build one MNIST graph from the architecture constructor for worker-local use.
"""
function fresh_mnist_graph(seed::T) where {T<:Integer}
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(Int(seed)),
    )
    temp!(graph, TEMP)
    return graph
end

"""
    make_each_worker_manager(layer, params, nworkers)

Create an old-spawn MNIST manager where every worker is independently built by
the public StatefulAlgorithms `MakeEachWorker` construction mode.
"""
function make_each_worker_manager(layer::L, params::P, nworkers::T) where {L,P,T<:Integer}
    recipe = (;
        makeworker = (idx, manager) -> begin
            worker_graph = fresh_mnist_graph(90_000 + 1_000 * Int(nworkers) + Int(idx))
            IsingLearning.sync_graph_params!(worker_graph, params)
            return IsingLearning._worker_process(layer, worker_graph)
        end,
        loadjob! = (slot, job, manager) -> begin
            IsingLearning._write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )

    return ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        sync_policy = NoSync(),
        poll_interval = 0.0,
    )
end

"""
    build_make_each_trainer(nworkers)

Construct the old-spawn trainer while creating every worker graph independently
instead of copying the first worker.
"""
function build_make_each_trainer(nworkers::T) where {T<:Integer}
    graph = fresh_mnist_graph(70_000 + Int(nworkers))
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

    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(LR)
    opt_state = Optimisers.setup(optimiser, params)
    manager = make_each_worker_manager(layer, params, nworkers)
    workers = collect(StatefulAlgorithms.workers(manager))
    worker_graphs = [IsingLearning._mnist_worker_state(worker).model for worker in workers]

    validation_graph = fresh_mnist_graph(110_000 + Int(nworkers))
    IsingLearning.sync_graph_params!(validation_graph, params)
    validation_worker = IsingLearning._validation_process(layer, validation_graph)
    validation_graph = StatefulAlgorithms.context(validation_worker).dynamics.model

    trainer = IsingLearning.MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        worker_graphs,
        workers,
        validation_graph,
        validation_worker,
        optimiser,
        manager,
    )
    return (; graph, layer, trainer, relaxation)
end

"""
    build_trainer(nworkers)

Construct the MNIST trainer used by the old spawn-manager comparison.
"""
function build_trainer(nworkers::T) where {T<:Integer}
    WORKER_INIT_MODE === :make_each && return build_make_each_trainer(nworkers)
    WORKER_INIT_MODE === :copy_first || throw(ArgumentError("unknown worker init mode $(WORKER_INIT_MODE)"))

    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(70_000 + Int(nworkers)),
    )
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
    trainer = init_mnist_trainer(layer; graph, numthreads = Int(nworkers), optimiser = Optimisers.Adam(LR))
    return (; graph, layer, trainer, relaxation)
end

"""
    make_jobs(xbatch, ybatch)

Create typed one-sample MNIST jobs for the current batch.
"""
function make_jobs(xbatch::X, ybatch::Y) where {X<:AbstractMatrix,Y<:AbstractMatrix}
    first_job = (; x = view(xbatch, :, first(axes(xbatch, 2))), y = view(ybatch, :, first(axes(ybatch, 2))))
    jobs = typeof(first_job)[]
    sizehint!(jobs, size(xbatch, 2))
    for sample_idx in axes(xbatch, 2)
        push!(jobs, (; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)))
    end
    return jobs
end

"""
    run_spawn_batch!(trainer, jobs)

Run one batch through the old spawn/poll/drain `ProcessManager` path.
"""
function run_spawn_batch!(trainer::T, jobs::J) where {T,J}
    IsingLearning._reset_batch_buffers!(trainer)
    run!(trainer.manager, jobs)
    return trainer
end

"""
    run_one_batch!(trainer, jobs)

Measure one old-spawn batch and return timing plus worker buffer diagnostics.
"""
function run_one_batch!(trainer::T, jobs::J) where {T,J}
    timing = timed_cpu() do
        run_spawn_batch!(trainer, jobs)
    end
    norms = [buffer_norm(IsingLearning._mnist_worker_state(worker).buffers) for worker in trainer.workers]
    return (;
        mode = "spawn",
        seconds = timing.wall,
        cpu_threads = timing.cpu_threads,
        active_workers = count(>(0), norms),
        min_worker_norm = minimum(norms),
        mean_worker_norm = mean(norms),
        max_worker_norm = maximum(norms),
    )
end

"""
    run_profile(nworkers)

Run the old spawn-manager profile for one worker count.
"""
function run_profile(nworkers::T) where {T<:Integer}
    setup = build_trainer(nworkers)
    trainer = setup.trainer
    csv_path = joinpath(OUTDIR, "mnist_spawn_profiles.csv")
    x, y = load_mnist_arrays(setup.layer; split = :train, limit = TRAIN_LIMIT)
    loader = MNISTDataLoader(x, y; batchsize = BATCHSIZE, shuffle = false, rng = Random.MersenneTwister(1))

    try
        for (batch_idx, (xbatch, ybatch)) in enumerate(loader)
            batch_idx > NBATCHES + WARMUP_BATCHES && break
            jobs = make_jobs(xbatch, ybatch)
            is_warmup = batch_idx <= WARMUP_BATCHES
            measured_batch = batch_idx - WARMUP_BATCHES
            result = run_one_batch!(trainer, jobs)

            if !is_warmup
                row = (;
                    timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                    workers = Int(nworkers),
                    threads = Threads.nthreads(),
                    mode = result.mode,
                    batch = measured_batch,
                    warmup_batches = WARMUP_BATCHES,
                    hidden = HIDDEN,
                    output_replicas = OUTPUT_REPLICAS,
                    batchsize = size(xbatch, 2),
                    jobs = length(jobs),
                    sweeps = SWEEPS,
                    relaxation = setup.relaxation,
                    graph_states = InteractiveIsing.nstates(setup.graph),
                    graph_edges = length(SparseArrays.getnzval(InteractiveIsing.adj(setup.graph))),
                    run_seconds = result.seconds,
                    run_cpu_threads = result.cpu_threads,
                    active_workers = result.active_workers,
                    min_worker_norm = result.min_worker_norm,
                    mean_worker_norm = result.mean_worker_norm,
                    max_worker_norm = result.max_worker_norm,
                )
                append_csv_row!(csv_path, row)
                println(row)
                flush(stdout)
            end
        end
    finally
        close_trainer!(trainer)
    end

    return csv_path
end

"""
    main()

Entry point for the MNIST old spawn-manager profile.
"""
function main()
    println(
        "MNIST spawn profile workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " output_replicas=", OUTPUT_REPLICAS,
        " batchsize=", BATCHSIZE,
        " batches=", NBATCHES,
        " warmup=", WARMUP_BATCHES,
        " sweeps=", SWEEPS,
        " limit=", TRAIN_LIMIT,
        " worker_init=", WORKER_INIT_MODE,
    )
    flush(stdout)
    csv_path = nothing
    for nworkers in WORKERS
        csv_path = run_profile(nworkers)
    end
    println("Saved MNIST spawn profile CSV: ", csv_path)
    println("Saved outputs in ", OUTDIR)
end

main()
