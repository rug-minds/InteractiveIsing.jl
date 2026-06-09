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

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_RUNTHREADED_WORKERS", "16,32"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_OUTPUT_REPLICAS", "4"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_BATCHSIZE", "32"))
const NBATCHES = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_BATCHES", "1"))
const WARMUP_BATCHES = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_WARMUP_BATCHES", "1"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_RUNTHREADED_SWEEPS", "50.0"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_RUNTHREADED_LIMIT", string(BATCHSIZE * (NBATCHES + WARMUP_BATCHES))))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_RUNTHREADED_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_RUNTHREADED_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_RUNTHREADED_BETA", "0.1"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_RUNTHREADED_LR", "0.003"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RUNTHREADED_WEIGHT_SCALE", "0.005"))
const MEASURE_CPU = parse(Bool, get(ENV, "ISING_MNIST_RUNTHREADED_CPU", "true"))
const MODES = Symbol.(split(get(ENV, "ISING_MNIST_RUNTHREADED_MODES", "normal,dynamic,greedy,static"), ","))
const WORKER_INIT_MODE = Symbol(get(ENV, "ISING_MNIST_RUNTHREADED_WORKER_INIT", "copy_first"))
const STATIC_THREAD_SLOTS = Threads.maxthreadid()
const OUTDIR = get(ENV, "ISING_MNIST_RUNTHREADED_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_runthreaded")))

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
    validate_graph_topology(label, graph)

Scan every sparse row index in `graph` and verify that the topology is valid
before manager workers are allowed to run dynamics.
"""
function validate_graph_topology(label::L, graph::G) where {L<:AbstractString,G}
    sp = InteractiveIsing.adj(graph).sp
    nstates = InteractiveIsing.nstates(graph)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    colptr = SparseArrays.getcolptr(sp)

    length(rowval) == length(nzval) || error("$label rowval length $(length(rowval)) != nzval length $(length(nzval))")
    size(sp, 1) == nstates || error("$label row count $(size(sp, 1)) != nstates $nstates")
    size(sp, 2) == nstates || error("$label col count $(size(sp, 2)) != nstates $nstates")
    first(colptr) == 1 || error("$label first colptr $(first(colptr)) != 1")
    last(colptr) == length(rowval) + 1 || error("$label last colptr $(last(colptr)) != rowval length + 1")

    bad_col = findfirst(idx -> colptr[idx] > colptr[idx + 1], 1:(length(colptr) - 1))
    isnothing(bad_col) || error("$label colptr decreases at column $bad_col")

    bad_ptr = findfirst(row -> !(1 <= row <= nstates), rowval)
    if !isnothing(bad_ptr)
        bad_col_for_ptr = searchsortedlast(colptr, bad_ptr)
        error("$label invalid rowval: ptr=$bad_ptr col=$bad_col_for_ptr row=$(rowval[bad_ptr]) nstates=$nstates")
    end

    return (; label, nstates, nnz = length(nzval), min_row = minimum(rowval), max_row = maximum(rowval))
end

"""
    validate_trainer_topologies!(trainer)

Validate prototype, worker, and validation graph sparse topologies immediately
after construction and before any MNIST jobs are dispatched.
"""
function validate_trainer_topologies!(trainer::T) where {T}
    println("pre_run_topology=", validate_graph_topology("prototype", trainer.prototype_graph))
    for (idx, graph) in enumerate(trainer.worker_graphs)
        println("pre_run_topology=", validate_graph_topology("worker_$idx", graph))
    end
    println("pre_run_topology=", validate_graph_topology("validation", trainer.validation_graph))
    flush(stdout)
    return trainer
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

Create an MNIST training manager where every worker is constructed by the
recipe `makeworker` callback. This uses the public StatefulAlgorithms `MakeEachWorker`
mode and avoids manager-side copying or deep-copying of slot 1.
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

Construct the MNIST trainer while letting `ProcessManager` create each worker
independently instead of copying the first worker.
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

Construct the MNIST 10x-hidden trainer used by the runthreaded comparison.
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
    run_manager_mode!(trainer, jobs, mode)

Reset worker-local gradient buffers and run one job batch with the requested
manager scheduler.
"""
function run_manager_mode!(trainer::T, jobs::J, mode::Symbol) where {T,J}
    IsingLearning._reset_batch_buffers!(trainer)
    if mode === :normal
        run!(trainer.manager, jobs)
    elseif mode === :dynamic
        runthreaded!(trainer.manager, jobs, StatefulAlgorithms.Dynamic())
    elseif mode === :static
        runthreaded!(trainer.manager, jobs, StatefulAlgorithms.Static())
    elseif mode === :greedy
        runthreaded!(trainer.manager, jobs, StatefulAlgorithms.Greedy())
    else
        throw(ArgumentError("unknown manager mode $(mode)"))
    end
    return trainer
end

"""
    run_one_mode!(trainer, jobs, mode)

Measure one manager mode and return timing plus worker buffer diagnostics.
"""
function run_one_mode!(trainer::T, jobs::J, mode::Symbol) where {T,J}
    timing = timed_cpu() do
        run_manager_mode!(trainer, jobs, mode)
    end
    norms = [buffer_norm(IsingLearning._mnist_worker_state(worker).buffers) for worker in trainer.workers]
    return (;
        mode = string(mode),
        seconds = timing.wall,
        cpu_threads = timing.cpu_threads,
        active_workers = count(>(0), norms),
        min_worker_norm = minimum(norms),
        mean_worker_norm = mean(norms),
        max_worker_norm = maximum(norms),
    )
end

"""
    available_modes(nworkers)

Return scheduler modes valid for this worker/thread configuration.
"""
function available_modes(nworkers::T) where {T<:Integer}
    valid = Set([:normal, :dynamic, :greedy])
    Int(nworkers) >= STATIC_THREAD_SLOTS && push!(valid, :static)
    modes = Symbol[]
    for mode in MODES
        if mode in valid
            push!(modes, mode)
        else
            println(
                "Skipping invalid mode ", mode,
                " for workers=", nworkers,
                " threads=", Threads.nthreads(),
                " static_slots=", STATIC_THREAD_SLOTS,
            )
        end
    end
    isempty(modes) && throw(ArgumentError("no valid manager modes selected"))
    return modes
end

"""
    run_profile(nworkers)

Run the native manager-schedule comparison for one worker count.
"""
function run_profile(nworkers::T) where {T<:Integer}
    setup = build_trainer(nworkers)
    trainer = setup.trainer
    validate_trainer_topologies!(trainer)
    csv_path = joinpath(OUTDIR, "mnist_runthreaded_profiles.csv")
    x, y = load_mnist_arrays(setup.layer; split = :train, limit = TRAIN_LIMIT)
    loader = MNISTDataLoader(x, y; batchsize = BATCHSIZE, shuffle = false, rng = Random.MersenneTwister(1))
    modes = available_modes(nworkers)

    try
        for (batch_idx, (xbatch, ybatch)) in enumerate(loader)
            batch_idx > NBATCHES + WARMUP_BATCHES && break
            jobs = make_jobs(xbatch, ybatch)
            is_warmup = batch_idx <= WARMUP_BATCHES
            measured_batch = batch_idx - WARMUP_BATCHES

            # Run every scheduler on the same jobs. `prepare!` resets each worker
            # before a job, and buffers are reset per mode above.
            mode_results = [run_one_mode!(trainer, jobs, mode) for mode in modes]

            if !is_warmup
                for result in mode_results
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
                end
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

Entry point for the MNIST native runthreaded manager comparison.
"""
function main()
    println(
        "MNIST runthreaded profile workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " static_slots=", STATIC_THREAD_SLOTS,
        " hidden=", HIDDEN,
        " output_replicas=", OUTPUT_REPLICAS,
        " batchsize=", BATCHSIZE,
        " batches=", NBATCHES,
        " warmup=", WARMUP_BATCHES,
        " sweeps=", SWEEPS,
        " limit=", TRAIN_LIMIT,
        " modes=", MODES,
        " worker_init=", WORKER_INIT_MODE,
    )
    flush(stdout)
    csv_path = nothing
    for nworkers in WORKERS
        csv_path = run_profile(nworkers)
    end
    println("Saved MNIST runthreaded profile CSV: ", csv_path)
    println("Saved outputs in ", OUTDIR)
end

main()
