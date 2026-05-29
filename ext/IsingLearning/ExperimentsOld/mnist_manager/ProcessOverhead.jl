using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using Statistics

const HIDDENS = parse.(Int, split(get(ENV, "ISING_MNIST_OVERHEAD_HIDDENS", "120,7840"), ","))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_OVERHEAD_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_OVERHEAD_SWEEPS", "5.0"))
const REPEATS = parse(Int, get(ENV, "ISING_MNIST_OVERHEAD_REPEATS", "5"))
const WARMUP = parse(Int, get(ENV, "ISING_MNIST_OVERHEAD_WARMUP", "1"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_OVERHEAD_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_OVERHEAD_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_OVERHEAD_BETA", "0.1"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_OVERHEAD_WEIGHT_SCALE", "0.005"))
const OUTDIR = get(ENV, "ISING_MNIST_OVERHEAD_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_process_overhead")))

mkpath(OUTDIR)

"""
    append_csv_row!(path, row)

Append a named-tuple row to `path`, writing the header if this is the first row.
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
    active_units(graph)

Return the number of active hidden/output units used to convert sweep counts to
raw local-update steps.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    build_overhead_setup(hidden)

Construct one MNIST graph, one contrastive layer, and one sample job for direct
process-vs-manager timing.
"""
function build_overhead_setup(hidden::Integer)
    graph = MNISTArchitecture(
        hidden = hidden,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(10_000 + hidden),
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
    x, y = load_mnist_arrays(layer; split = :train, limit = 1)
    job = (; x = view(x, :, 1), y = view(y, :, 1))
    params = IsingLearning.read_graph_params(graph)
    return (; graph, layer, job, params, relaxation)
end

"""
    time_direct_worker!(worker, job)

Run a single already-constructed `Process` directly and return both the external
latencies around `run`/`wait`/`fetch` and the process' internal runtime clock.
"""
function time_direct_worker!(worker::W, job::J) where {W<:Process,J}
    prepare_seconds = @elapsed begin
        IsingLearning._write_example!(worker, job.x, job.y)
        Processes.reset!(worker)
    end
    fetched = Ref{Any}(nothing)
    run_call_seconds = 0.0
    wait_fetch_seconds = 0.0
    run_wait_fetch_seconds = @elapsed begin
        run_call_seconds = @elapsed run(worker)
        wait_fetch_seconds = @elapsed begin
            wait(worker)
            fetched[] = fetch(worker)
        end
    end
    internal_seconds = Processes.runtime(worker)
    close_seconds = @elapsed close(worker)
    return (; prepare_seconds, run_call_seconds, wait_fetch_seconds, run_wait_fetch_seconds, internal_seconds, close_seconds)
end

"""
    build_one_worker_manager(worker)

Wrap one existing worker process in the same `ProcessManager` recipe used for
MNIST training.
"""
function build_one_worker_manager(worker::W) where {W<:Process}
    recipe = (;
        prepare! = (slot, job, manager) -> begin
            IsingLearning._write_example!(slot.worker, job.x, job.y)
            resetworker!(slot)
            return nothing
        end,
    )
    return ProcessManager(
        recipe;
        workers = (worker,),
        flush_policy = NoFlush(),
        poll_interval = 0.0,
    )
end

"""
    time_one_hidden(hidden)

Measure direct `Process` execution and one-slot `ProcessManager` execution for
one hidden size. Each measured run uses a fresh worker so manager close/finalize
semantics are identical across trials.
"""
function time_one_hidden(hidden::Integer)
    setup = build_overhead_setup(hidden)
    csv_path = joinpath(OUTDIR, "process_overhead.csv")
    total_trials = WARMUP + REPEATS

    for trial in 1:total_trials
        direct_graph = IsingLearning._worker_graph(setup.graph, setup.params)
        direct_worker = IsingLearning._worker_process(setup.layer, direct_graph)
        direct = time_direct_worker!(direct_worker, setup.job)

        manager_graph = IsingLearning._worker_graph(setup.graph, setup.params)
        manager_worker = IsingLearning._worker_process(setup.layer, manager_graph)
        manager = build_one_worker_manager(manager_worker)
        manager_run_seconds = @elapsed run!(manager, (setup.job,))
        manager_internal_seconds = Processes.runtime(manager_worker)
        close(manager)

        is_warmup = trial <= WARMUP
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            hidden,
            output_replicas = OUTPUT_REPLICAS,
            sweeps = SWEEPS,
            relaxation = setup.relaxation,
            trial = trial - WARMUP,
            warmup = is_warmup,
            direct_prepare_seconds = direct.prepare_seconds,
            direct_run_call_seconds = direct.run_call_seconds,
            direct_wait_fetch_seconds = direct.wait_fetch_seconds,
            direct_run_wait_fetch_seconds = direct.run_wait_fetch_seconds,
            direct_internal_seconds = direct.internal_seconds,
            direct_close_seconds = direct.close_seconds,
            manager_run_seconds,
            manager_internal_seconds,
            overhead_vs_direct_latency_seconds = manager_run_seconds - direct.run_wait_fetch_seconds,
            overhead_vs_direct_latency_ratio = manager_run_seconds / direct.run_wait_fetch_seconds,
            direct_external_minus_internal_seconds = direct.run_wait_fetch_seconds - direct.internal_seconds,
            manager_external_minus_internal_seconds = manager_run_seconds - manager_internal_seconds,
        )
        is_warmup || append_csv_row!(csv_path, row)
        println(row)
        flush(stdout)
    end
    return csv_path
end

"""
    main()

Run all configured single-process overhead probes.
"""
function main()
    println(
        "MNIST process overhead hiddens=", HIDDENS,
        " threads=", Threads.nthreads(),
        " sweeps=", SWEEPS,
        " repeats=", REPEATS,
        " warmup=", WARMUP,
    )
    csv_path = nothing
    for hidden in HIDDENS
        csv_path = time_one_hidden(hidden)
    end
    println("Saved process overhead CSV: ", csv_path)
    println("Saved outputs in ", OUTDIR)
end

main()
