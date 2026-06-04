using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Random
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_DIRECT_WORKERS", "16,32"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_DIRECT_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_DIRECT_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_DIRECT_SWEEPS", "500.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_DIRECT_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_DIRECT_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_DIRECT_BETA", "0.1"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_DIRECT_WEIGHT_SCALE", "0.005"))
const OUTDIR = get(ENV, "ISING_MNIST_DIRECT_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_direct_concurrency")))

mkpath(OUTDIR)

"""
    append_csv_row!(path, row)

Append a named-tuple row to a CSV file, writing the header on first use.
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

Return active hidden/output units for converting sweep counts to local-update
steps.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    build_setup(nsamples)

Create the shared prototype graph/layer, load `nsamples` MNIST jobs, and read
the synchronized parameter vector used to initialize each direct worker graph.
"""
function build_setup(nsamples::Integer)
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(90_000 + nsamples),
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
    x, y = load_mnist_arrays(layer; split = :train, limit = Int(nsamples))
    jobs = [(; x = view(x, :, sample_idx), y = view(y, :, sample_idx)) for sample_idx in 1:Int(nsamples)]
    params = IsingLearning.read_graph_params(graph)
    return (; graph, layer, jobs, params, relaxation)
end

"""
    build_workers(layer, graph, params, nworkers)

Construct independent direct `Process` workers matching the MNIST manager
worker algorithm, but without wrapping them in a `ProcessManager`.
"""
function build_workers(layer::L, graph::G, params::P, nworkers::Integer) where {L,G,P}
    workers = Process[]
    sizehint!(workers, Int(nworkers))
    for _ in 1:Int(nworkers)
        worker_graph = IsingLearning._worker_graph(graph, params)
        push!(workers, IsingLearning._worker_process(layer, worker_graph))
    end
    return workers
end

"""
    run_direct_wave!(workers, jobs)

Write one job per worker, launch all workers directly, then wait/fetch every
task. This is the no-manager comparison for one ProcessManager scheduling wave.
"""
function run_direct_wave!(workers::W, jobs::J) where {W<:AbstractVector{<:Process},J<:AbstractVector}
    length(jobs) <= length(workers) || throw(ArgumentError("direct wave needs jobs <= workers"))

    write_reset_seconds = @elapsed begin
        for idx in eachindex(jobs)
            IsingLearning._write_example!(workers[idx], jobs[idx].x, jobs[idx].y)
            StatefulAlgorithms.reset!(workers[idx])
        end
    end

    launch_seconds = @elapsed begin
        for idx in eachindex(jobs)
            run(workers[idx])
        end
    end

    wait_fetch_seconds = @elapsed begin
        for idx in eachindex(jobs)
            wait(workers[idx])
            fetch(workers[idx])
        end
    end

    internal = [StatefulAlgorithms.runtime(workers[idx]) for idx in eachindex(jobs)]
    close_seconds = @elapsed begin
        for idx in eachindex(jobs)
            close(workers[idx])
        end
    end

    return (;
        write_reset_seconds,
        launch_seconds,
        wait_fetch_seconds,
        total_latency_seconds = launch_seconds + wait_fetch_seconds,
        close_seconds,
        min_internal_seconds = minimum(internal),
        mean_internal_seconds = mean(internal),
        max_internal_seconds = maximum(internal),
    )
end

"""
    run_config(nworkers)

Run one direct-concurrency wave with `nworkers` jobs and `nworkers` direct
processes, then write a CSV row.
"""
function run_config(nworkers::Integer)
    setup = build_setup(nworkers)
    workers = build_workers(setup.layer, setup.graph, setup.params, nworkers)
    timing = run_direct_wave!(workers, setup.jobs)

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        workers = Int(nworkers),
        jobs = Int(nworkers),
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        sweeps = SWEEPS,
        relaxation = setup.relaxation,
        write_reset_seconds = timing.write_reset_seconds,
        launch_seconds = timing.launch_seconds,
        wait_fetch_seconds = timing.wait_fetch_seconds,
        total_latency_seconds = timing.total_latency_seconds,
        close_seconds = timing.close_seconds,
        min_internal_seconds = timing.min_internal_seconds,
        mean_internal_seconds = timing.mean_internal_seconds,
        max_internal_seconds = timing.max_internal_seconds,
    )
    append_csv_row!(joinpath(OUTDIR, "direct_concurrency.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Compare direct concurrent Process waves without ProcessManager.
"""
function main()
    println(
        "MNIST direct process concurrency workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " sweeps=", SWEEPS,
    )
    for nworkers in WORKERS
        run_config(nworkers)
    end
    println("Saved outputs in ", OUTDIR)
end

main()
