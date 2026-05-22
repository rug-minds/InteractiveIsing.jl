using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Random
using SparseArrays

const CHECKED_CONTRACTION = get(ENV, "ISING_MNIST_REUSE_CHECKED_CONTRACTION", "false") == "true"
const NWORKERS = parse(Int, get(ENV, "ISING_MNIST_REUSE_WORKERS", "1"))
const JOBS_PER_WORKER = parse(Int, get(ENV, "ISING_MNIST_REUSE_JOBS_PER_WORKER", "2"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_REUSE_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_REUSE_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_REUSE_SWEEPS", "5.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_REUSE_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_REUSE_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_REUSE_BETA", "0.1"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_REUSE_WEIGHT_SCALE", "0.005"))

if CHECKED_CONTRACTION
    @eval InteractiveIsing begin
        """
            column_contraction(i, v, sp; transform, transform_weight)

        Diagnostic replacement for the normal LoopVectorization kernel. It keeps
        the same semantics but performs normal Julia bounds checks so a bad
        cached active index or malformed sparse row index becomes a `BoundsError`
        instead of an unchecked vectorized load.
        """
        @inline function column_contraction(
            i::I,
            v::AbstractArray{T},
            sp::SparseArrays.SparseMatrixCSC{T};
            transform::F = identity,
            transform_weight::FW = identity,
        ) where {I<:Integer,T,F,FW}
            checkbounds(axes(sp, 2), Int(i))
            total = zero(T)
            rowval = SparseArrays.getrowval(sp)
            nzval = SparseArrays.getnzval(sp)
            for ptr in SparseArrays.nzrange(sp, Int(i))
                j = rowval[ptr]
                checkbounds(v, j)
                total += transform_weight(nzval[ptr]) * transform(v[j])
            end
            return total
        end
    end
end

"""
    process_memory_mb()

Return the current process working set and private memory in MiB on Windows.
"""
function process_memory_mb()
    if Sys.iswindows()
        cmd = `powershell -NoProfile -Command "\$p=Get-Process -Id $(getpid()); \"\$([math]::Round(\$p.WorkingSet64/1MB,1)),\$([math]::Round(\$p.PrivateMemorySize64/1MB,1))\""`
        fields = split(strip(read(cmd, String)), ",")
        return (; working_set = parse(Float64, fields[1]), private = parse(Float64, fields[2]))
    end
    return (; working_set = NaN, private = NaN)
end

"""
    active_units(graph)

Return the unclamped MNIST units used to convert sweeps to local update steps.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    build_setup(nsamples)

Build the 10x MNIST graph, matching worker layer, and `nsamples` MNIST jobs for
process-reuse diagnostics.
"""
function build_setup(nsamples::T) where {T<:Integer}
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(120_000 + Int(nsamples)),
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
    jobs = [(; x = view(x, :, idx), y = view(y, :, idx)) for idx in 1:Int(nsamples)]
    return (; graph, layer, jobs, params = IsingLearning.read_graph_params(graph), relaxation)
end

"""
    build_workers(layer, graph, params, nworkers)

Create independent MNIST `Process` workers from a shared parameter snapshot.
"""
function build_workers(layer::L, graph::G, params::P, nworkers::T) where {L,G,P,T<:Integer}
    workers = Process[]
    sizehint!(workers, Int(nworkers))
    for _ in 1:Int(nworkers)
        worker_graph = IsingLearning._worker_graph(graph, params)
        push!(workers, IsingLearning._worker_process(layer, worker_graph))
    end
    return workers
end

"""
    assert_worker_indices(worker)

Check the active spin cache and sparse matrix row indices for the worker graph.
"""
function assert_worker_indices(worker::W) where {W<:Process}
    ctx = IsingLearning._mnist_worker_state(worker)
    graph = ctx.model
    n = InteractiveIsing.nstates(graph)
    sp = InteractiveIsing.adj(graph).sp
    active = ctx.free_context.active_spins
    isempty(active) && error("free_context.active_spins is empty")
    minimum(active) >= 1 || error("active spin below one")
    maximum(active) <= n || error("active spin above nstates")
    rowval = SparseArrays.getrowval(sp)
    minimum(rowval) >= 1 || error("sparse row below one")
    maximum(rowval) <= n || error("sparse row above nstates")
    return (; active = length(active), nstates = n, nnz = nnz(sp))
end

"""
    run_reuse_debug!()

Run multiple jobs through each reusable `Process` worker, printing state before
every reuse so the failing job is visible in the log.
"""
function run_reuse_debug!()
    njobs = NWORKERS * JOBS_PER_WORKER
    setup = build_setup(njobs)
    workers = build_workers(setup.layer, setup.graph, setup.params, NWORKERS)

    println(
        "MNIST process reuse debug workers=", NWORKERS,
        " jobs_per_worker=", JOBS_PER_WORKER,
        " hidden=", HIDDEN,
        " sweeps=", SWEEPS,
        " relaxation=", setup.relaxation,
        " checked_contraction=", CHECKED_CONTRACTION,
        " threads=", Threads.nthreads(),
    )
    println("memory_after_workers=", process_memory_mb())

    seconds = @elapsed begin
        Threads.@threads for worker_idx in eachindex(workers)
            worker = workers[worker_idx]
            for local_job in 1:JOBS_PER_WORKER
                job_idx = (local_job - 1) * NWORKERS + worker_idx
                println("worker=", worker_idx, " local_job=", local_job, " before=", assert_worker_indices(worker))
                IsingLearning._write_example!(worker, setup.jobs[job_idx].x, setup.jobs[job_idx].y)
                Processes.reset!(worker)
                Processes.runprocessinline!(worker)
                println("worker=", worker_idx, " local_job=", local_job, " ticks=", Processes.getticks(worker), " runtime=", Processes.runtime(worker))
                flush(stdout)
            end
        end
    end

    println("elapsed=", seconds)
    println("memory_done=", process_memory_mb())
    for worker in workers
        close(worker)
    end
    return nothing
end

run_reuse_debug!()
