using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using InteractiveUtils
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_PROFILE_WORKERS", "16"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_PROFILE_HIDDEN", "120"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_PROFILE_BATCHSIZE", "64"))
const NBATCHES = parse(Int, get(ENV, "ISING_MNIST_PROFILE_BATCHES", "2"))
const WARMUP_BATCHES = parse(Int, get(ENV, "ISING_MNIST_PROFILE_WARMUP_BATCHES", "1"))
const MINIT = parse(Int, get(ENV, "ISING_MNIST_PROFILE_MINIT", "1"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_PROFILE_SWEEPS", "5.0"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_PROFILE_LIMIT", string(BATCHSIZE * (NBATCHES + WARMUP_BATCHES))))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_PROFILE_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_PROFILE_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_PROFILE_BETA", "0.1"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_PROFILE_LR", "0.003"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_PROFILE_WEIGHT_SCALE", "0.005"))
const MEASURE_CPU = parse(Bool, get(ENV, "ISING_MNIST_PROFILE_CPU", "true"))
const WRITE_WARNTYPE = parse(Bool, get(ENV, "ISING_MNIST_PROFILE_WARNTYPE", "true"))
const OUTDIR = get(ENV, "ISING_MNIST_PROFILE_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_profiles")))

mkpath(OUTDIR)

function append_csv_row!(path, row)
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

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

function timed_cpu(f)
    cpu0 = process_cpu_seconds()
    wall = @elapsed f()
    cpu1 = process_cpu_seconds()
    cpu_seconds = (isnan(cpu0) || isnan(cpu1)) ? NaN : cpu1 - cpu0
    cpu_threads = (isnan(cpu_seconds) || wall < 0.02) ? NaN : cpu_seconds / wall
    return (; wall, cpu_seconds, cpu_threads)
end

function profile_active_units(graph)
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

function build_profile_trainer(nworkers::Integer)
    local graph
    build_graph_timing = timed_cpu() do
        graph = MNISTArchitecture(
            hidden = HIDDEN,
            precision = Float32,
            weight_scale = WEIGHT_SCALE,
            rng = Random.MersenneTwister(10_000 + nworkers),
        )
        temp!(graph, TEMP)
    end

    relaxation = max(1, round(Int, SWEEPS * profile_active_units(graph)))
    dynamics = LocalLangevin(stepsize = STEPSIZE, adjusted = false)
    local layer
    build_layer_timing = timed_cpu() do
        layer = MNISTLayer(
            graph = graph,
            β = BETA,
            free_relaxation_steps = relaxation,
            nudged_relaxation_steps = relaxation,
            dynamics_algorithm = dynamics,
            nudged_dynamics_algorithm = deepcopy(dynamics),
            validation_algorithm = deepcopy(dynamics),
        )
    end

    local trainer
    build_trainer_timing = timed_cpu() do
        trainer = init_mnist_trainer(layer; graph, numthreads = nworkers, optimiser = Optimisers.Adam(LR))
    end

    return (; graph, layer, trainer, relaxation, build_graph_timing, build_layer_timing, build_trainer_timing)
end

function make_jobs(xbatch, ybatch)
    jobs = NamedTuple[]
    sizehint!(jobs, size(xbatch, 2) * MINIT)
    for sample_idx in axes(xbatch, 2)
        for _ in 1:MINIT
            push!(jobs, (; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)))
        end
    end
    return jobs
end

function buffer_norm(buffer)
    total = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :α) && (total += sum(abs2, buffer.α))
    return sqrt(total)
end

function parameter_bytes(graph)
    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    bytes = sizeof(eltype(graph)) * (length(SparseArrays.getnzval(InteractiveIsing.adj(graph))) + length(b))
    quadratic = IsingLearning.hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)
    isnothing(quadratic) || (bytes += sizeof(eltype(graph)) * length(b))
    return bytes
end

function graph_param_views(graph)
    return (;
        w = SparseArrays.getnzval(InteractiveIsing.adj(graph)),
        b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b),
    )
end

function striped_sync_params!(graphs, params)
    isempty(graphs) && return graphs
    nthreads = Threads.nthreads()

    Threads.@threads for thread_idx in 1:nthreads
        first_w = ((thread_idx - 1) * length(params.w)) ÷ nthreads + 1
        last_w = (thread_idx * length(params.w)) ÷ nthreads
        for graph in graphs
            w = SparseArrays.getnzval(InteractiveIsing.adj(graph))
            @inbounds for idx in first_w:last_w
                w[idx] = params.w[idx]
            end
        end
    end

    Threads.@threads for thread_idx in 1:nthreads
        first_b = ((thread_idx - 1) * length(params.b)) ÷ nthreads + 1
        last_b = (thread_idx * length(params.b)) ÷ nthreads
        for graph in graphs
            b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
            @inbounds for idx in first_b:last_b
                b[idx] = params.b[idx]
            end
        end
    end

    return graphs
end

function graph_threaded_sync_params!(graphs, params)
    Threads.@threads for graph_idx in eachindex(graphs)
        graph = graphs[graph_idx]
        SparseArrays.getnzval(InteractiveIsing.adj(graph)) .= params.w
        InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b) .= params.b
    end
    return graphs
end

function max_sync_error(graphs, params)
    isempty(graphs) && return 0.0
    err = zero(eltype(params.w))
    for graph in graphs
        views = graph_param_views(graph)
        err = max(err, maximum(abs.(views.w .- params.w)))
        err = max(err, maximum(abs.(views.b .- params.b)))
    end
    return Float64(err)
end

function profile_write_state!(trainer, jobs)
    workers = trainer.workers
    nworkers = length(workers)
    timing = timed_cpu() do
        for (job_idx, job) in enumerate(jobs)
            worker = workers[mod1(job_idx, nworkers)]
            IsingLearning._write_example!(worker, job.x, job.y)
        end
    end
    return timing
end

function profile_reset_workers!(trainer)
    manager_slots = collect(slots(trainer.manager))
    timing = timed_cpu() do
        for slot in manager_slots
            resetworker!(slot)
        end
    end
    return timing
end

function profile_apply_input_target!(trainer, jobs)
    graphs = trainer.worker_graphs
    ngraphs = length(graphs)
    timing = timed_cpu() do
        for (job_idx, job) in enumerate(jobs)
            graph = graphs[mod1(job_idx, ngraphs)]
            IsingLearning.apply_input(graph, job.x)
            IsingLearning.apply_targets(graph, job.y)
        end
    end
    return timing
end

function profile_syncs!(trainer)
    graph_list = Any[trainer.prototype_graph]
    append!(graph_list, trainer.worker_graphs)
    push!(graph_list, trainer.validation_graph)

    serial_timing = timed_cpu() do
        IsingLearning._broadcast_params!(trainer)
    end
    striped_timing = timed_cpu() do
        striped_sync_params!(graph_list, trainer.params)
    end
    graph_threaded_timing = timed_cpu() do
        graph_threaded_sync_params!(graph_list, trainer.params)
    end
    sync_error = max_sync_error(graph_list, trainer.params)
    return (; serial_timing, striped_timing, graph_threaded_timing, sync_error, graph_count = length(graph_list))
end

function write_warntype_report(path, trainer, jobs, batch_gradient)
    isempty(jobs) && return path
    worker = first(trainer.workers)
    slot = first(collect(slots(trainer.manager)))
    job = first(jobs)
    graph_list = Any[trainer.prototype_graph]
    append!(graph_list, trainer.worker_graphs)
    push!(graph_list, trainer.validation_graph)

    open(path, "w") do io
        println(io, "Threads.nthreads() = ", Threads.nthreads())
        println(io, "trainer type = ", typeof(trainer))
        println(io, "manager type = ", typeof(trainer.manager))
        println(io, "slot type = ", typeof(slot))
        println(io, "worker type = ", typeof(worker))
        println(io, "job type = ", typeof(job))
        println(io, "batch gradient type = ", typeof(batch_gradient))
        println(io)

        redirect_stdout(io) do
            println("==== _write_example! ====")
            @code_warntype optimize=true IsingLearning._write_example!(worker, job.x, job.y)
            println("==== resetworker! ====")
            @code_warntype optimize=true resetworker!(slot)
            println("==== apply_input ====")
            @code_warntype optimize=true IsingLearning.apply_input(Processes.context(worker).dynamics.model, job.x)
            println("==== apply_targets ====")
            @code_warntype optimize=true IsingLearning.apply_targets(Processes.context(worker).dynamics.model, job.y)
            println("==== run! manager ====")
            @code_warntype optimize=true run!(trainer.manager, jobs)
            println("==== _collect_batch_gradient! ====")
            @code_warntype optimize=true IsingLearning._collect_batch_gradient!(trainer, batch_gradient, length(jobs))
            println("==== _broadcast_params! ====")
            @code_warntype optimize=true IsingLearning._broadcast_params!(trainer)
            println("==== striped_sync_params! ====")
            @code_warntype optimize=true striped_sync_params!(graph_list, trainer.params)
            println("==== graph_threaded_sync_params! ====")
            @code_warntype optimize=true graph_threaded_sync_params!(graph_list, trainer.params)
        end
    end
    return path
end

function run_profile(nworkers::Integer)
    setup = build_profile_trainer(nworkers)
    graph, layer, trainer = setup.graph, setup.layer, setup.trainer
    batch_gradient = IsingLearning.gradient_buffer(graph)
    csv_path = joinpath(OUTDIR, "mnist_profiles.csv")
    warntype_path = joinpath(OUTDIR, "mnist_profiles_warntype_workers$(nworkers).txt")

    local x, y
    load_timing = timed_cpu() do
        x, y = load_mnist_arrays(layer; split = :train, limit = TRAIN_LIMIT)
    end
    loader = MNISTDataLoader(x, y; batchsize = BATCHSIZE, shuffle = false, rng = Random.MersenneTwister(1))

    try
        for (batch_idx, (xbatch, ybatch)) in enumerate(loader)
            batch_idx > NBATCHES + WARMUP_BATCHES && break
            is_warmup = batch_idx <= WARMUP_BATCHES
            measured_batch_idx = batch_idx - WARMUP_BATCHES

            local jobs
            job_build_timing = timed_cpu() do
                jobs = make_jobs(xbatch, ybatch)
            end
            write_state_timing = profile_write_state!(trainer, jobs)
            reset_worker_timing = profile_reset_workers!(trainer)
            apply_input_target_timing = profile_apply_input_target!(trainer, jobs)

            reset_buffers_timing = timed_cpu() do
                IsingLearning._reset_batch_buffers!(trainer)
            end
            manager_run_timing = timed_cpu() do
                run!(trainer.manager, jobs)
            end

            worker_norms = [buffer_norm(Processes.context(worker)._state.buffers) for worker in trainer.workers]

            collect_gradient_timing = timed_cpu() do
                IsingLearning._collect_batch_gradient!(trainer, batch_gradient, length(jobs))
            end
            gradient_norm = buffer_norm(batch_gradient)

            optimiser_timing = timed_cpu() do
                trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
            end
            sync = profile_syncs!(trainer)

            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                workers = nworkers,
                batch = measured_batch_idx,
                warmup_batches = WARMUP_BATCHES,
                hidden = HIDDEN,
                batchsize = size(xbatch, 2),
                jobs = length(jobs),
                minit = MINIT,
                sweeps = SWEEPS,
                relaxation = setup.relaxation,
                graph_states = InteractiveIsing.nstates(graph),
                graph_edges = length(SparseArrays.getnzval(InteractiveIsing.adj(graph))),
                param_bytes_per_graph = parameter_bytes(graph),
                synced_graphs = sync.graph_count,
                build_graph_seconds = setup.build_graph_timing.wall,
                build_graph_cpu_threads = setup.build_graph_timing.cpu_threads,
                build_layer_seconds = setup.build_layer_timing.wall,
                build_layer_cpu_threads = setup.build_layer_timing.cpu_threads,
                build_trainer_seconds = setup.build_trainer_timing.wall,
                build_trainer_cpu_threads = setup.build_trainer_timing.cpu_threads,
                load_seconds = load_timing.wall,
                load_cpu_threads = load_timing.cpu_threads,
                job_build_seconds = job_build_timing.wall,
                job_build_cpu_threads = job_build_timing.cpu_threads,
                write_state_seconds = write_state_timing.wall,
                write_state_cpu_threads = write_state_timing.cpu_threads,
                reset_worker_seconds = reset_worker_timing.wall,
                reset_worker_cpu_threads = reset_worker_timing.cpu_threads,
                apply_input_target_seconds = apply_input_target_timing.wall,
                apply_input_target_cpu_threads = apply_input_target_timing.cpu_threads,
                reset_buffers_seconds = reset_buffers_timing.wall,
                reset_buffers_cpu_threads = reset_buffers_timing.cpu_threads,
                manager_run_seconds = manager_run_timing.wall,
                manager_run_cpu_threads = manager_run_timing.cpu_threads,
                collect_gradient_seconds = collect_gradient_timing.wall,
                collect_gradient_cpu_threads = collect_gradient_timing.cpu_threads,
                optimiser_seconds = optimiser_timing.wall,
                optimiser_cpu_threads = optimiser_timing.cpu_threads,
                serial_sync_seconds = sync.serial_timing.wall,
                serial_sync_cpu_threads = sync.serial_timing.cpu_threads,
                striped_sync_seconds = sync.striped_timing.wall,
                striped_sync_cpu_threads = sync.striped_timing.cpu_threads,
                graph_threaded_sync_seconds = sync.graph_threaded_timing.wall,
                graph_threaded_sync_cpu_threads = sync.graph_threaded_timing.cpu_threads,
                striped_sync_error = sync.sync_error,
                active_workers = count(>(0), worker_norms),
                min_worker_norm = minimum(worker_norms),
                mean_worker_norm = mean(worker_norms),
                max_worker_norm = maximum(worker_norms),
                gradient_norm,
            )
            if !is_warmup
                append_csv_row!(csv_path, row)
                println(row)
                flush(stdout)
            end

            WRITE_WARNTYPE && measured_batch_idx == 1 && write_warntype_report(warntype_path, trainer, jobs, batch_gradient)
        end
    finally
        close_trainer!(trainer)
    end

    return csv_path
end

function main()
    println(
        "MNIST profiles workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " batchsize=", BATCHSIZE,
        " nbatches=", NBATCHES,
        " warmup_batches=", WARMUP_BATCHES,
        " minit=", MINIT,
        " sweeps=", SWEEPS,
        " limit=", TRAIN_LIMIT,
        " temp=", TEMP,
        " stepsize=", STEPSIZE,
        " beta=", BETA,
        " measure_cpu=", MEASURE_CPU,
        " write_warntype=", WRITE_WARNTYPE,
    )
    flush(stdout)

    csv_path = nothing
    for nworkers in WORKERS
        csv_path = run_profile(nworkers)
    end

    println("Saved MNIST profile CSV: ", csv_path)
    println("Saved profile outputs in ", OUTDIR)
end

main()
