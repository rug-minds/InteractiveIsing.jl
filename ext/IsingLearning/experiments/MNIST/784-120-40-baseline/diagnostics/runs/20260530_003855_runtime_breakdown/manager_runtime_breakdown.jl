using Dates
using Optimisers

const MANAGER_BREAKDOWN_DIR = @__DIR__
const MANAGER_BREAKDOWN_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(MANAGER_BREAKDOWN_BASELINE)

"""Print one timestamped manager-breakdown diagnostic line."""
function manager_breakdown_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Return a compact baseline config for manager timing diagnostics."""
function manager_breakdown_config()
    workers = min(32, Threads.nthreads())
    return InputFieldMNISTConfig(;
        workers,
        epochs = 1,
        batchsize = 128,
        train_per_class = 60,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        seed = 20260526,
        outdir = String(MANAGER_BREAKDOWN_DIR),
    )
end

"""Construct a diagnostic ProcessManager whose flush is timed manually."""
function diagnostic_input_field_manager(layer::L, source::G, config::C, execution) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    C<:InputFieldMNISTConfig,
}
    params = IsingLearning.read_graph_params(source)
    optimiser = Optimisers.Adam(config.lr)
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(layer))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        IsingLearning.gradient_buffer(source),
        Ref(0),
        Optimisers.setup(optimiser, params),
        Ref(zeros(FT, INPUT_DIM, 0)),
        Ref(zeros(FT, NCLASSES * config.output_replicas, 0)),
    )
    runtime_sums = zeros(Float64, config.workers)
    runtime_counts = zeros(Int, config.workers)
    recipe = (;
        runtime_sums,
        runtime_counts,
        makeworker = (idx, manager) -> input_field_worker(algorithm, manager.state.layer, shared_worker_graph(manager.state.source_graph)),
        loadjob! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x[] = job.x
            ctx.y[] = job.y
            manager.state.nsamples[] += 1
            StatefulAlgorithms.resetworker!(slot)
            return nothing
        end,
        providearguments = (slot, job, manager) -> (; phase_beta = manager.config.β),
        afterjob! = (slot, job, manager) -> begin
            manager.recipe.runtime_sums[slot.idx] += StatefulAlgorithms.runtime(slot.worker)
            manager.recipe.runtime_counts[slot.idx] += 1
            return nothing
        end,
        sync_to_state! = flush_manager_buffers!,
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        sync_policy = StatefulAlgorithms.NoSync(),
        execution,
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = InputFieldMNISTJob{Vector{FT},Vector{FT}},
    )
end

"""Append one manager-breakdown row to CSV."""
function append_manager_breakdown_row!(row::R) where {R<:NamedTuple}
    path = joinpath(MANAGER_BREAKDOWN_DIR, "manager_runtime_breakdown_internal.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the concrete manager execution object for one diagnostic schedule name."""
function manager_execution(schedule_name::S) where {S<:Symbol}
    schedule_name === :dynamic && return StatefulAlgorithms.ThreadedWorkers(StatefulAlgorithms.Dynamic())
    schedule_name === :greedy && return StatefulAlgorithms.ThreadedWorkers(StatefulAlgorithms.Greedy())
    schedule_name === :static && return StatefulAlgorithms.ThreadedWorkers(StatefulAlgorithms.Static())
    schedule_name === :spawn && return nothing
    throw(ArgumentError("unknown schedule $(schedule_name)"))
end

"""Run one manually split manager minibatch and return phase timings."""
function timed_manager_batch!(
    manager::M,
    jobs_buffer::B,
    xtrain::X,
    ytrain::Y,
    indices::V,
    schedule_name::S,
    batch_idx::I,
) where {
    M<:StatefulAlgorithms.ProcessManager,
    B<:InputFieldMNISTJobBuffer,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    V<:AbstractVector{Int},
    S<:Symbol,
    I<:Integer,
}
    fill_seconds = @elapsed jobs = fill_jobs!(jobs_buffer, xtrain, ytrain, indices)
    clear_seconds = @elapsed clear_manager_buffers!(manager)
    fill!(manager.recipe.runtime_sums, 0.0)
    fill!(manager.recipe.runtime_counts, 0)
    run_seconds = @elapsed begin
        StatefulAlgorithms.run!(manager, jobs)
    end
    worker_internal_sum_seconds = sum(manager.recipe.runtime_sums)
    worker_internal_max_seconds = maximum(manager.recipe.runtime_sums)
    worker_internal_min_seconds = minimum(manager.recipe.runtime_sums)
    worker_run_count_min = minimum(manager.recipe.runtime_counts)
    worker_run_count_max = maximum(manager.recipe.runtime_counts)
    flush_seconds = @elapsed flush_manager_buffers!(manager)
    update_seconds = @elapsed begin
        manager.state.opt_state, params = Optimisers.update(
            manager.state.opt_state,
            manager.state.params[],
            manager.state.batch_gradient,
        )
        manager.state.params[] = params
    end
    sync_seconds = @elapsed sync_after_update!(manager, manager.state.params[])
    total_seconds = fill_seconds + clear_seconds + run_seconds + flush_seconds + update_seconds + sync_seconds
    return (;
        batch = Int(batch_idx),
        schedule = String(schedule_name),
        examples = length(indices),
        fill_seconds,
        clear_seconds,
        run_seconds,
        flush_seconds,
        update_seconds,
        sync_seconds,
        total_seconds,
        seconds_per_example = total_seconds / length(indices),
        run_seconds_per_example = run_seconds / length(indices),
        worker_internal_sum_seconds,
        worker_internal_max_seconds,
        worker_internal_min_seconds,
        worker_run_count_min,
        worker_run_count_max,
        effective_parallelism = worker_internal_sum_seconds / run_seconds,
    )
end

"""Measure manager runtime split over warmed minibatches for dynamic and static scheduling."""
function main()
    mkpath(MANAGER_BREAKDOWN_DIR)
    config = manager_breakdown_config()
    manager_breakdown_log("building baseline for manager breakdown"; threads = Threads.nthreads(), workers = config.workers)
    setup_seconds = @elapsed setup = build_layer(config)
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    jobs_buffer = InputFieldMNISTJobBuffer(config.batchsize, INPUT_DIM, NCLASSES * config.output_replicas)

    measured_batches = parse(Int, get(ENV, "ISING_MNIST_MANAGER_DIAG_BATCHES", "3"))
    schedules = config.workers >= Threads.maxthreadid() ? (:dynamic, :greedy, :static, :spawn) : (:dynamic, :greedy, :spawn)
    config.workers < Threads.maxthreadid() && manager_breakdown_log(
        "skipping static schedule because it requires one slot per max thread";
        workers = config.workers,
        maxthreadid = Threads.maxthreadid(),
    )
    all_rows = NamedTuple[]

    for schedule_name in schedules
        manager_breakdown_log("constructing diagnostic manager"; schedule = schedule_name)
        execution = manager_execution(schedule_name)
        manager_seconds = @elapsed manager = diagnostic_input_field_manager(setup.layer, setup.graph, config, execution)
        try
            warm_indices = collect(1:config.batchsize)
            manager_breakdown_log("warming manager batch"; schedule = schedule_name)
            warmup = timed_manager_batch!(manager, jobs_buffer, xtrain, ytrain, warm_indices, schedule_name, 0)
            manager_breakdown_log("warmup done"; warmup...)

            for batch_idx in 1:measured_batches
                start_idx = config.batchsize * batch_idx + 1
                stop_idx = start_idx + config.batchsize - 1
                indices = collect(start_idx:stop_idx)
                row = merge(
                    (;
                        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                        architecture = "784-120-40",
                        threads = Threads.nthreads(),
                        workers = config.workers,
                        sweeps = config.sweeps,
                        beta = config.β,
                        relaxation_steps = setup.relaxation_steps,
                        setup_seconds,
                        data_seconds,
                        manager_seconds,
                    ),
                    timed_manager_batch!(manager, jobs_buffer, xtrain, ytrain, indices, schedule_name, batch_idx),
                )
                push!(all_rows, row)
                append_manager_breakdown_row!(row)
                manager_breakdown_log("measured manager batch"; row...)
            end
        finally
            close(manager)
        end
    end

    manager_breakdown_log("manager breakdown complete"; rows = length(all_rows))
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
