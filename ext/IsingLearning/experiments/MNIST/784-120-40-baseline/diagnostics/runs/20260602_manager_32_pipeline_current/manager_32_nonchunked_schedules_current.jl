using Dates
using Optimisers
using Statistics

const RUN_DIR = @__DIR__
const BASELINE_PATH = normpath(joinpath(RUN_DIR, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(BASELINE_PATH)

"""Append one non-chunked manager schedule timing row to a CSV file."""
function append_nonchunked_schedule_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the current full-learning config used for non-chunked schedule timing."""
function nonchunked_schedule_config()
    return InputFieldMNISTConfig(;
        workers = min(32, Threads.nthreads()),
        epochs = 1,
        batchsize = 128,
        scheduler = "spawn",
        chunk_size = 0,
        train_per_class = 80,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        stepsize = 0.5f0,
        seed = 20260526,
        outdir = String(RUN_DIR),
    )
end

"""Construct a per-example-job manager using the current input-field state shape."""
function nonchunked_input_field_manager(
    layer::L,
    source::G,
    config::C,
    input_hidden_w::R,
) where {L<:IsingLearning.LayeredIsingGraphLayer,G,C<:InputFieldMNISTConfig,R<:Base.RefValue}
    params = input_field_params(source, input_hidden_w[])
    optimiser = Optimisers.Adam(config.lr)
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(layer))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        input_field_gradient_buffer(source, input_hidden_w[]),
        Ref(0),
        Optimisers.setup(optimiser, params),
        Ref(zeros(FT, INPUT_DIM, 0)),
        Ref(zeros(FT, NCLASSES * config.output_replicas, 0)),
        input_hidden_w,
    )
    recipe = (;
        makeworker = (idx, manager) -> input_field_worker(
            algorithm,
            manager.state.layer,
            shared_worker_graph(manager.state.source_graph),
            manager.state.input_hidden_w,
        ),
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            copyto!(ctx.x[], job.x)
            copyto!(ctx.y[], job.y)
            manager.state.nsamples[] += 1
            StatefulAlgorithms.resetworker!(slot)
            return nothing
        end,
        runarguments = (slot, job, manager) -> (; phase_beta = manager.config.β),
        flush! = flush_manager_buffers!,
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = StatefulAlgorithms.NoFlush(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = InputFieldMNISTJob{Vector{FT},Vector{FT}},
    )
end

"""Return the concrete threaded schedule for a non-chunked schedule name."""
function nonchunked_schedule(schedule_name::S) where {S<:Symbol}
    schedule_name === :dynamic && return Processes.Dynamic()
    schedule_name === :greedy && return Processes.Greedy()
    schedule_name === :static && return Processes.Static()
    schedule_name === :channelworkers && return Processes.ChannelWorkers()
    schedule_name === :spawn && return nothing
    throw(ArgumentError("unknown schedule $(schedule_name)"))
end

"""Run one per-example-job manager minibatch and time the full learning pipeline."""
function time_nonchunked_schedule_batch!(
    manager::M,
    jobs_buffer::B,
    xtrain::X,
    ytrain::Y,
    indices::V,
    schedule_name::S,
) where {
    M<:StatefulAlgorithms.ProcessManager,
    B<:InputFieldMNISTJobBuffer,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    V<:AbstractVector{Int},
    S<:Symbol,
}
    fill_seconds = @elapsed jobs = fill_jobs!(jobs_buffer, xtrain, ytrain, indices)
    clear_seconds = @elapsed clear_manager_buffers!(manager)
    schedule = nonchunked_schedule(schedule_name)
    run_seconds = @elapsed begin
        if isnothing(schedule)
            StatefulAlgorithms.run!(manager, jobs)
        else
            StatefulAlgorithms.run!(manager, jobs, schedule)
        end
    end
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
        schedule = String(schedule_name),
        examples = length(indices),
        jobs = length(jobs),
        fill_seconds,
        clear_seconds,
        run_seconds,
        flush_seconds,
        update_seconds,
        sync_seconds,
        total_seconds,
        run_seconds_per_example = run_seconds / length(indices),
        total_seconds_per_example = total_seconds / length(indices),
        examples_per_second = length(indices) / total_seconds,
    )
end

"""Benchmark current non-chunked manager schedules over warmed minibatches."""
function main()
    mkpath(RUN_DIR)
    csv_path = joinpath(RUN_DIR, "manager_32_nonchunked_schedules_current.csv")
    summary_path = joinpath(RUN_DIR, "manager_32_nonchunked_schedules_current_summary.csv")
    rm(csv_path; force = true)
    rm(summary_path; force = true)

    repeats = parse(Int, get(ENV, "ISING_MNIST_NONCHUNKED_SCHEDULE_BATCHES", "3"))
    schedule_names = Tuple(Symbol(strip(part)) for part in split(get(ENV, "ISING_MNIST_NONCHUNKED_SCHEDULES", "greedy,spawn"), ",") if !isempty(strip(part)))
    config = nonchunked_schedule_config()

    println(now(), " begin nonchunked manager schedules threads=$(Threads.nthreads()) workers=$(config.workers) batchsize=$(config.batchsize) sweeps=$(config.sweeps) schedules=$(schedule_names)")
    flush(stdout)
    setup_seconds = @elapsed setup = build_layer(config)
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    jobs_buffer = InputFieldMNISTJobBuffer(config.batchsize, INPUT_DIM, NCLASSES * config.output_replicas)

    all_rows = NamedTuple[]
    for schedule_name in schedule_names
        input_hidden_w = Ref(copy(setup.input_hidden_w))
        manager_seconds = @elapsed manager = nonchunked_input_field_manager(setup.layer, setup.graph, config, input_hidden_w)
        try
            println(now(), " warmup schedule=$(schedule_name)")
            flush(stdout)
            warm_indices = collect(1:config.batchsize)
            warmup = time_nonchunked_schedule_batch!(manager, jobs_buffer, xtrain, ytrain, warm_indices, schedule_name)
            println(now(), " warmup schedule=$(schedule_name) total=$(warmup.total_seconds) spe=$(warmup.total_seconds_per_example) eps=$(warmup.examples_per_second)")
            flush(stdout)

            for batch_idx in 1:repeats
                first_idx = config.batchsize * batch_idx + 1
                last_idx = first_idx + config.batchsize - 1
                indices = collect(first_idx:last_idx)
                timing = time_nonchunked_schedule_batch!(manager, jobs_buffer, xtrain, ytrain, indices, schedule_name)
                row = merge(
                    (;
                        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                        batch = batch_idx,
                        threads = Threads.nthreads(),
                        workers = config.workers,
                        batchsize = config.batchsize,
                        sweeps = config.sweeps,
                        beta = config.β,
                        temp = config.temp,
                        stepsize = config.stepsize,
                        relaxation_steps = setup.relaxation_steps,
                        setup_seconds,
                        data_seconds,
                        manager_seconds,
                    ),
                    timing,
                )
                push!(all_rows, row)
                append_nonchunked_schedule_row!(csv_path, row)
                println(row)
                flush(stdout)
            end
        finally
            close(manager)
        end
    end

    for schedule_name in schedule_names
        rows = [row for row in all_rows if row.schedule == String(schedule_name)]
        isempty(rows) && continue
        spe = map(row -> row.total_seconds_per_example, rows)
        eps = map(row -> row.examples_per_second, rows)
        summary = (;
            schedule = String(schedule_name),
            batches = length(rows),
            threads = Threads.nthreads(),
            workers = config.workers,
            batchsize = config.batchsize,
            sweeps = config.sweeps,
            median_total_seconds_per_example = median(spe),
            mean_total_seconds_per_example = mean(spe),
            median_examples_per_second = median(eps),
            mean_examples_per_second = mean(eps),
        )
        append_nonchunked_schedule_row!(summary_path, summary)
        println(now(), " summary=$(summary)")
        flush(stdout)
    end

    println(now(), " csv=$(csv_path)")
    println(now(), " summary_csv=$(summary_path)")
    flush(stdout)
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
