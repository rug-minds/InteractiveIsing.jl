const TICK_MONITOR_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    TICK_MONITOR_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

const START_TIME = time()

"""Print one flushed progress line for a non-waiting normal `Process` monitor."""
function marker(label::S) where {S<:AbstractString}
    println(label, " elapsed=", round(time() - START_TIME; digits = 3))
    flush(stdout)
    return nothing
end

"""Print process progress counters without waiting for the worker to finish."""
function print_worker_progress(worker, label::S) where {S<:AbstractString}
    println(
        label,
        " elapsed=", round(time() - START_TIME; digits = 3),
        " ticks=", Processes.getticks(worker),
        " loopidx=", Processes.loopint(worker),
        " task_is_nothing=", isnothing(worker.task),
    )
    flush(stdout)
    return nothing
end

"""Start one single-example Process and sample `ticks`/`loopidx` while it runs."""
function main_normal_process_tick_monitor()
    marker("after include")
    config = updated_config(langevin_learning_config(); batchsize = 1)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    marker("after data")

    setup = build_layer(config)
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
    clear_buffer!(worker_context(worker).buffers)
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    marker("before run")

    run(worker; phase_beta = config.β)
    marker("after run returned")
    print_worker_progress(worker, "poll_0")

    for poll_idx in 1:24
        sleep(5)
        print_worker_progress(worker, "poll_$(poll_idx)")
        task = worker.task
        !isnothing(task) && istaskdone(task) && break
    end

    # Do not call close(worker) here: close waits, which is exactly what this
    # diagnostic is avoiding. The outer harness kills the Julia process if the
    # worker task is still running after the polling window.
    marker("monitor finished polling")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_normal_process_tick_monitor()
end
