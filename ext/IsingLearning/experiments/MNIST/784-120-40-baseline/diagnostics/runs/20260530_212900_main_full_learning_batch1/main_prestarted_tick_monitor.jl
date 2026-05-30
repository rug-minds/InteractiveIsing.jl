const PRESTARTED_MONITOR_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    PRESTARTED_MONITOR_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

"""Run a monitor task before launching the one-example normal Process."""
function main_prestarted_tick_monitor()
    t0 = time()
    println("after include elapsed=", round(time() - t0; digits = 3), " threads=", Threads.nthreads())
    flush(stdout)

    config = updated_config(langevin_learning_config(); batchsize = 1)
    println("after config elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    println("after data elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    setup = build_layer(config)
    println("after build_layer elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    println("after resolve elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
    println("after worker elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    clear_buffer!(worker_context(worker).buffers)
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    println("before monitor elapsed=", round(time() - t0; digits = 3))
    flush(stdout)

    monitor = Threads.@spawn begin
        for poll_idx in 0:24
            poll_idx == 0 || sleep(5)
            println(
                "poll_", poll_idx,
                " elapsed=", round(time() - t0; digits = 3),
                " ticks=", Processes.getticks(worker),
                " loopidx=", Processes.loopint(worker),
                " task_is_nothing=", isnothing(worker.task),
            )
            flush(stdout)
        end
    end

    println("before run elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    run(worker; phase_beta = config.β)
    println("after run returned elapsed=", round(time() - t0; digits = 3))
    flush(stdout)
    fetch(monitor)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_prestarted_tick_monitor()
end
