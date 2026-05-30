const IMMEDIATE_TICKS_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    IMMEDIATE_TICKS_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

"""Read `ticks` and `loopidx` immediately after launching one normal Process."""
function main_normal_process_immediate_ticks()
    t0 = time()
    println("after include elapsed=", round(time() - t0; digits = 3))
    flush(stdout)

    config = updated_config(langevin_learning_config(); batchsize = 1)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    setup = build_layer(config)
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
    clear_buffer!(worker_context(worker).buffers)
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    println("before run elapsed=", round(time() - t0; digits = 3))
    flush(stdout)

    run(worker; phase_beta = config.β)

    # Do not print/flush between `run` and these reads; avoid yielding to the
    # worker before we capture the immediate counters.
    ticks0 = Processes.getticks(worker)
    loopidx0 = Processes.loopint(worker)
    task_nothing0 = isnothing(worker.task)
    println(
        "after run immediate elapsed=", round(time() - t0; digits = 3),
        " ticks=", ticks0,
        " loopidx=", loopidx0,
        " task_is_nothing=", task_nothing0,
    )
    flush(stdout)

    sleep(2)
    println(
        "after sleep2 elapsed=", round(time() - t0; digits = 3),
        " ticks=", Processes.getticks(worker),
        " loopidx=", Processes.loopint(worker),
        " task_is_nothing=", isnothing(worker.task),
    )
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_normal_process_immediate_ticks()
end
