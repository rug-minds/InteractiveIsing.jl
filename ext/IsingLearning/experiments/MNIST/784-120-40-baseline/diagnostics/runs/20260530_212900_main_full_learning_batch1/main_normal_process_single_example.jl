const NORMAL_PROCESS_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    NORMAL_PROCESS_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

const START_TIME = time()

"""Print one flushed progress marker for the single normal-Process run."""
function marker(label::S) where {S<:AbstractString}
    println(label, " elapsed=", round(time() - START_TIME; digits = 3))
    flush(stdout)
    return nothing
end

"""Run one normal `Processes.Process` on one MNIST sample with `run` + `wait`."""
function main_normal_process_single_example()
    marker("after include")
    config = updated_config(langevin_learning_config(); batchsize = 1)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    marker("after data")

    setup = build_layer(config)
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
    marker("after worker")

    try
        clear_buffer!(worker_context(worker).buffers)
        load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
        Processes.reset!(worker)
        marker("before run")
        run(worker; phase_beta = config.β)
        marker("after run returned")
        wait(worker)
        marker("after wait")
        fetch(worker)
        marker("after fetch")
    finally
        close(worker)
        marker("after close")
    end
    println("ticks=$(Processes.getticks(worker))")
    println("done")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_normal_process_single_example()
end
