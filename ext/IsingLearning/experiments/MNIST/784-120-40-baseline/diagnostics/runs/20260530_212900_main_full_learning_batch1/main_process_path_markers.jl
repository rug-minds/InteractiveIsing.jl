const PROCESS_MARKER_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    PROCESS_MARKER_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

"""Print a timestamped marker and flush immediately for hang localization."""
function marker(label::S) where {S<:AbstractString}
    println(label, " elapsed=", round(time() - START_TIME; digits = 3))
    flush(stdout)
    return nothing
end

const START_TIME = time()

"""Run the serial Process path step by step, without the bespoke control path."""
function main_process_path_markers()
    marker("after include")
    config = updated_config(langevin_learning_config(); batchsize = 1)
    marker("after config")
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    marker("after data")
    setup = build_layer(config)
    marker("after build_layer")

    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    initial_params = input_field_params(source_graph, input_hidden_w_ref[])
    marker("after params")
    batch_gradient = input_field_gradient_buffer(source_graph, input_hidden_w_ref[])
    marker("after batch_gradient")
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    marker("after resolve")
    worker_graph_ref = shared_worker_graph(source_graph)
    marker("after shared_worker_graph")
    worker = input_field_worker(algorithm, setup.layer, worker_graph_ref, input_hidden_w_ref)
    marker("after input_field_worker")

    try
        clear_buffer!(worker_context(worker).buffers)
        marker("after clear worker buffers")
        load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
        marker("after load sample")
        Processes.reset!(worker)
        marker("after reset")
        Processes.runprocessinline!(worker; phase_beta = config.β)
        marker("after runprocessinline")
        clear_buffer!(batch_gradient)
        add_buffer!(batch_gradient, worker_context(worker).buffers)
        scale_buffer!(batch_gradient, inv(FT(config.β) * FT(config.batchsize)))
        marker("after gradient copy scale")
    finally
        close(worker)
        marker("after close")
    end
    println("done")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_process_path_markers()
end
