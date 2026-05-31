using Dates

const RUN_DIR = @__DIR__
const HELPER = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER)

"""Print a timestamped progress marker and flush immediately."""
function logline(message::M) where {M<:AbstractString}
    println(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), " ", message)
    flush(stdout)
    return nothing
end

"""Return a compact summary of the worker's stored/dynamic context state."""
function context_storage_summary(worker::P) where {P<:Processes.Process}
    algo = Processes.getalgo(worker)
    stored_context = Processes.getstoredcontext(algo)
    current_context = Processes.context(worker)
    runtime_context = getfield(worker, :runtime_context)
    return (;
        runtime_context_is_nothing = isnothing(runtime_context),
        current_matches_stored_type = typeof(current_context) === typeof(stored_context),
        current_type = string(typeof(current_context)),
        stored_type = string(typeof(stored_context)),
        runtime_type = isnothing(runtime_context) ? "nothing" : string(typeof(runtime_context)),
        widened_names = string(fieldnames(typeof(Processes.getwidened(current_context)))),
        input_names = string(fieldnames(typeof(Processes.getruntimeinput(current_context)))),
        global_names = string(fieldnames(typeof(Processes.getglobals(current_context)))),
    )
end

"""Run one Process sample and report whether the worker stays on typed storage."""
function main()
    config = updated_config(langevin_learning_config(); batchsize = 1)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    setup = build_layer(config)
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), input_hidden_w_ref)

    try
        logline("before run $(context_storage_summary(worker))")
        load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
        Processes.reset!(worker)
        logline("after reset $(context_storage_summary(worker))")
        logline("runprocessinline begin")
        Processes.runprocessinline!(worker; phase_beta = config.β)
        logline("runprocessinline end")
        logline("after run $(context_storage_summary(worker))")
    finally
        close(worker)
        logline("closed")
    end
end

main()
