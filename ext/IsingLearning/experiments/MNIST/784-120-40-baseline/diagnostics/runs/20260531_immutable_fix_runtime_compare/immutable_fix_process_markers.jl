using Dates
using Optimisers

const RUN_DIR = @__DIR__
const HELPER = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER)

"""Print a timestamped progress marker and flush immediately for live polling."""
function logline(message::M) where {M<:AbstractString}
    println(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), " ", message)
    flush(stdout)
    return nothing
end

"""Run the serial Processes full-learning path with markers around each expensive call."""
function main()
    batchsize = parse(Int, get(ENV, "ISING_RUNTIME_COMPARE_BATCHSIZE", "1"))
    config = updated_config(langevin_learning_config(); batchsize = batchsize)

    logline("begin process marker diagnostic threads=$(Threads.nthreads()) batchsize=$(config.batchsize) sweeps=$(config.sweeps)")
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    logline("build setup")
    setup = build_layer(config)
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    initial_params = input_field_params(source_graph, input_hidden_w_ref[])
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
    batch_gradient = input_field_gradient_buffer(source_graph, input_hidden_w_ref[])

    logline("resolve algorithm")
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    logline("build worker")
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)

    try
        run_batch! = function (label::L) where {L<:AbstractString}
            logline("$label clear worker buffers")
            clear_buffer!(worker_context(worker).buffers)
            @inbounds for sample_idx in 1:config.batchsize
                logline("$label sample=$sample_idx load")
                load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
                logline("$label sample=$sample_idx reset begin")
                Processes.reset!(worker)
                logline("$label sample=$sample_idx reset end")
                logline("$label sample=$sample_idx runprocessinline begin")
                Processes.runprocessinline!(worker; phase_beta = config.β)
                logline("$label sample=$sample_idx runprocessinline end")
            end
            logline("$label gradient copy begin")
            clear_buffer!(batch_gradient)
            add_buffer!(batch_gradient, worker_context(worker).buffers)
            scale_buffer!(batch_gradient, inv(FT(config.β) * FT(config.batchsize)))
            logline("$label optimizer begin")
            opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
            IsingLearning.sync_graph_params!(source_graph, (; w = params.w, b = params.b))
            input_hidden_w_ref[] = params.w_input
            logline("$label batch end")
            return nothing
        end

        logline("warmup begin")
        run_batch!("warmup")
        logline("warmup end")

        logline("restore initial params")
        params = initial_params
        opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
        IsingLearning.sync_graph_params!(source_graph, (; w = initial_params.w, b = initial_params.b))
        input_hidden_w_ref[] = initial_params.w_input
        clear_buffer!(worker_context(worker).buffers)

        logline("timed begin")
        wall = @elapsed run_batch!("timed")
        logline("timed end wall=$wall seconds_per_example=$(wall / config.batchsize)")
        return wall
    finally
        logline("close begin")
        close(worker)
        logline("close end")
    end
end

main()
