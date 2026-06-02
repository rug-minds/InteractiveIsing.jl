using Dates
using Profile

const RUN_DIR = @__DIR__
const HELPER_PATH = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER_PATH)

"""Run one warmed serial NonGenerated Process sample and print a flat profile."""
function main()
    config = langevin_learning_config()
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    setup = build_layer(config)
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), input_hidden_w_ref)
    looptype = Processes.NonGenerated()

    run_one! = function (sample_idx::I) where {I<:Integer}
        load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
        Processes.reset!(worker)
        @atomic worker.shouldrun = true
        @atomic worker.paused = false
        algo = Processes.getalgo(worker)
        result = Processes.loop(
            worker,
            algo,
            Processes.context(worker),
            Processes.lifetime(worker),
            (; phase_beta = config.β),
            Processes.Resuming{false}(),
            looptype,
        )
        worker.lastresult = result
        worker.loopidx = 1
        @atomic worker.shouldrun = false
        return result
    end

    try
        println(now(), " warmup begin")
        flush(stdout)
        warmup_wall = @elapsed run_one!(1)
        println(now(), " warmup end wall=$(warmup_wall)")
        flush(stdout)

        Profile.clear()
        println(now(), " profile begin")
        flush(stdout)
        wall = @elapsed Profile.@profile run_one!(2)
        println(now(), " profile end wall=$(wall)")
        flush(stdout)
        Profile.print(format = :flat, sortedby = :count, maxdepth = 80)
    finally
        close(worker)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
