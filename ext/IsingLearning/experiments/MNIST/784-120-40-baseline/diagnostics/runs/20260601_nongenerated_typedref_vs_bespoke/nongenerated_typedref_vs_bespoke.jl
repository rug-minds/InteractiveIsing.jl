using Dates
using Optimisers
using Statistics

const RUN_DIR = @__DIR__
const HELPER_PATH = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER_PATH)

"""Append one benchmark row to the run-local comparison CSV."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run one full serial `Process` minibatch with an explicit loop algorithm."""
function time_serial_process_learning_minibatch!(
    setup,
    xtrain::X,
    ytrain::Y,
    config::C,
    looptype::LT,
) where {X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig,LT}
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    initial_params = input_field_params(source_graph, input_hidden_w_ref[])
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
    batch_gradient = input_field_gradient_buffer(source_graph, input_hidden_w_ref[])
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)

    try
        run_batch! = function ()
            clear_buffer!(worker_context(worker).buffers)
            @inbounds for sample_idx in 1:config.batchsize
                load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
                StatefulAlgorithms.reset!(worker)
                @atomic worker.shouldrun = true
                @atomic worker.paused = false
                algo = StatefulAlgorithms.getalgo(worker)
                result = StatefulAlgorithms.loop(
                    worker,
                    algo,
                    StatefulAlgorithms.context(worker),
                    StatefulAlgorithms.lifetime(worker),
                    (; phase_beta = config.β),
                    StatefulAlgorithms.Resuming{false}(),
                    looptype,
                )
                worker.lastresult = result
                worker.loopidx = 1
                @atomic worker.shouldrun = false
            end
            clear_buffer!(batch_gradient)
            add_buffer!(batch_gradient, worker_context(worker).buffers)
            scale_buffer!(batch_gradient, inv(FT(config.β) * FT(config.batchsize)))
            opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
            IsingLearning.sync_graph_params!(source_graph, (; w = params.w, b = params.b))
            input_hidden_w_ref[] = params.w_input
            return nothing
        end

        warmup_wall = @elapsed run_batch!()
        params = initial_params
        opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
        IsingLearning.sync_graph_params!(source_graph, (; w = initial_params.w, b = initial_params.b))
        input_hidden_w_ref[] = initial_params.w_input
        clear_buffer!(worker_context(worker).buffers)

        wall = @elapsed run_batch!()
        return (; wall, warmup_wall, seconds_per_example = wall / config.batchsize)
    finally
        close(worker)
    end
end

"""Benchmark bespoke full learning against serial StatefulAlgorithms full learning."""
function main()
    mkpath(RUN_DIR)
    csv_path = joinpath(RUN_DIR, "nongenerated_typedref_vs_bespoke.csv")
    rm(csv_path; force = true)

    config = langevin_learning_config()
    repeats = parse(Int, get(ENV, "ISING_MNIST_TYPEDREF_REPEATS", "3"))
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    looptype = StatefulAlgorithms.NonGenerated()

    println(now(), " begin typed-ref NonGenerated vs bespoke threads=$(Threads.nthreads()) batchsize=$(config.batchsize) sweeps=$(config.sweeps) looptype=$(looptype)")
    flush(stdout)

    rows = NamedTuple[]
    for repeat_idx in 1:repeats
        direct_setup = build_layer(config)
        process_setup = build_layer(config)

        println(now(), " direct rep=$(repeat_idx) begin")
        flush(stdout)
        direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
        println(now(), " direct rep=$(repeat_idx) wall=$(direct.wall) spe=$(direct.seconds_per_example)")
        flush(stdout)

        println(now(), " process rep=$(repeat_idx) begin")
        flush(stdout)
        process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config, looptype)
        println(now(), " process rep=$(repeat_idx) warmup=$(process.warmup_wall) wall=$(process.wall) spe=$(process.seconds_per_example) over_direct=$(process.wall / direct.wall)")
        flush(stdout)

        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            repeat = repeat_idx,
            threads = Threads.nthreads(),
            batchsize = config.batchsize,
            sweeps = config.sweeps,
            looptype = string(looptype),
            direct_seconds = direct.wall,
            direct_seconds_per_example = direct.seconds_per_example,
            process_warmup_seconds = process.warmup_wall,
            process_seconds = process.wall,
            process_seconds_per_example = process.seconds_per_example,
            process_over_bespoke = process.wall / direct.wall,
        )
        append_row!(csv_path, row)
        push!(rows, row)
    end

    process_over = map(row -> row.process_over_bespoke, rows)
    direct_spe = map(row -> row.direct_seconds_per_example, rows)
    process_spe = map(row -> row.process_seconds_per_example, rows)
    summary = (;
        repeats,
        direct_median_spe = median(direct_spe),
        process_median_spe = median(process_spe),
        process_over_bespoke_median = median(process_over),
        process_over_bespoke_mean = mean(process_over),
    )
    append_row!(joinpath(RUN_DIR, "nongenerated_typedref_vs_bespoke_summary.csv"), summary)
    println(now(), " summary=$(summary)")
    println(now(), " csv=$(csv_path)")
    flush(stdout)
    return (; rows, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
