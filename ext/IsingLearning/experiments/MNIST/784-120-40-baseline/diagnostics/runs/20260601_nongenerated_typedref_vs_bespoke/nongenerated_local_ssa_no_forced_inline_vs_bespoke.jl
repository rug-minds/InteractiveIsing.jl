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

"""Append one benchmark row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
Run the repeat loop with a local context variable and no call-site forced `_step!` inline.

This keeps the context in SSA form for SROA while avoiding the original
`stablecontext = @inline _step!(...)` codegen shape that stalled.
"""
function local_ssa_no_forced_inline_loop(
    process::P,
    algo::F,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:Processes.AbstractProcess,F<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime}
    @assert Processes.isresolved(algo)
    @inline Processes.before_while(process)

    step_plan = @inline Processes.getplan(algo)
    step_wiring = @inline Processes.getwiring(step_plan)
    runtime_context = @inline Processes._merge_runtime_inputs(context, inputs)

    stable_context = Processes._step!(
        step_plan,
        runtime_context,
        step_wiring,
        Processes.Namespace{nothing}(),
        process,
        r,
        Processes.Stable(),
    )
    @inline Processes.tick!(process)
    @inline Processes.inc!(process)

    start_idx = @inline Processes.loopidx(process)
    end_idx = @inline Processes.repeats(r)
    for _ in start_idx:end_idx
        stable_context = Processes._step!(
            step_plan,
            stable_context,
            step_wiring,
            Processes.Namespace{nothing}(),
            process,
            r,
            Processes.Stable(),
        )
        @inline Processes.tick!(process)
        @inline Processes.inc!(process)
        if @inline Processes.breakcondition(r, process, stable_context)
            break
        end
    end
    return @inline Processes.after_while(process, algo, stable_context, context)
end

"""Run one full serial `Process` minibatch through `local_ssa_no_forced_inline_loop`."""
function time_local_ssa_process_learning_minibatch!(
    setup,
    xtrain::X,
    ytrain::Y,
    config::C,
) where {X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig}
    source_graph = setup.graph
    input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
    initial_params = input_field_params(source_graph, input_hidden_w_ref[])
    params = initial_params
    opt_state = Optimisers.setup(Optimisers.Adam(config.lr), initial_params)
    batch_gradient = input_field_gradient_buffer(source_graph, input_hidden_w_ref[])
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)

    try
        run_batch! = function ()
            clear_buffer!(worker_context(worker).buffers)
            @inbounds for sample_idx in 1:config.batchsize
                load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
                Processes.reset!(worker)
                @atomic worker.shouldrun = true
                @atomic worker.paused = false
                algo = Processes.getalgo(worker)
                result = local_ssa_no_forced_inline_loop(
                    worker,
                    algo,
                    Processes.context(worker),
                    Processes.lifetime(worker),
                    (; phase_beta = config.β),
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

"""Benchmark local-SSA NonGenerated stepping against bespoke full learning."""
function main()
    mkpath(RUN_DIR)
    csv_path = joinpath(RUN_DIR, "nongenerated_local_ssa_no_forced_inline_vs_bespoke.csv")
    rm(csv_path; force = true)

    config = langevin_learning_config()
    repeats = parse(Int, get(ENV, "ISING_MNIST_LOCAL_SSA_REPEATS", "1"))
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    println(now(), " begin local-ssa no-forced-inline vs bespoke threads=$(Threads.nthreads()) batchsize=$(config.batchsize) sweeps=$(config.sweeps)")
    flush(stdout)

    rows = NamedTuple[]
    for repeat_idx in 1:repeats
        direct_setup = build_layer(config)
        process_setup = build_layer(config)

        direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
        process = time_local_ssa_process_learning_minibatch!(process_setup, xtrain, ytrain, config)
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            repeat = repeat_idx,
            threads = Threads.nthreads(),
            batchsize = config.batchsize,
            sweeps = config.sweeps,
            direct_seconds = direct.wall,
            direct_seconds_per_example = direct.seconds_per_example,
            process_warmup_seconds = process.warmup_wall,
            process_seconds = process.wall,
            process_seconds_per_example = process.seconds_per_example,
            process_over_bespoke = process.wall / direct.wall,
        )
        append_row!(csv_path, row)
        push!(rows, row)
        println(now(), " rep=$(repeat_idx) direct_spe=$(direct.seconds_per_example) process_spe=$(process.seconds_per_example) over=$(process.wall / direct.wall)")
        flush(stdout)
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
    append_row!(joinpath(RUN_DIR, "nongenerated_local_ssa_no_forced_inline_vs_bespoke_summary.csv"), summary)
    println(now(), " summary=$(summary)")
    println(now(), " csv=$(csv_path)")
    flush(stdout)
    return (; rows, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
