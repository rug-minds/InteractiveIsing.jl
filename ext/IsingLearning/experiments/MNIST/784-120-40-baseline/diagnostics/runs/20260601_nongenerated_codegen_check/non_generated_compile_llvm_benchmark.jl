using Dates
using InteractiveUtils
using Statistics

const RUN_DIR = @__DIR__
const HELPER_PATH = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER_PATH)

"""Return field names for a process subcontext container."""
function nt_names(x::T) where {T}
    return Tuple(fieldnames(T))
end

"""Run one MNIST worker sample through the explicit NonGenerated loop path."""
function nongenerated_entry!(worker::P, phase_beta::Float32) where {P<:Processes.Process}
    Processes.runprocessinline!(worker; phase_beta = phase_beta, looptype = Processes.NonGenerated())
    return worker
end

"""Time one full serial Process minibatch using an explicit loop type."""
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
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)

    try
        run_batch! = function ()
            clear_buffer!(worker_context(worker).buffers)
            @inbounds for sample_idx in 1:config.batchsize
                load_sample_into_worker!(worker_context(worker), xtrain, ytrain, sample_idx)
                Processes.reset!(worker)
                Processes.runprocessinline!(worker; phase_beta = config.β, looptype = looptype)
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
        return (; warmup_wall, wall, seconds_per_example = wall / config.batchsize)
    finally
        close(worker)
    end
end

"""Build a one-sample worker suitable for codegen and context-shape probes."""
function build_probe_worker(config::C, xtrain::X, ytrain::Y) where {C<:InputFieldMNISTConfig,X,Y}
    setup = build_layer(config)
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    return worker
end

function main()
    config = updated_config(langevin_learning_config(); batchsize = 128, workers = 1)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    println(now(), " begin NonGenerated check threads=", Threads.nthreads(), " batchsize=", config.batchsize)
    println(now(), " sys_looptype=", Processes.sys_looptype, " explicit_looptype=", Processes.NonGenerated())
    flush(stdout)

    worker = build_probe_worker(config, xtrain, ytrain)
    try
        before = Processes.context(worker)
        println("probe_before_subcontexts=", nt_names(Processes.get_subcontexts(before)))
        println("probe_before_widened=", nt_names(Processes.getwidened(before)))
        flush(stdout)

        compile_wall = @elapsed nongenerated_entry!(worker, config.β)
        after = Processes.context(worker)
        println(now(), " nongenerated first sample wall=", compile_wall)
        println("probe_after_subcontexts=", nt_names(Processes.get_subcontexts(after)))
        println("probe_after_widened=", nt_names(Processes.getwidened(after)))
        println("probe_same_context_type=", typeof(before) === typeof(after))
        flush(stdout)

        llvm_path = joinpath(RUN_DIR, "nongenerated_entry_code_llvm.ll")
        open(llvm_path, "w") do io
            code_llvm(io, nongenerated_entry!, Tuple{typeof(worker), Float32}; raw = true, dump_module = true)
        end
        println(now(), " llvm_path=", llvm_path)
        println(now(), " llvm_bytes=", filesize(llvm_path))
        flush(stdout)
    finally
        close(worker)
    end

    csv = joinpath(RUN_DIR, "nongenerated_vs_generated_direct.csv")
    open(csv, "w") do io
        println(io, "label,looptype,warmup_wall_seconds,wall_seconds,seconds_per_example")

        direct = time_direct_learning_minibatch!(build_layer(config), xtrain, ytrain, config)
        println(io, join(("bespoke_direct", "none", "", direct.wall, direct.seconds_per_example), ","))
        println(now(), " direct wall=", direct.wall, " spe=", direct.seconds_per_example)
        flush(stdout)

        generated = time_serial_process_learning_minibatch!(
            build_layer(config),
            xtrain,
            ytrain,
            config,
            Processes.Generated(),
        )
        println(io, join(("serial_process", string(Processes.Generated()), generated.warmup_wall, generated.wall, generated.seconds_per_example), ","))
        println(now(), " generated warmup=", generated.warmup_wall, " wall=", generated.wall, " spe=", generated.seconds_per_example)
        flush(stdout)

        nongenerated = time_serial_process_learning_minibatch!(
            build_layer(config),
            xtrain,
            ytrain,
            config,
            Processes.NonGenerated(),
        )
        println(io, join(("serial_process", string(Processes.NonGenerated()), nongenerated.warmup_wall, nongenerated.wall, nongenerated.seconds_per_example), ","))
        println(now(), " nongenerated warmup=", nongenerated.warmup_wall, " wall=", nongenerated.wall, " spe=", nongenerated.seconds_per_example)
        flush(stdout)
    end
    println(now(), " csv=", csv)
    return nothing
end

main()
