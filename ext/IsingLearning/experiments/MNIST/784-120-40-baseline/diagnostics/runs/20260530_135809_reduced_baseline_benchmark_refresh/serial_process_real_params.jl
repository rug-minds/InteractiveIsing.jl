using Dates
using Optimisers

const SERIAL_REAL_DIR = @__DIR__
const SERIAL_REAL_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(SERIAL_REAL_BASELINE)

"""Append one serial real-parameter timing row."""
function append_serial_real_row!(row::R) where {R<:NamedTuple}
    path = joinpath(SERIAL_REAL_DIR, "serial_process_real_params.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the real baseline timing config used for serial and manager comparison."""
function serial_real_config(; workers::Int = 1)
    return InputFieldMNISTConfig(;
        workers,
        epochs = 1,
        batchsize = 128,
        scheduler = "spawn",
        chunk_size = 0,
        train_per_class = 80,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        stepsize = 0.5f0,
        seed = 20260526,
        outdir = String(SERIAL_REAL_DIR),
    )
end

"""Write one sample into a normal worker and run the current Process path inline."""
function run_one_serial_process_sample!(worker::W, x::X, y::Y, sample_idx::I, config::C) where {
    W<:Processes.Process,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    I<:Integer,
    C<:InputFieldMNISTConfig,
}
    ctx = worker_context(worker)
    load_sample_into_worker!(ctx, x, y, sample_idx)
    Processes.reset!(worker)
    return Processes.runprocessinline!(worker; phase_beta = config.β)
end

"""Measure warmed serial Process samples with the same config as the manager run."""
function measure_serial_process!(setup, xtrain::X, ytrain::Y, config::C) where {X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig}
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(setup.input_hidden_w))
    try
        println("[", Dates.format(now(), "HH:MM:SS"), "] warming serial process")
        flush(stdout)
        warmups = parse(Int, get(ENV, "ISING_MNIST_SERIAL_REAL_WARMUPS", "4"))
        @inbounds for sample_idx in 1:warmups
            run_one_serial_process_sample!(worker, xtrain, ytrain, sample_idx, config)
        end

        rows = NamedTuple[]
        measured = parse(Int, get(ENV, "ISING_MNIST_SERIAL_REAL_EXAMPLES", "8"))
        for sample_idx in (warmups + 1):(warmups + measured)
            seconds = @elapsed run_one_serial_process_sample!(worker, xtrain, ytrain, sample_idx, config)
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                mode = "serial_process_inline",
                threads = Threads.nthreads(),
                workers = 1,
                sample_idx,
                sweeps = config.sweeps,
                beta = config.β,
                temp = config.temp,
                stepsize = config.stepsize,
                relaxation_steps = setup.relaxation_steps,
                examples = 1,
                seconds,
                seconds_per_example = seconds,
                examples_per_second = inv(seconds),
            )
            push!(rows, row)
            append_serial_real_row!(row)
            println(row)
            flush(stdout)
        end
        return rows
    finally
        close(worker)
    end
end

"""Run one 1-worker manager batch for a direct manager-vs-serial sanity point."""
function measure_one_worker_manager!(setup, xtrain::X, ytrain::Y, config::C) where {X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig}
    manager = input_field_manager(setup.layer, setup.graph, config, Ref(setup.input_hidden_w))
    jobs_buffer = InputFieldMNISTChunkBuffer(1, config.batchsize)
    try
        indices = collect(1:config.batchsize)
        set_manager_inputs!(manager, xtrain, ytrain)
        jobs = fill_chunk_jobs!(jobs_buffer, indices, config.batchsize)
        clear_manager_buffers!(manager)
        println("[", Dates.format(now(), "HH:MM:SS"), "] warming one-worker manager")
        flush(stdout)
        manager.state.nsamples[] = sum(length, jobs)
        Processes.run!(manager, jobs)

        clear_manager_buffers!(manager)
        seconds = @elapsed begin
            manager.state.nsamples[] = sum(length, jobs)
            Processes.run!(manager, jobs)
        end
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            mode = "one_worker_manager_chunk",
            threads = Threads.nthreads(),
            workers = 1,
            sample_idx = 0,
            sweeps = config.sweeps,
            beta = config.β,
            temp = config.temp,
            stepsize = config.stepsize,
            relaxation_steps = setup.relaxation_steps,
            examples = length(indices),
            seconds,
            seconds_per_example = seconds / length(indices),
            examples_per_second = length(indices) / seconds,
        )
        append_serial_real_row!(row)
        println(row)
        flush(stdout)
        return row
    finally
        close(manager)
    end
end

"""Run serial real-parameter Process diagnostics."""
function main()
    mkpath(SERIAL_REAL_DIR)
    rm(joinpath(SERIAL_REAL_DIR, "serial_process_real_params.csv"); force = true)
    config = serial_real_config(workers = 1)
    println("[", Dates.format(now(), "HH:MM:SS"), "] building setup")
    flush(stdout)
    setup = build_layer(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    serial_rows = measure_serial_process!(setup, xtrain, ytrain, config)
    manager_row = measure_one_worker_manager!(setup, xtrain, ytrain, config)
    return (; serial_rows, manager_row)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
