using Dates

const SINGLE_PROCESS_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SINGLE_PROCESS_MANAGER_FILE = joinpath(SINGLE_PROCESS_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "single_process_inline_vs_process"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = "1"
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")

include(SINGLE_PROCESS_MANAGER_FILE)

using Random
using SparseArrays

"""Print one timestamped diagnostic line."""
function single_process_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Refresh the direct-loop combined magnetic field buffer."""
function direct_refresh_field!(field::V, model::M) where {V<:AbstractVector,M<:LocalMNISTModel}
    field .= base_magfield(model.graph).b
    field .+= sample_magfield(model.graph).b
    return field
end

"""Randomize one graph state while leaving the inactive input layer at zero."""
function direct_randomize_state!(model::M, rng::R) where {M<:LocalMNISTModel,R<:Random.AbstractRNG}
    s = II.state(model.graph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(rng, Bool) ? one(eltype(s)) : -one(eltype(s))
    end
    fill!(II.state(model.graph[1]), 0f0)
    return s
end

"""Compute graph energy for the direct-loop Bilinear + fused-field Hamiltonian."""
function direct_energy(
    A::SparseMatrixCSC{T,I},
    state::S,
    field::F,
) where {T,I,S<:AbstractVector,F<:AbstractVector}
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    nz = SparseArrays.nonzeros(A)
    energy = zero(eltype(state))
    @inbounds for col in 1:size(A, 2)
        scol = state[col]
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            energy -= eltype(state)(0.5) * nz[ptr] * state[rows[ptr]] * scol
        end
        energy -= field[col] * scol
    end
    return energy
end

"""Return the direct-loop geometric free-phase temperature."""
@inline function direct_free_temperature(config::C, step::I, nsteps::J) where {C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    return config.hot_temp * (config.cold_temp / config.hot_temp)^progress
end

"""Return the direct-loop reverse-anneal nudged-phase temperature."""
@inline function direct_nudge_temperature(config::C, step::I, nsteps::J) where {C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    progress <= 0.5f0 && return config.cold_temp + (progress / 0.5f0) * (config.reverse_temp - config.cold_temp)
    return config.reverse_temp + ((progress - 0.5f0) / 0.5f0) * (config.cold_temp - config.reverse_temp)
end

"""Run one direct Metropolis phase against CSC columns."""
function direct_phase!(
    model::M,
    active_idxs::V,
    field::F,
    nsteps::I,
    rng::R;
    nudged::Bool,
) where {M<:LocalMNISTModel,V<:AbstractVector{<:Integer},F<:AbstractVector,I<:Integer,R<:Random.AbstractRNG}
    config = model.config
    A = II.adj(model.graph)
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    nz = SparseArrays.nonzeros(A)
    state = II.state(model.graph)
    nactive = length(active_idxs)

    @inbounds for step in 1:Int(nsteps)
        idx = active_idxs[rand(rng, 1:nactive)]
        local_field = field[idx]
        for ptr in colptr[idx]:(colptr[idx + 1] - 1)
            local_field += nz[ptr] * state[rows[ptr]]
        end
        ΔE = 2f0 * state[idx] * local_field
        T = nudged ? direct_nudge_temperature(config, step, nsteps) : direct_free_temperature(config, step, nsteps)
        if ΔE <= 0f0 || rand(rng, PMNIST_FT) < exp(-ΔE / T)
            state[idx] = -state[idx]
        end
    end
    return model
end

"""Run one direct contrastive sample with the same high-level work as the worker."""
function direct_one_sample!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    buffers::B,
    active_idxs::V,
    rng::R,
) where {
    G<:NamedTuple,
    M<:LocalMNISTModel,
    X<:AbstractVector,
    Y<:AbstractVector,
    B<:NamedTuple,
    V<:AbstractVector{<:Integer},
    R<:Random.AbstractRNG,
}
    config = model.config
    A = II.adj(model.graph)
    free_steps = config.free_sweeps * length(II.state(model.graph))
    nudge_steps = config.nudge_sweeps * length(II.state(model.graph))
    free_best = PMNIST_FT(Inf)
    nudged_best = PMNIST_FT(Inf)

    for _ in 1:config.free_reads
        direct_randomize_state!(model, rng)
        install_sample_bias!(model, x)
        direct_refresh_field!(buffers.field, model)
        direct_phase!(model, active_idxs, buffers.field, free_steps, rng; nudged = false)
        energy = direct_energy(A, II.state(model.graph), buffers.field)
        if energy < free_best
            free_best = energy
            buffers.free_state .= II.state(model.graph)
        end
    end

    for _ in 1:config.nudge_reads
        II.state(model.graph) .= buffers.free_state
        install_nudged_sample_bias!(model, x, y)
        direct_refresh_field!(buffers.field, model)
        direct_phase!(model, active_idxs, buffers.field, nudge_steps, rng; nudged = true)
        energy = direct_energy(A, II.state(model.graph), buffers.field)
        if energy < nudged_best
            nudged_best = energy
            buffers.nudged_state .= II.state(model.graph)
        end
    end

    return finish_contrastive_sample!(gradient, model, x, y, buffers.free_state, buffers.nudged_state)
end

"""Create an isolated normal `Process` worker using the real MNIST worker algorithm."""
function normal_worker_process(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    algorithm = StatefulAlgorithms.resolve(local_worker_algorithm(mnist_dynamics_algorithm(), source.config, length(II.state(source.graph))))
    return local_worker(source, worker_idx, algorithm)
end

"""Create an isolated `InlineProcess` worker using the real MNIST worker algorithm."""
function inline_worker_process(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    model = worker_model(source, worker_idx)
    graph_state = II.state(model.graph)
    algorithm = StatefulAlgorithms.resolve(local_worker_algorithm(mnist_dynamics_algorithm(), source.config, length(graph_state)))

    # InlineProcess stores a concrete context type. The worker routine widens a
    # few subcontexts on the first run (`current_T`, `stats`, `energy`), so warm a
    # normal process once and use that stable post-run context for the inline run.
    warm_process = StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            mnist_model = model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            y = zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas),
            gradient = gradient_buffer(model),
            free_state = similar(graph_state),
            nudged_state = similar(graph_state),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            nudged_best_energy = Ref(PMNIST_FT(Inf)),
            rng = model.rng,
            nsamples = Ref(0),
            ncorrect = Ref(0),
            nskipped = Ref(0),
            total_loss = Ref(0f0),
        ),
        StatefulAlgorithms.Init(:dynamics; model = model.graph);
        repeat = 1,
    )
    run(warm_process)
    wait(warm_process)
    stable_context = StatefulAlgorithms.context(warm_process)

    return StatefulAlgorithms.InlineProcess(
        algorithm;
        context = stable_context,
        repeats = 1,
        threaded = false,
    )
end

"""Set one worker context's sample input and target."""
@inline function set_worker_sample!(context, xtrain::X, ytrain::Y, sample_idx::I) where {X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    state_context = context._state
    state_context.x .= view(xtrain, :, Int(sample_idx))
    state_context.y .= view(ytrain, :, Int(sample_idx))
    return context
end

"""Time a few samples through a normal asynchronous `Process` worker."""
@inline function time_normal_process_samples!(worker::P, xtrain::X, ytrain::Y, nsamples::I) where {P<:StatefulAlgorithms.Process,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    ctx = worker_context(worker)
    return @elapsed begin
        @inbounds for sample_idx in 1:Int(nsamples)
            ctx.x .= view(xtrain, :, sample_idx)
            ctx.y .= view(ytrain, :, sample_idx)
            StatefulAlgorithms.reset!(worker)
            @inline run(worker)
            @inline wait(worker)
        end
    end
end

"""Time a few samples through a synchronous `InlineProcess` worker."""
@inline function time_inline_process_samples!(worker::P, xtrain::X, ytrain::Y, nsamples::I) where {P<:StatefulAlgorithms.InlineProcess,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    return @elapsed begin
        @inbounds for sample_idx in 1:Int(nsamples)
            @inline set_worker_sample!(StatefulAlgorithms.context(worker), xtrain, ytrain, sample_idx)
            @inline run(worker; threaded = false)
        end
    end
end

"""Time the direct fused sample loop for the same sample count."""
function time_direct_samples!(model::M, xtrain::X, ytrain::Y, nsamples::I) where {M<:LocalMNISTModel,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    gradient = gradient_buffer(model)
    buffers = (;
        field = zeros(PMNIST_FT, length(II.state(model.graph))),
        free_state = similar(II.state(model.graph)),
        nudged_state = similar(II.state(model.graph)),
    )
    active_idxs = collect(II.sampling_indices(model.graph.index_set))
    rng = Random.MersenneTwister(model.config.seed + 90_001)
    return @elapsed begin
        @inbounds for sample_idx in 1:Int(nsamples)
            direct_one_sample!(gradient, model, view(xtrain, :, sample_idx), view(ytrain, :, sample_idx), buffers, active_idxs, rng)
        end
    end
end

"""Run direct, normal Process, and InlineProcess timing for a few samples."""
function main()
    nsamples = parse(Int, get(ENV, "ISING_SINGLE_PROCESS_NSAMPLES", "5"))
    config = LocalMNISTManagerConfig(;
        name = "single_process_inline_vs_process",
        workers = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = @__DIR__,
    )
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    steps_per_sample = (config.free_reads * config.free_sweeps + config.nudge_reads * config.nudge_sweeps) * length(II.state(init_model(config, config.seed).graph))

    single_process_log("constructing workers"; nsamples, steps_per_sample)
    direct_model = init_model(config, config.seed)
    normal_source = init_model(config, config.seed)
    inline_source = init_model(config, config.seed)
    normal_worker = normal_worker_process(normal_source, 1)
    inline_worker = inline_worker_process(inline_source, 1)

    # Warm each path once so the printed measurements are not first-call compilation.
    time_direct_samples!(direct_model, xtrain, ytrain, 1)
    time_normal_process_samples!(normal_worker, xtrain, ytrain, 1)
    time_inline_process_samples!(inline_worker, xtrain, ytrain, 1)

    direct_model = init_model(config, config.seed)
    normal_worker = normal_worker_process(init_model(config, config.seed), 1)
    inline_worker = inline_worker_process(init_model(config, config.seed), 1)

    direct_seconds = time_direct_samples!(direct_model, xtrain, ytrain, nsamples)
    normal_seconds = time_normal_process_samples!(normal_worker, xtrain, ytrain, nsamples)
    inline_seconds = time_inline_process_samples!(inline_worker, xtrain, ytrain, nsamples)

    println("path,nsamples,total_seconds,seconds_per_sample,samples_per_second,steps_per_second")
    for (label, seconds) in (
        ("direct_fused", direct_seconds),
        ("normal_process", normal_seconds),
        ("inline_process", inline_seconds),
    )
        println(join((
            label,
            nsamples,
            round(seconds; digits = 6),
            round(seconds / nsamples; digits = 6),
            round(nsamples / seconds; digits = 6),
            round((nsamples * steps_per_sample) / seconds; digits = 3),
        ), ","))
    end
end

main()
