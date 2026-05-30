using Dates

const DESTRUCTURED_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DESTRUCTURED_MANAGER_FILE = joinpath(DESTRUCTURED_ARCH, "mnist_local_manager_grid.jl")

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "destructured_process_timing"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = "1"
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")

include(DESTRUCTURED_MANAGER_FILE)

using Random

struct DestructuredMNISTWorker{M} <: Processes.ProcessAlgorithm
    model::M
end

"""Print one timestamped diagnostic line."""
function destructured_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Return the free-phase geometric temperature for a destructured phase tuple."""
@inline function destructured_free_temperature(phase::P, step::I, nsteps::J) where {P<:NamedTuple,I<:Integer,J<:Integer}
    config = phase.config
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    return config.hot_temp * (config.cold_temp / config.hot_temp)^progress
end

"""Return the nudged reverse-anneal temperature for a destructured phase tuple."""
@inline function destructured_nudge_temperature(phase::P, step::I, nsteps::J) where {P<:NamedTuple,I<:Integer,J<:Integer}
    config = phase.config
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    progress <= 0.5f0 && return config.cold_temp + (progress / 0.5f0) * (config.reverse_temp - config.cold_temp)
    return config.reverse_temp + ((progress - 0.5f0) / 0.5f0) * (config.cold_temp - config.reverse_temp)
end

"""Randomize a graph state while keeping inactive input spins zero."""
@inline function destructured_randomize_state!(phase::P) where {P<:NamedTuple}
    graph = phase.graph
    rng = phase.rng
    state = II.state(graph)
    @inbounds for idx in eachindex(state)
        state[idx] = rand(rng, Bool) ? one(eltype(state)) : -one(eltype(state))
    end
    fill!(II.state(graph[1]), 0f0)
    return phase
end

"""Run package-level Metropolis steps using only destructured phase fields."""
@inline function destructured_metropolis_phase!(phase::P, proposal, nsteps::I; nudged::Bool) where {P<:NamedTuple,I<:Integer}
    graph = phase.graph
    rng = phase.rng
    hamiltonian = phase.hamiltonian
    proposer = phase.proposer
    metro = II.Metropolis()
    @inbounds for step in 1:Int(nsteps)
        T = nudged ? destructured_nudge_temperature(phase, step, nsteps) : destructured_free_temperature(phase, step, nsteps)
        II.temp!(graph, T)

        proposal = rand(rng, proposer)
        ΔE = II.calculate(II.ΔH(), hamiltonian, graph, proposal)
        if ΔE <= zero(PMNIST_FT) || rand(rng, PMNIST_FT) < exp(-ΔE / T)
            proposal = II.accept(proposer, proposal)
        end
        II.update!(metro, hamiltonian, graph, proposal)
    end
    return proposal
end

"""Run one free read from destructured sample, buffer, and phase tuples."""
@inline function destructured_free_read!(sample::S, buffers::B, phase::P, proposal) where {S<:NamedTuple,B<:NamedTuple,P<:NamedTuple}
    destructured_randomize_state!(phase)
    install_sample_bias!(sample.model, sample.x)
    proposal = destructured_metropolis_phase!(phase, proposal, phase.free_steps; nudged = false)
    energy = graph_energy(phase.graph)
    if energy < buffers.free_best_energy[]
        buffers.free_best_energy[] = energy
        buffers.free_state .= II.state(phase.graph)
    end
    return proposal
end

"""Run one nudged read from destructured sample, buffer, and phase tuples."""
@inline function destructured_nudged_read!(sample::S, buffers::B, phase::P, proposal) where {S<:NamedTuple,B<:NamedTuple,P<:NamedTuple}
    II.state(phase.graph) .= buffers.free_state
    install_nudged_sample_bias!(sample.model, sample.x, sample.y)
    proposal = destructured_metropolis_phase!(phase, proposal, phase.nudge_steps; nudged = true)
    energy = graph_energy(phase.graph)
    if energy < buffers.nudged_best_energy[]
        buffers.nudged_best_energy[] = energy
        buffers.nudged_state .= II.state(phase.graph)
    end
    return proposal
end

"""Reset one capture buffer before repeated free or nudged reads."""
@inline function destructured_reset_capture!(best_energy::Base.RefValue{T}, state::V) where {T<:Real,V<:AbstractVector}
    best_energy[] = PMNIST_FT(Inf)
    fill!(state, 0f0)
    return nothing
end

"""Finish one contrastive sample from destructured tuples and update counters."""
@inline function destructured_finish_sample!(sample::S, buffers::B, counters::C) where {S<:NamedTuple,B<:NamedTuple,C<:NamedTuple}
    stats = finish_contrastive_sample!(
        sample.gradient,
        sample.model,
        sample.x,
        sample.y,
        buffers.free_state,
        buffers.nudged_state,
    )
    update_worker_stats!(counters.nsamples, counters.ncorrect, counters.nskipped, counters.total_loss, stats)
    return stats
end

"""Initialize the steady-state fields for one destructured worker process."""
function Processes.init(worker::DestructuredMNISTWorker{M}, context) where {M<:LocalMNISTModel}
    model = worker.model
    graph = model.graph
    graph_state = II.state(graph)
    hamiltonian = II.init!(graph.hamiltonian, graph)
    proposer = II.get_proposer(graph)
    proposal = rand(model.rng, proposer)
    return (;
        model,
        x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
        y = zeros(PMNIST_FT, PMNIST_NCLASSES * model.config.output_replicas),
        gradient = gradient_buffer(model),
        free_state = similar(graph_state),
        nudged_state = similar(graph_state),
        free_best_energy = Ref(PMNIST_FT(Inf)),
        nudged_best_energy = Ref(PMNIST_FT(Inf)),
        rng = model.rng,
        hamiltonian,
        proposer,
        proposal,
        nsamples = Ref(0),
        ncorrect = Ref(0),
        nskipped = Ref(0),
        total_loss = Ref(0f0),
    )
end

"""Run one sample after destructuring context at the ProcessAlgorithm boundary."""
function Processes.step!(worker::DestructuredMNISTWorker{M}, context) where {M<:LocalMNISTModel}
    (;
        model,
        x,
        y,
        gradient,
        free_state,
        nudged_state,
        free_best_energy,
        nudged_best_energy,
        rng,
        hamiltonian,
        proposer,
        proposal,
        nsamples,
        ncorrect,
        nskipped,
        total_loss,
    ) = context

    config = model.config
    graph = model.graph
    phase = (;
        config,
        graph,
        rng,
        hamiltonian,
        proposer,
        free_steps = max(1, config.free_sweeps * length(II.state(graph))),
        nudge_steps = max(1, config.nudge_sweeps * length(II.state(graph))),
    )
    sample = (; model, x, y, gradient)
    buffers = (; free_state, nudged_state, free_best_energy, nudged_best_energy)
    counters = (; nsamples, ncorrect, nskipped, total_loss)

    next_proposal = proposal
    destructured_reset_capture!(buffers.free_best_energy, buffers.free_state)
    @inbounds for _ in 1:max(1, config.free_reads)
        next_proposal = @inline destructured_free_read!(sample, buffers, phase, next_proposal)
    end

    destructured_reset_capture!(buffers.nudged_best_energy, buffers.nudged_state)
    @inbounds for _ in 1:max(1, config.nudge_reads)
        next_proposal = @inline destructured_nudged_read!(sample, buffers, phase, next_proposal)
    end

    @inline destructured_finish_sample!(sample, buffers, counters)
    return (; proposal = next_proposal)
end

"""Time normal Process runs while keeping context access at the outer boundary."""
function time_destructured_process!(process::P, xtrain::X, ytrain::Y, nsamples::I) where {P<:Processes.Process,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    return @elapsed begin
        @inbounds for sample_idx in 1:Int(nsamples)
            process_state = Processes.context(process)[1]
            process_state.x .= view(xtrain, :, sample_idx)
            process_state.y .= view(ytrain, :, sample_idx)
            Processes.reset!(process)
            @inline run(process)
            @inline wait(process)
        end
    end
end

"""Run the destructured normal-Process timing diagnostic."""
function main()
    nsamples = parse(Int, get(ENV, "ISING_DESTRUCTURED_NSAMPLES", "5"))
    config = LocalMNISTManagerConfig(;
        name = "destructured_process_timing",
        workers = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = @__DIR__,
    )
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    source = init_model(config, config.seed)

    warm_process = Processes.Process(DestructuredMNISTWorker(worker_model(source, 1)); repeats = 1)
    time_destructured_process!(warm_process, xtrain, ytrain, 1)

    process = Processes.Process(DestructuredMNISTWorker(worker_model(source, 2)); repeats = 1)
    seconds = time_destructured_process!(process, xtrain, ytrain, nsamples)
    steps_per_sample = (config.free_reads * config.free_sweeps + config.nudge_reads * config.nudge_sweeps) * length(II.state(source.graph))

    println("path,nsamples,total_seconds,seconds_per_sample,samples_per_second,steps_per_second")
    println(join((
        "destructured_normal_process",
        nsamples,
        round(seconds; digits = 6),
        round(seconds / nsamples; digits = 6),
        round(nsamples / seconds; digits = 6),
        round((nsamples * steps_per_sample) / seconds; digits = 3),
    ), ","))
end

main()
