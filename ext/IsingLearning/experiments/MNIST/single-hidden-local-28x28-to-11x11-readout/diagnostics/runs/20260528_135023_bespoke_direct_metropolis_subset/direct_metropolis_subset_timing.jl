using Dates
using Random
using SparseArrays

const DIRECT_T0 = time()
const DIRECT_ARCH = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DIRECT_MANAGER_FILE = joinpath(DIRECT_ARCH, "mnist_local_manager_grid.jl")
const DIRECT_OUTDIR = @__DIR__

ENV["ISING_MNIST_PM_PROGRESS"] = "false"
ENV["ISING_MNIST_PM_PROGRESS_BAR"] = "false"
ENV["ISING_MNIST_PM_NAME"] = "bespoke_direct_metropolis_subset"
ENV["ISING_MNIST_PM_DYNAMICS"] = "metropolis"
ENV["ISING_MNIST_PM_WORKERS"] = "1"
ENV["ISING_MNIST_PM_EPOCHS"] = "1"
ENV["ISING_MNIST_PM_BATCHSIZE"] = get(ENV, "ISING_MNIST_PM_BATCHSIZE", "32")
ENV["ISING_MNIST_PM_RADIUS"] = get(ENV, "ISING_MNIST_PM_RADIUS", "8")
ENV["ISING_MNIST_PM_FREE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50")
ENV["ISING_MNIST_PM_NUDGE_SWEEPS"] = get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50")
ENV["ISING_MNIST_PM_FREE_READS"] = get(ENV, "ISING_MNIST_PM_FREE_READS", "3")
ENV["ISING_MNIST_PM_NUDGE_READS"] = get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3")
ENV["ISING_MNIST_PM_OPTIMIZER"] = "adam"
ENV["ISING_MNIST_PM_LR_W0"] = get(ENV, "ISING_MNIST_PM_LR_W0", "0.004")
ENV["ISING_MNIST_PM_LR_W12"] = get(ENV, "ISING_MNIST_PM_LR_W12", "0.004")
ENV["ISING_MNIST_PM_LR_W2O"] = get(ENV, "ISING_MNIST_PM_LR_W2O", "0.004")
ENV["ISING_MNIST_PM_LR_B"] = get(ENV, "ISING_MNIST_PM_LR_B", "0.0004")
ENV["ISING_MNIST_PM_GRADIENT_NORMALIZATION"] = get(ENV, "ISING_MNIST_PM_GRADIENT_NORMALIZATION", "mean")

include(DIRECT_MANAGER_FILE)

"""Print one timestamped diagnostic progress line."""
function direct_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Refresh the combined local field `base_bias + sample_bias` used by direct Metropolis."""
function refresh_total_field!(field::V, model::M) where {V<:AbstractVector,M<:LocalMNISTModel}
    field .= base_magfield(model.graph).b
    field .+= sample_magfield(model.graph).b
    return field
end

"""Randomize graph state and keep inactive input spins at zero."""
function randomize_direct_state!(model::M, rng::R) where {M<:LocalMNISTModel,R<:Random.AbstractRNG}
    state = II.state(model.graph)
    @inbounds for idx in eachindex(state)
        state[idx] = rand(rng, Bool) ? one(eltype(state)) : -one(eltype(state))
    end
    fill!(II.state(model.graph[1]), 0f0)
    return state
end

"""Return the energy of a Bilinear + combined-field Ising graph."""
function direct_graph_energy(
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

"""Return the Metropolis temperature at a geometric-schedule step."""
@inline function geometric_temperature(config::C, step::I, nsteps::J) where {C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    return config.hot_temp * (config.cold_temp / config.hot_temp)^progress
end

"""Return the Metropolis temperature at a reverse-anneal-schedule step."""
@inline function reverse_anneal_temperature(config::C, step::I, nsteps::J) where {C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    progress = nsteps <= 1 ? 1f0 : PMNIST_FT(step - 1) / PMNIST_FT(nsteps - 1)
    if progress <= 0.5f0
        return config.cold_temp + (progress / 0.5f0) * (config.reverse_temp - config.cold_temp)
    end
    return config.reverse_temp + ((progress - 0.5f0) / 0.5f0) * (config.cold_temp - config.reverse_temp)
end

"""Run direct single-spin Metropolis updates against the graph CSC storage."""
function direct_metropolis_phase!(
    model::M,
    active_idxs::V,
    field::F,
    nsteps::I,
    rng::R;
    reverse_anneal::Bool,
) where {M<:LocalMNISTModel,V<:AbstractVector{<:Integer},F<:AbstractVector,I<:Integer,R<:Random.AbstractRNG}
    config = model.config
    A = II.adj(model.graph)
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    nz = SparseArrays.nonzeros(A)
    state = II.state(model.graph)
    accepted = 0
    nactive = length(active_idxs)

    @inbounds for step in 1:Int(nsteps)
        idx = active_idxs[rand(rng, 1:nactive)]
        local_field = field[idx]
        for ptr in colptr[idx]:(colptr[idx + 1] - 1)
            local_field += nz[ptr] * state[rows[ptr]]
        end

        ΔE = 2f0 * state[idx] * local_field
        T = reverse_anneal ? reverse_anneal_temperature(config, step, nsteps) : geometric_temperature(config, step, nsteps)
        if ΔE <= 0f0 || rand(rng, PMNIST_FT) < exp(-ΔE / T)
            state[idx] = -state[idx]
            accepted += 1
        end
    end
    return accepted
end

"""Run one direct free/nudged contrastive sample and accumulate its gradient."""
function direct_contrastive_sample!(
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
    graph = model.graph
    A = II.adj(graph)
    config = model.config
    free_steps = config.free_sweeps * length(II.state(graph))
    nudge_steps = config.nudge_sweeps * length(II.state(graph))
    free_best = PMNIST_FT(Inf)
    nudged_best = PMNIST_FT(Inf)
    accepted_free = 0
    accepted_nudged = 0

    # Free reads start from independent random states; keep the best terminal energy.
    for _ in 1:config.free_reads
        randomize_direct_state!(model, rng)
        install_sample_bias!(model, x)
        refresh_total_field!(buffers.field, model)
        accepted_free += direct_metropolis_phase!(model, active_idxs, buffers.field, free_steps, rng; reverse_anneal = false)
        energy = direct_graph_energy(A, II.state(graph), buffers.field)
        if energy < free_best
            free_best = energy
            buffers.free_state .= II.state(graph)
        end
    end

    # Nudged reads restart from the chosen free state; keep the best terminal energy.
    for _ in 1:config.nudge_reads
        II.state(graph) .= buffers.free_state
        install_nudged_sample_bias!(model, x, y)
        refresh_total_field!(buffers.field, model)
        accepted_nudged += direct_metropolis_phase!(model, active_idxs, buffers.field, nudge_steps, rng; reverse_anneal = true)
        energy = direct_graph_energy(A, II.state(graph), buffers.field)
        if energy < nudged_best
            nudged_best = energy
            buffers.nudged_state .= II.state(graph)
        end
    end

    stats = finish_contrastive_sample!(gradient, model, x, y, buffers.free_state, buffers.nudged_state)
    total_steps = config.free_reads * free_steps + config.nudge_reads * nudge_steps
    return (; stats..., accepted_free, accepted_nudged, total_steps)
end

"""Apply the same Adam update as the ProcessManager path after one direct minibatch."""
function apply_direct_update!(
    optimizer_state::O,
    params::P,
    model::M,
    batch_gradient::G,
    optimizer_gradient::H,
    nsamples::I,
) where {O<:NamedTuple,P<:NamedTuple,M<:LocalMNISTModel,G<:NamedTuple,H<:NamedTuple,I<:Integer}
    write_optimizer_gradient!(optimizer_gradient, batch_gradient, model.config, nsamples)
    state_w, w = Optimisers.update(optimizer_state.w, params.w, optimizer_gradient.w)
    state_b, b = Optimisers.update(optimizer_state.b, params.b, optimizer_gradient.b)
    params = (; w, b)
    install_params!(model, params)
    return (; optimizer_state = (; w = state_w, b = state_b), params = trainable_params(model))
end

"""Run a direct Metropolis training subset without ProcessManager or process algorithms."""
function run_direct_subset!(
    model::M,
    xtrain::X,
    ytrain::Y,
    nsamples::I,
) where {M<:LocalMNISTModel,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    config = model.config
    active_idxs = collect(II.sampling_indices(model.graph.index_set))
    buffers = (;
        field = zeros(PMNIST_FT, length(II.state(model.graph))),
        free_state = similar(II.state(model.graph)),
        nudged_state = similar(II.state(model.graph)),
    )
    batch_gradient = gradient_buffer(model)
    optimizer_gradient = gradient_buffer(model)
    params = trainable_params(model)
    optimizer_state = optimizer_states(config, params)
    rng = Random.MersenneTwister(config.seed + 70_001)

    elapsed = @elapsed begin
        ncorrect = 0
        nskipped = 0
        total_loss = 0f0
        accepted_free = 0
        accepted_nudged = 0
        attempted_steps = 0
        batch_samples = 0
        batch_count = 0

        clear_gradient!(batch_gradient)
        for sample_idx in 1:Int(nsamples)
            result = direct_contrastive_sample!(
                batch_gradient,
                model,
                view(xtrain, :, sample_idx),
                view(ytrain, :, sample_idx),
                buffers,
                active_idxs,
                rng,
            )
            batch_samples += 1
            ncorrect += result.correct ? 1 : 0
            nskipped += result.skipped ? 1 : 0
            total_loss += result.loss
            accepted_free += result.accepted_free
            accepted_nudged += result.accepted_nudged
            attempted_steps += result.total_steps

            if batch_samples == config.batchsize || sample_idx == Int(nsamples)
                update = apply_direct_update!(optimizer_state, params, model, batch_gradient, optimizer_gradient, batch_samples)
                optimizer_state = update.optimizer_state
                params = update.params
                clear_gradient!(batch_gradient)
                batch_samples = 0
                batch_count += 1
            end

            if sample_idx == 1 || sample_idx % 25 == 0 || sample_idx == Int(nsamples)
                direct_log(
                    "direct subset progress";
                    sample = sample_idx,
                    nsamples = Int(nsamples),
                    elapsed_s = round(time() - DIRECT_T0; digits = 3),
                )
            end
        end

        global DIRECT_LAST_STATS = (;
            nsamples = Int(nsamples),
            batch_count,
            accuracy = ncorrect / Int(nsamples),
            loss = total_loss / Int(nsamples),
            skipped = nskipped,
            accepted_free,
            accepted_nudged,
            attempted_steps,
            acceptance_rate = (accepted_free + accepted_nudged) / attempted_steps,
        )
    end
    return merge(DIRECT_LAST_STATS, (; elapsed_seconds = elapsed))
end

"""Write one key-value text settings file for the direct diagnostic."""
function write_direct_settings!(path::P, config::C, subset_n::I, full_n::J) where {P<:AbstractString,C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    open(path, "w") do io
        println(io, "# Direct Metropolis subset timing")
        println(io)
        println(io, "- created: `$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))`")
        println(io, "- manager source: `$DIRECT_MANAGER_FILE`")
        println(io, "- ProcessManager: `false`")
        println(io, "- dynamics: manual direct Metropolis over `II.adj(model.graph)` CSC storage")
        println(io, "- radius: `$(config.local_radius)`")
        println(io, "- free/nudge sweeps: `$(config.free_sweeps)` / `$(config.nudge_sweeps)`")
        println(io, "- free/nudge reads: `$(config.free_reads)` / `$(config.nudge_reads)`")
        println(io, "- batch size: `$(config.batchsize)`")
        println(io, "- measured subset samples: `$subset_n` / `$full_n`")
        println(io, "- subset default: `ceil(full_epoch_samples * 60 / 220)`")
    end
    return path
end

"""Run the warmup and measured direct-Metropolis timing diagnostic."""
function main()
    mkpath(DIRECT_OUTDIR)
    config = LocalMNISTManagerConfig(;
        name = "bespoke_direct_metropolis_subset",
        workers = 1,
        epochs = 1,
        local_radius = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "8")),
        progress = false,
        progress_bar = false,
        outdir = DIRECT_OUTDIR,
    )

    direct_log("loading balanced MNIST subset")
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    full_n = size(xtrain, 2)
    default_subset = ceil(Int, full_n * 60 / 220)
    subset_n = min(full_n, parse(Int, get(ENV, "ISING_MNIST_DIRECT_SUBSET", string(default_subset))))
    warmup_n = min(subset_n, parse(Int, get(ENV, "ISING_MNIST_DIRECT_WARMUP_SUBSET", "1")))
    write_direct_settings!(joinpath(DIRECT_OUTDIR, "settings.md"), config, subset_n, full_n)
    direct_log("direct diagnostic configured"; full_n, subset_n, warmup_n, batchsize = config.batchsize)

    direct_log("warmup model initializing")
    warmup_model = init_model(config, config.seed)
    warmup_stats = run_direct_subset!(warmup_model, xtrain, ytrain, warmup_n)
    direct_log("warmup finished"; elapsed_s = round(warmup_stats.elapsed_seconds; digits = 3), samples = warmup_n)

    direct_log("measured model initializing")
    model = init_model(config, config.seed)
    stats = run_direct_subset!(model, xtrain, ytrain, subset_n)
    estimated_epoch_seconds = stats.elapsed_seconds * full_n / subset_n
    seconds_per_sample = stats.elapsed_seconds / subset_n
    steps_per_second = stats.attempted_steps / stats.elapsed_seconds

    csv_path = joinpath(DIRECT_OUTDIR, "summary.csv")
    open(csv_path, "w") do io
        println(io, "subset_samples,full_epoch_samples,elapsed_seconds,estimated_epoch_seconds,seconds_per_sample,batches,accuracy,loss,skipped,attempted_steps,accepted_free,accepted_nudged,acceptance_rate,steps_per_second")
        println(io, join((
            subset_n,
            full_n,
            round(stats.elapsed_seconds; digits = 6),
            round(estimated_epoch_seconds; digits = 6),
            round(seconds_per_sample; digits = 9),
            stats.batch_count,
            round(stats.accuracy; digits = 6),
            round(stats.loss; digits = 6),
            stats.skipped,
            stats.attempted_steps,
            stats.accepted_free,
            stats.accepted_nudged,
            round(stats.acceptance_rate; digits = 9),
            round(steps_per_second; digits = 3),
        ), ","))
    end

    direct_log(
        "measured direct subset finished";
        subset_n,
        elapsed_s = round(stats.elapsed_seconds; digits = 3),
        estimated_epoch_s = round(estimated_epoch_seconds; digits = 3),
        seconds_per_sample = round(seconds_per_sample; digits = 4),
        steps_per_second = round(steps_per_second; digits = 1),
        summary = csv_path,
    )
    return stats
end

main()
