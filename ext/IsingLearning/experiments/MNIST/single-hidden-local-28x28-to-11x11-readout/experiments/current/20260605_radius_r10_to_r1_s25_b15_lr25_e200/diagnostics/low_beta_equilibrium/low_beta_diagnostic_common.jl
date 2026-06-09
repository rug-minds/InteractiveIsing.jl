include(joinpath(@__DIR__, "..", "..", "..", "..", "..", "mnist_local_manager_grid.jl"))

const LOW_BETA_SERIES_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const LOW_BETA_DEFAULT_CHECKPOINT = normpath(joinpath(LOW_BETA_SERIES_DIR, "r8", "best_params.bin"))

"""Parse a comma-separated `Int` environment value for low-beta diagnostics."""
function lowbeta_parse_ints(value::S) where {S<:AbstractString}
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Parse a comma-separated `Float32` environment value for low-beta diagnostics."""
function lowbeta_parse_floats(value::S) where {S<:AbstractString}
    return [parse(PMNIST_FT, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

Base.@kwdef struct LowBetaDiagnosticConfig{S<:AbstractString}
    checkpoint::S = get(ENV, "ISING_MNIST_LOW_BETA_CHECKPOINT", LOW_BETA_DEFAULT_CHECKPOINT)
    outdir::S = get(ENV, "ISING_MNIST_LOW_BETA_OUTDIR", @__DIR__)
    per_class::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_PER_CLASS", "1"))
    repeat_counts::Vector{Int} = lowbeta_parse_ints(get(ENV, "ISING_MNIST_LOW_BETA_REPEATS", "1,4"))
    burnin_sweeps::Vector{Int} = lowbeta_parse_ints(get(ENV, "ISING_MNIST_LOW_BETA_BURNIN_SWEEPS", "25,50"))
    average_sweeps::Vector{Int} = lowbeta_parse_ints(get(ENV, "ISING_MNIST_LOW_BETA_AVERAGE_SWEEPS", "25"))
    nudge_betas::Vector{PMNIST_FT} = lowbeta_parse_floats(get(ENV, "ISING_MNIST_LOW_BETA_BETAS", "0.05,0.1,0.25,0.5,1.5"))
    sample_every_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_SAMPLE_EVERY_SWEEPS", "1"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_SEED", "991337"))
end

"""Return a checkpoint-backed r8 model and immutable base-bias copy."""
function lowbeta_checkpoint_model(config::C) where {C<:LowBetaDiagnosticConfig}
    saved = load_checkpoint(config.checkpoint)
    base_config = hasproperty(saved, :config) ? saved.config : LocalMNISTManagerConfig(local_radius = 8)
    model_config = copy_config(
        base_config;
        name = "low_beta_equilibrium_diagnostic",
        workers = 1,
        progress = false,
        progress_bar = false,
        train_per_class = config.per_class,
        test_per_class = config.per_class,
        outdir = config.outdir,
    )
    model = init_model(model_config)
    install_checkpoint_params!(model, saved)
    base_bias = copy(base_magfield(model.graph).b)
    sample_buffer = zeros(PMNIST_FT, length(base_bias))
    return (; model, base_bias, sample_buffer, saved)
end

"""Randomize only the active sampled spins of the local-MNIST graph."""
function lowbeta_randomize_active!(model::M, rng::R) where {M<:LocalMNISTModel,R<:Random.AbstractRNG}
    s = II.state(model.graph)
    @inbounds for idx in II.sampling_indices(model.graph)
        s[idx] = rand(rng, Bool) ? one(PMNIST_FT) : -one(PMNIST_FT)
    end
    return model
end

"""Initialize one concrete Metropolis context for manual diagnostic stepping."""
function lowbeta_metropolis_context(model::M, rng::R, T::Real) where {M<:LocalMNISTModel,R<:Random.AbstractRNG}
    ctx = StatefulAlgorithms.init(II.Metropolis(), (; model = model.graph))
    return (; ctx..., rng, T = PMNIST_FT(T))
end

"""Run Metropolis for one step while preserving the concrete context object."""
@inline function lowbeta_step_metropolis!(ctx::C) where {C}
    StatefulAlgorithms.step!(II.Metropolis(), ctx)
    return ctx
end

"""Run a segmented geometric temperature schedule for manual Metropolis steps."""
function lowbeta_run_temperature_steps!(ctx::C, steps::I, start_T::Real, stop_T::Real) where {C,I<:Integer}
    total = max(Int(steps), 1)
    nsegments = min(64, total)
    @inbounds for segment_idx in 1:nsegments
        progress = nsegments == 1 ? 1f0 : PMNIST_FT(segment_idx - 1) / PMNIST_FT(nsegments - 1)
        T = PMNIST_FT(start_T) * (PMNIST_FT(stop_T) / PMNIST_FT(start_T))^progress
        segment_ctx = (; ctx..., T)
        first_step = fld((segment_idx - 1) * total, nsegments) + 1
        last_step = fld(segment_idx * total, nsegments)
        for _ in first_step:last_step
            lowbeta_step_metropolis!(segment_ctx)
        end
    end
    return ctx
end

"""Average output spin states during a cold sampling window."""
function lowbeta_average_output!(
    ctx::C,
    model::M,
    average_steps::I,
    sample_interval::J,
) where {C,M<:LocalMNISTModel,I<:Integer,J<:Integer}
    sum_output = zeros(PMNIST_FT, length(model.output_idxs))
    count = 0
    interval = max(1, Int(sample_interval))
    @inbounds for step_idx in 1:Int(average_steps)
        lowbeta_step_metropolis!(ctx)
        if step_idx % interval == 0
            sum_output .+= @view II.state(model.graph)[model.output_idxs]
            count += 1
        end
    end
    count == 0 && error("time average recorded no output samples")
    return (; ctx, mean_output = sum_output ./ PMNIST_FT(count), count, final_state = copy(II.state(model.graph)))
end

"""Run one free readout trajectory from one random initial state."""
function lowbeta_free_timeavg_readout!(
    model::M,
    x::X,
    base_bias::B,
    sample_buffer::S,
    burnin_sweeps::I,
    average_sweeps::J,
    sample_every_sweeps::K,
    rng::R,
) where {
    M<:LocalMNISTModel,
    X<:AbstractVector,
    B<:AbstractVector,
    S<:AbstractVector,
    I<:Integer,
    J<:Integer,
    K<:Integer,
    R<:Random.AbstractRNG,
}
    nstates = length(II.sampling_indices(model.graph))
    lowbeta_randomize_active!(model, rng)
    install_sample_bias!(model, x, base_bias, sample_buffer)
    ctx = lowbeta_metropolis_context(model, rng, model.config.hot_temp)
    ctx = lowbeta_run_temperature_steps!(ctx, Int(burnin_sweeps) * nstates, model.config.hot_temp, model.config.cold_temp)
    return lowbeta_average_output!(ctx, model, Int(average_sweeps) * nstates, Int(sample_every_sweeps) * nstates)
end

"""Run one nudged readout trajectory from an already-relaxed free state."""
function lowbeta_nudged_timeavg_readout!(
    model::M,
    free_state::F,
    x::X,
    y::Y,
    beta::Real,
    base_bias::B,
    sample_buffer::S,
    burnin_sweeps::I,
    average_sweeps::J,
    sample_every_sweeps::K,
    rng::R,
) where {
    M<:LocalMNISTModel,
    F<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    B<:AbstractVector,
    S<:AbstractVector,
    I<:Integer,
    J<:Integer,
    K<:Integer,
    R<:Random.AbstractRNG,
}
    nstates = length(II.sampling_indices(model.graph))
    II.state(model.graph) .= free_state
    install_sample_bias!(model, x, base_bias, sample_buffer, y, beta)
    ctx = lowbeta_metropolis_context(model, rng, model.config.cold_temp)
    ctx = lowbeta_run_temperature_steps!(ctx, Int(burnin_sweeps) * nstates, model.config.cold_temp, model.config.reverse_temp)
    return lowbeta_average_output!(ctx, model, Int(average_sweeps) * nstates, Int(sample_every_sweeps) * nstates)
end

"""Return class scores from a time-averaged output vector."""
function lowbeta_scores(output::V, replicas::I) where {V<:AbstractVector,I<:Integer}
    return class_scores(output, replicas)
end

"""Return the 1-based class encoded by a repeated-replica target vector."""
function lowbeta_truth(y::Y, replicas::I) where {Y<:AbstractVector,I<:Integer}
    return argmax(class_scores(y, replicas))
end

"""Append a vector of named tuples to a CSV file with one shared header."""
function lowbeta_write_rows(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    isempty(rows) && return path
    mkpath(dirname(path))
    names = Symbol[]
    for row in rows
        for name in propertynames(row)
            name in names || push!(names, name)
        end
    end
    open(path, "w") do io
        println(io, join(names, ","))
        for row in rows
            println(io, join((hasproperty(row, name) ? getproperty(row, name) : "" for name in names), ","))
        end
    end
    return path
end
