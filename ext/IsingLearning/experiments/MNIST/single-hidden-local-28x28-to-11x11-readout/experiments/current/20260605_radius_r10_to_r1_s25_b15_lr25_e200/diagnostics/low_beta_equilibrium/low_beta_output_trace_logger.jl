include(joinpath(@__DIR__, "low_beta_diagnostic_common.jl"))

mutable struct MNISTOutputTraceStore{R}
    rows::R
end

MNISTOutputTraceStore() = MNISTOutputTraceStore(NamedTuple[])

"""
    MNISTOutputTraceLogger(store, output_idxs, output_replicas, sample_idx, repeat_idx, phase)

ProcessAlgorithm-style logger that records class scores from the current graph
state whenever it is scheduled by the diagnostic loop.
"""
struct MNISTOutputTraceLogger{Store,Idxs,Phase} <: StatefulAlgorithms.ProcessAlgorithm
    store::Store
    output_idxs::Idxs
    output_replicas::Int
    sample_idx::Int
    repeat_idx::Int
    phase::Phase
end

StatefulAlgorithms.init(::MNISTOutputTraceLogger, context) = (; sweep = 0)

function StatefulAlgorithms.step!(logger::MNISTOutputTraceLogger, context)
    sweep = context.sweep + 1
    output = @view II.state(context.model)[logger.output_idxs]
    scores = class_scores(output, logger.output_replicas)
    push!(
        logger.store.rows,
        (;
            sample_idx = logger.sample_idx,
            repeat_idx = logger.repeat_idx,
            phase = String(logger.phase),
            sweep,
            pred = argmax(scores) - 1,
            score0 = scores[1],
            score1 = scores[2],
            score2 = scores[3],
            score3 = scores[4],
            score4 = scores[5],
            score5 = scores[6],
            score6 = scores[7],
            score7 = scores[8],
            score8 = scores[9],
            score9 = scores[10],
        ),
    )
    return (; sweep)
end

Base.@kwdef struct LowBetaTraceConfig{S<:AbstractString}
    checkpoint::S = get(ENV, "ISING_MNIST_LOW_BETA_CHECKPOINT", LOW_BETA_DEFAULT_CHECKPOINT)
    outdir::S = get(ENV, "ISING_MNIST_LOW_BETA_OUTDIR", @__DIR__)
    samples::Vector{Int} = lowbeta_parse_ints(get(ENV, "ISING_MNIST_LOW_BETA_TRACE_SAMPLES", "1,2,3"))
    repeats::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_REPEATS", "2"))
    burnin_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_BURNIN_SWEEPS", "80"))
    nudge_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_NUDGE_SWEEPS", "80"))
    beta::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_BETA", "0.1"))
    log_every_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_EVERY_SWEEPS", "1"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_LOW_BETA_TRACE_SEED", "812377"))
end

"""Run manual Metropolis sweeps and call a ProcessAlgorithm trace logger."""
function lowbeta_run_logged_sweeps!(
    ctx::C,
    model::M,
    logger::L,
    logger_context,
    sweeps::I,
    log_every_sweeps::J,
    start_T::Real,
    stop_T::Real,
) where {C,M<:LocalMNISTModel,L<:MNISTOutputTraceLogger,I<:Integer,J<:Integer}
    nstates = length(II.sampling_indices(model.graph))
    log_interval = max(1, Int(log_every_sweeps))
    total_steps = max(1, Int(sweeps) * nstates)
    nsegments = min(64, total_steps)
    logged_sweeps = 0
    @inbounds for segment_idx in 1:nsegments
        progress = nsegments == 1 ? 1f0 : PMNIST_FT(segment_idx - 1) / PMNIST_FT(nsegments - 1)
        T = PMNIST_FT(start_T) * (PMNIST_FT(stop_T) / PMNIST_FT(start_T))^progress
        segment_ctx = (; ctx..., T)
        first_step = fld((segment_idx - 1) * total_steps, nsegments) + 1
        last_step = fld(segment_idx * total_steps, nsegments)
        for step_idx in first_step:last_step
            lowbeta_step_metropolis!(segment_ctx)
            current_sweep = fld(step_idx, nstates)
            if current_sweep > logged_sweeps && current_sweep % log_interval == 0
                logged_sweeps = current_sweep
                logger_context = merge(logger_context, StatefulAlgorithms.step!(logger, logger_context))
            end
        end
    end
    return (; ctx, logger_context)
end

"""Log free and low-beta nudged output trajectories for a few samples."""
function run_low_beta_output_trace_logger(; config::C = LowBetaTraceConfig()) where {C<:LowBetaTraceConfig}
    mkpath(config.outdir)
    diag_config = LowBetaDiagnosticConfig(checkpoint = config.checkpoint, outdir = config.outdir, per_class = 1)
    loaded = lowbeta_checkpoint_model(diag_config)
    model = loaded.model
    base_bias = loaded.base_bias
    sample_buffer = loaded.sample_buffer
    xtest, ytest = balanced_mnist(:test, 1, model.config)
    store = MNISTOutputTraceStore()
    rng = Random.MersenneTwister(config.seed)

    for sample_idx in config.samples
        sample_idx in axes(xtest, 2) || continue
        x = @view xtest[:, sample_idx]
        y = @view ytest[:, sample_idx]
        for repeat_idx in 1:config.repeats
            repeat_rng = Random.MersenneTwister(rand(rng, UInt))
            lowbeta_randomize_active!(model, repeat_rng)
            install_sample_bias!(model, x, base_bias, sample_buffer)
            ctx = lowbeta_metropolis_context(model, repeat_rng, model.config.hot_temp)
            free_logger = MNISTOutputTraceLogger(store, model.output_idxs, model.config.output_replicas, sample_idx, repeat_idx, :free)
            free_logger_ctx = merge((; model = model.graph), StatefulAlgorithms.init(free_logger, (; model = model.graph)))
            logged = lowbeta_run_logged_sweeps!(
                ctx,
                model,
                free_logger,
                free_logger_ctx,
                config.burnin_sweeps,
                config.log_every_sweeps,
                model.config.hot_temp,
                model.config.cold_temp,
            )

            free_state = copy(II.state(model.graph))
            II.state(model.graph) .= free_state
            install_sample_bias!(model, x, base_bias, sample_buffer, y, config.beta)
            nudged_ctx = lowbeta_metropolis_context(model, repeat_rng, model.config.cold_temp)
            nudged_logger = MNISTOutputTraceLogger(store, model.output_idxs, model.config.output_replicas, sample_idx, repeat_idx, :nudged)
            nudged_logger_ctx = merge((; model = model.graph), StatefulAlgorithms.init(nudged_logger, (; model = model.graph)))
            lowbeta_run_logged_sweeps!(
                nudged_ctx,
                model,
                nudged_logger,
                nudged_logger_ctx,
                config.nudge_sweeps,
                config.log_every_sweeps,
                model.config.cold_temp,
                model.config.reverse_temp,
            )
        end
        println("logged sample ", sample_idx)
        flush(stdout)
    end

    csv_path = joinpath(config.outdir, "low_beta_output_trace_logger.csv")
    png_path = joinpath(config.outdir, "low_beta_output_trace_logger.png")
    lowbeta_write_rows(csv_path, store.rows)
    plot_low_beta_output_traces(png_path, store.rows)
    println("wrote ", csv_path)
    println("wrote ", png_path)
    return (; rows = store.rows, csv_path, png_path)
end

"""Plot true class-score trajectories emitted by the trace logger."""
function plot_low_beta_output_traces(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    isempty(rows) && return nothing
    CM = ensure_cairomakie()
    fig = CM.Figure(size = (1200, 800))
    ax = CM.Axis(fig[1, 1], xlabel = "logged sweep", ylabel = "class score", title = "Output class-score traces")
    colors = [:steelblue, :orange, :seagreen, :purple, :firebrick, :gray40]
    groups = unique((row.sample_idx, row.repeat_idx, row.phase) for row in rows)
    for (group_idx, group) in enumerate(groups)
        sample_idx, repeat_idx, phase = group
        group_rows = [row for row in rows if row.sample_idx == sample_idx && row.repeat_idx == repeat_idx && row.phase == phase]
        xs = [row.sweep for row in group_rows]
        scores = [maximum((row.score0, row.score1, row.score2, row.score3, row.score4, row.score5, row.score6, row.score7, row.score8, row.score9)) for row in group_rows]
        CM.lines!(
            ax,
            xs,
            scores;
            color = colors[mod1(group_idx, length(colors))],
            linestyle = phase == "free" ? :solid : :dash,
            label = "s$(sample_idx) r$(repeat_idx) $(phase)",
        )
    end
    CM.axislegend(ax, position = :rb)
    CM.save(path, fig)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_low_beta_output_trace_logger()
end
