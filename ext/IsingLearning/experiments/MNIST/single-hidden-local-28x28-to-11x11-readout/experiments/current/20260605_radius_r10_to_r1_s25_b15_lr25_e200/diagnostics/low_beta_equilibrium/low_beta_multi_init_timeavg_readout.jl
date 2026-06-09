include(joinpath(@__DIR__, "low_beta_diagnostic_common.jl"))

"""Return prediction-count text for a vector of 1-based predictions."""
function lowbeta_prediction_counts(predictions::V) where {V<:AbstractVector{Int}}
    counts = zeros(Int, PMNIST_NCLASSES)
    @inbounds for pred in predictions
        counts[pred] += 1
    end
    return join(counts, "-")
end

"""Evaluate free and low-beta nudged readout after multi-init time averaging."""
function run_low_beta_multi_init_timeavg(; config::C = LowBetaDiagnosticConfig()) where {C<:LowBetaDiagnosticConfig}
    mkpath(config.outdir)
    loaded = lowbeta_checkpoint_model(config)
    model = loaded.model
    base_bias = loaded.base_bias
    sample_buffer = loaded.sample_buffer
    xtest, ytest = balanced_mnist(:test, config.per_class, model.config)

    rows = NamedTuple[]
    sample_rows = NamedTuple[]
    max_repeats = maximum(config.repeat_counts)
    rng = Random.MersenneTwister(config.seed)
    println("low-beta readout diagnostic")
    println("checkpoint=", config.checkpoint)
    println("samples=", size(xtest, 2), " max_repeats=", max_repeats)
    flush(stdout)

    for burnin_sweeps in config.burnin_sweeps
        for average_sweeps in config.average_sweeps
            free_scores_by_repeat = Array{PMNIST_FT,3}(undef, PMNIST_NCLASSES, size(xtest, 2), max_repeats)
            free_states = Vector{Vector{PMNIST_FT}}(undef, size(xtest, 2) * max_repeats)

            for sample_idx in axes(xtest, 2)
                x = @view xtest[:, sample_idx]
                y = @view ytest[:, sample_idx]
                truth = lowbeta_truth(y, model.config.output_replicas)
                for repeat_idx in 1:max_repeats
                    repeat_rng = Random.MersenneTwister(rand(rng, UInt))
                    free = lowbeta_free_timeavg_readout!(
                        model,
                        x,
                        base_bias,
                        sample_buffer,
                        burnin_sweeps,
                        average_sweeps,
                        config.sample_every_sweeps,
                        repeat_rng,
                    )
                    scores = lowbeta_scores(free.mean_output, model.config.output_replicas)
                    free_scores_by_repeat[:, sample_idx, repeat_idx] .= scores
                    free_states[(sample_idx - 1) * max_repeats + repeat_idx] = free.final_state
                    push!(sample_rows, (;
                        mode = "free_repeat",
                        sample_idx,
                        repeat_idx,
                        burnin_sweeps,
                        average_sweeps,
                        beta = "",
                        truth = truth - 1,
                        pred = argmax(scores) - 1,
                        target_score = scores[truth],
                        max_score = maximum(scores),
                        score_margin = scores[truth] - maximum(scores[d] for d in 1:PMNIST_NCLASSES if d != truth),
                    ))
                end
            end

            for repeat_count in config.repeat_counts
                preds = Int[]
                losses = PMNIST_FT[]
                margins = PMNIST_FT[]
                for sample_idx in axes(xtest, 2)
                    y = @view ytest[:, sample_idx]
                    truth = lowbeta_truth(y, model.config.output_replicas)
                    scores = vec(sum(@view(free_scores_by_repeat[:, sample_idx, 1:repeat_count]); dims = 2)) ./ PMNIST_FT(repeat_count)
                    pred = argmax(scores)
                    push!(preds, pred)
                    push!(losses, sum(abs2, y .- repeat(scores, inner = model.config.output_replicas)) / 2)
                    push!(margins, scores[truth] - maximum(scores[d] for d in 1:PMNIST_NCLASSES if d != truth))
                end
                accuracy = mean(preds .== [lowbeta_truth(@view(ytest[:, idx]), model.config.output_replicas) for idx in axes(ytest, 2)])
                push!(rows, (;
                    mode = "free_multi_init_timeavg",
                    repeat_count,
                    burnin_sweeps,
                    average_sweeps,
                    beta = "",
                    samples = size(xtest, 2),
                    accuracy,
                    mean_loss = mean(losses),
                    mean_target_margin = mean(margins),
                    pred_counts = lowbeta_prediction_counts(preds),
                ))
            end

            for beta in config.nudge_betas
                nudged_scores_by_repeat = Array{PMNIST_FT,3}(undef, PMNIST_NCLASSES, size(xtest, 2), max_repeats)
                for sample_idx in axes(xtest, 2)
                    x = @view xtest[:, sample_idx]
                    y = @view ytest[:, sample_idx]
                    truth = lowbeta_truth(y, model.config.output_replicas)
                    for repeat_idx in 1:max_repeats
                        repeat_rng = Random.MersenneTwister(rand(rng, UInt))
                        free_state = free_states[(sample_idx - 1) * max_repeats + repeat_idx]
                        nudged = lowbeta_nudged_timeavg_readout!(
                            model,
                            free_state,
                            x,
                            y,
                            beta,
                            base_bias,
                            sample_buffer,
                            burnin_sweeps,
                            average_sweeps,
                            config.sample_every_sweeps,
                            repeat_rng,
                        )
                        scores = lowbeta_scores(nudged.mean_output, model.config.output_replicas)
                        nudged_scores_by_repeat[:, sample_idx, repeat_idx] .= scores
                        push!(sample_rows, (;
                            mode = "nudged_repeat",
                            sample_idx,
                            repeat_idx,
                            burnin_sweeps,
                            average_sweeps,
                            beta,
                            truth = truth - 1,
                            pred = argmax(scores) - 1,
                            target_score = scores[truth],
                            max_score = maximum(scores),
                            score_margin = scores[truth] - maximum(scores[d] for d in 1:PMNIST_NCLASSES if d != truth),
                        ))
                    end
                end

                for repeat_count in config.repeat_counts
                    preds = Int[]
                    margins = PMNIST_FT[]
                    target_shifts = PMNIST_FT[]
                    for sample_idx in axes(xtest, 2)
                        y = @view ytest[:, sample_idx]
                        truth = lowbeta_truth(y, model.config.output_replicas)
                        free_scores = vec(sum(@view(free_scores_by_repeat[:, sample_idx, 1:repeat_count]); dims = 2)) ./ PMNIST_FT(repeat_count)
                        nudged_scores = vec(sum(@view(nudged_scores_by_repeat[:, sample_idx, 1:repeat_count]); dims = 2)) ./ PMNIST_FT(repeat_count)
                        pred = argmax(nudged_scores)
                        push!(preds, pred)
                        push!(margins, nudged_scores[truth] - maximum(nudged_scores[d] for d in 1:PMNIST_NCLASSES if d != truth))
                        push!(target_shifts, nudged_scores[truth] - free_scores[truth])
                    end
                    accuracy = mean(preds .== [lowbeta_truth(@view(ytest[:, idx]), model.config.output_replicas) for idx in axes(ytest, 2)])
                    push!(rows, (;
                        mode = "nudged_multi_init_timeavg",
                        repeat_count,
                        burnin_sweeps,
                        average_sweeps,
                        beta,
                        samples = size(xtest, 2),
                        accuracy,
                        mean_loss = "",
                        mean_target_margin = mean(margins),
                        pred_counts = lowbeta_prediction_counts(preds),
                        mean_target_shift = mean(target_shifts),
                    ))
                end
            end

            println("finished burnin=", burnin_sweeps, " average=", average_sweeps)
            flush(stdout)
        end
    end

    summary_csv = joinpath(config.outdir, "low_beta_multi_init_timeavg_summary.csv")
    samples_csv = joinpath(config.outdir, "low_beta_multi_init_timeavg_samples.csv")
    lowbeta_write_rows(summary_csv, rows)
    lowbeta_write_rows(samples_csv, sample_rows)
    plot_low_beta_summary(joinpath(config.outdir, "low_beta_multi_init_timeavg_summary.png"), rows)
    println("wrote ", summary_csv)
    println("wrote ", samples_csv)
    return (; rows, sample_rows, summary_csv, samples_csv)
end

"""Plot aggregate low-beta readout accuracy and target response."""
function plot_low_beta_summary(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    CM = ensure_cairomakie()
    fig = CM.Figure(size = (1200, 760))
    ax_acc = CM.Axis(fig[1, 1], xlabel = "setting", ylabel = "accuracy", title = "Low-beta readout accuracy")
    ax_shift = CM.Axis(fig[2, 1], xlabel = "setting", ylabel = "mean target shift", title = "Nudged target-score shift")
    labels = String[]
    accs = Float64[]
    shifts_x = Int[]
    shifts = Float64[]
    for row in rows
        push!(labels, string(row.mode, " r", row.repeat_count, " b", row.burnin_sweeps, " a", row.average_sweeps, " β", row.beta))
        push!(accs, Float64(row.accuracy))
        if hasproperty(row, :mean_target_shift)
            push!(shifts_x, length(labels))
            push!(shifts, Float64(row.mean_target_shift))
        end
    end
    xs = collect(eachindex(accs))
    CM.barplot!(ax_acc, xs, accs, color = :steelblue)
    !isempty(shifts) && CM.scatterlines!(ax_shift, shifts_x, shifts, color = :orange)
    ax_acc.xticks = (xs, labels)
    ax_shift.xticks = (xs, labels)
    ax_acc.xticklabelrotation = pi / 3
    ax_shift.xticklabelrotation = pi / 3
    CM.save(path, fig)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_low_beta_multi_init_timeavg()
end
