include("mnist_784_120_40_nudgeless_adam.jl")

"""Return the last row with validation metrics from one completed run."""
function last_evaluated_row(rows::R) where {R<:AbstractVector}
    for row in Iterators.reverse(rows)
        ismissing(row.test_accuracy) || return row
    end
    return last(rows)
end

"""Run a targeted local grid for the beta-zero covariance learning rule."""
function run_nudgeless_grid()
    root = joinpath(@__DIR__, "experiments", "current", "nudgeless_covariance_grid_32w_20ep")
    mkpath(root)
    summary_path = joinpath(root, "grid_summary.csv")
    isfile(summary_path) && rm(summary_path)

    base = (;
        workers = 32,
        epochs = 20,
        batchsize = 128,
        train_per_class = 20,
        test_per_class = 20,
        train_eval_per_class = 20,
        eval_every = 5,
        sweeps = FT(0.05),
        covariance_samples = 20,
        covariance_sample_sweeps = FT(0.02),
        covariance_kick_steps = 2,
        covariance_stepsize = FT(1.0),
        covariance_noise_temp_factor = FT(1.0),
        stepsize = FT(0.05),
        weight_scale = FT(0.005),
        weight_decay = FT(0.0),
        seed = 20260526,
    )

    specs = [
        (; name = "signp_T5e-4_lr1e-4_mix002", gradient_sign = FT(1), temp = FT(5f-4), lr = FT(1f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signm_T5e-4_lr1e-4_mix002", gradient_sign = FT(-1), temp = FT(5f-4), lr = FT(1f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signp_T5e-4_lr3e-4_mix002", gradient_sign = FT(1), temp = FT(5f-4), lr = FT(3f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signm_T5e-4_lr3e-4_mix002", gradient_sign = FT(-1), temp = FT(5f-4), lr = FT(3f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signp_T1e-3_lr1e-4_mix002", gradient_sign = FT(1), temp = FT(1f-3), lr = FT(1f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signm_T1e-3_lr1e-4_mix002", gradient_sign = FT(-1), temp = FT(1f-3), lr = FT(1f-4), covariance_sample_sweeps = FT(0.02), covariance_stepsize = FT(1.0), covariance_noise_temp_factor = FT(1.0), covariance_kick_steps = 2),
        (; name = "signp_T5e-4_lr1e-4_mix010_softkick", gradient_sign = FT(1), temp = FT(5f-4), lr = FT(1f-4), covariance_sample_sweeps = FT(0.10), covariance_stepsize = FT(0.5), covariance_noise_temp_factor = FT(0.5), covariance_kick_steps = 1),
        (; name = "signm_T5e-4_lr1e-4_mix010_softkick", gradient_sign = FT(-1), temp = FT(5f-4), lr = FT(1f-4), covariance_sample_sweeps = FT(0.10), covariance_stepsize = FT(0.5), covariance_noise_temp_factor = FT(0.5), covariance_kick_steps = 1),
    ]

    summaries = NamedTuple[]
    for (idx, spec) in pairs(specs)
        outdir = joinpath(root, spec.name)
        println("grid $(idx)/$(length(specs)): ", spec.name)
        config = InputFieldMNISTConfig(;
            base...,
            temp = spec.temp,
            lr = spec.lr,
            gradient_sign = spec.gradient_sign,
            covariance_sample_sweeps = spec.covariance_sample_sweeps,
            covariance_stepsize = spec.covariance_stepsize,
            covariance_noise_temp_factor = spec.covariance_noise_temp_factor,
            covariance_kick_steps = spec.covariance_kick_steps,
            outdir,
        )
        result = run_config!(config)
        final_row = last(result.rows)
        eval_row = last_evaluated_row(result.rows)
        train_seconds = [
            row.seconds for row in result.rows
            if row.epoch >= 2 && !ismissing(row.seconds)
        ]
        steady_epoch_seconds = isempty(train_seconds) ? missing : sum(train_seconds) / length(train_seconds)
        summary = (;
            idx,
            name = spec.name,
            gradient_sign = spec.gradient_sign,
            temp = spec.temp,
            lr = spec.lr,
            covariance_sample_sweeps = spec.covariance_sample_sweeps,
            covariance_stepsize = spec.covariance_stepsize,
            covariance_noise_temp_factor = spec.covariance_noise_temp_factor,
            covariance_kick_steps = spec.covariance_kick_steps,
            best_accuracy = result.best_accuracy,
            final_epoch = final_row.epoch,
            final_test_accuracy = eval_row.test_accuracy,
            final_test_loss = eval_row.test_loss,
            final_train_accuracy = eval_row.train_accuracy,
            final_train_loss = eval_row.train_loss,
            final_grad_w_norm = final_row.grad_w_norm,
            final_grad_b_norm = final_row.grad_b_norm,
            final_grad_w_input_norm = final_row.grad_w_input_norm,
            steady_epoch_seconds,
            outdir,
        )
        append_row!(summary_path, summary)
        push!(summaries, summary)
        println(summary)
        flush(stdout)
    end
    return summaries
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_nudgeless_grid()
end
