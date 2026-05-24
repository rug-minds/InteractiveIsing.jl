using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "analytic_2_4_1.jl"))

"""
    curriculum_stages()

Return `(target_scale, epochs, learning_rate)` stages for scalar XOR.
Small targets are learned first so the single output spin does not immediately
saturate into the wrong attractor.
"""
function curriculum_stages()
    return (
        (FT(0.25), parse(Int, get(ENV, "ISING_241_STAGE1_EPOCHS", "1200")), FT(0.0010)),
        (FT(0.50), parse(Int, get(ENV, "ISING_241_STAGE2_EPOCHS", "1200")), FT(0.0008)),
        (FT(1.00), parse(Int, get(ENV, "ISING_241_STAGE3_EPOCHS", "2400")), FT(0.0005)),
    )
end

"""
    reset_optimizer!(trainer, lr)

Restart Adam at a new learning rate for a curriculum stage.
"""
function reset_optimizer!(trainer, lr)
    trainer.optimiser = Optimisers.Adam(lr)
    trainer.opt_state = Optimisers.setup(trainer.optimiser, trainer.params)
    return trainer
end

"""
    copy_params(params)

Deep-copy the trainable parameter NamedTuple so a later epoch cannot mutate the
stored best checkpoint.
"""
copy_params(params) = map(copy, params)

"""Mutable best-checkpoint record used during validation."""
mutable struct Best241{P,M,T}
    mse::T
    epoch::Int
    acc::T
    means::M
    params::P
end

"""
    restore_params!(trainer, params)

Restore a copied parameter checkpoint and broadcast it to all worker graphs.
"""
function restore_params!(trainer, params)
    trainer.params = copy_params(params)
    IsingLearning._broadcast_params!(trainer)
    return trainer
end

"""
    maybe_store_best!(best, epoch, metrics, trainer)

Update `best` when validation MSE improves.
"""
function maybe_store_best!(best, epoch, metrics, trainer)
    if metrics.mse < best.mse
        best.mse = metrics.mse
        best.epoch = epoch
        best.acc = metrics.acc
        best.means = copy(metrics.means)
        best.params = copy_params(trainer.params)
    end
    return best
end

"""
    train_stage!(trainer, x, y, config, rows, batch_gradient, epoch, scale, nepochs, lr)

Train one target-scale stage and append validation rows every `config.log_every`.
Validation is always measured against the full `±1` XOR target.
"""
function train_stage!(trainer, x, yfull, config, rows, batch_gradient, epoch, scale, nepochs, lr, best)
    ystage = scale .* yfull
    xbatch, ybatch = repeated_batch(x, ystage, config.minit)
    reset_optimizer!(trainer, lr)
    println("stage target_scale=", scale, " lr=", lr)
    for _ in 1:nepochs
        epoch += 1
        IsingLearning._run_minibatch!(trainer, xbatch, ybatch, batch_gradient)
        config.weight_decay > 0 && (trainer.params.w .*= (one(FT) - lr * config.weight_decay))
        IsingLearning._broadcast_params!(trainer)
        if epoch % config.log_every == 0
            metrics = evaluate_241!(trainer, x, yfull, config; seed_offset = config.base_seed + 70_000_000)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push_row!(rows, epoch, metrics, grad_norm)
            maybe_store_best!(best, epoch, metrics, trainer)
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc,
                " grad=", round(grad_norm, digits = 4), " means=", round.(metrics.means, digits = 3))
        end
    end
    return epoch
end

"""
    run_curriculum()

Train scalar `2 -> 4 -> 1` XOR from random initialization using unadjusted
`LocalLangevin` and a target-scale curriculum.
"""
function run_curriculum()
    outdir = get(
        ENV,
        "ISING_241_DIR",
        joinpath(@__DIR__, "runs", "simple_2_4_1_curriculum_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)
    config = Analytic241Config(
        epochs = 1,
        log_every = parse(Int, get(ENV, "ISING_241_LOG_EVERY", "300")),
        minit = parse(Int, get(ENV, "ISING_241_MINIT", "1")),
        eval_repeats = parse(Int, get(ENV, "ISING_241_EVAL_REPEATS", "24")),
        free_relaxation = parse(Int, get(ENV, "ISING_241_FREE", "1200")),
        nudged_relaxation = parse(Int, get(ENV, "ISING_241_NUDGED", "1200")),
        β = parse(FT, get(ENV, "ISING_241_BETA", "1.0")),
        target_scale = one(FT),
        lr = FT(0.0008),
        weight_decay = parse(FT, get(ENV, "ISING_241_WEIGHT_DECAY", "1e-4")),
        temp = parse(FT, get(ENV, "ISING_241_TEMP", "0.07")),
        stepsize = parse(FT, get(ENV, "ISING_241_STEPSIZE", "0.8")),
        random_weight_scale = parse(FT, get(ENV, "ISING_241_RANDOM_WEIGHT_SCALE", "0.12")),
        random_bias_scale = parse(FT, get(ENV, "ISING_241_RANDOM_BIAS_SCALE", "0.02")),
        init = :random,
    )
    trainer = trainer_241(config)
    x, yfull = xor_dataset_241()
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    initial = evaluate_241!(trainer, x, yfull, config; seed_offset = config.base_seed + 70_000_000)
    push_row!(rows, 0, initial, zero(FT))
    best_state = Best241(initial.mse, 0, initial.acc, copy(initial.means), copy_params(trainer.params))
    println("epoch=0 mse=", round(initial.mse, digits = 6), " acc=", initial.acc,
        " means=", round.(initial.means, digits = 3))

    epoch = 0
    for (scale, nepochs, lr) in curriculum_stages()
        epoch = train_stage!(trainer, x, yfull, config, rows, batch_gradient, epoch, scale, nepochs, lr, best_state)
    end

    restore_params!(trainer, best_state.params)
    restored = evaluate_241!(trainer, x, yfull, config; seed_offset = config.base_seed + 70_000_000)
    push_row!(rows, epoch + 1, restored, zero(FT))
    println("restored best epoch=", best_state.epoch, " mse=", round(restored.mse, digits = 6),
        " acc=", restored.acc, " means=", round.(restored.means, digits = 3))

    csv_path = write_csv_241(joinpath(outdir, "metrics.csv"), rows)
    png_path = plot_241(joinpath(outdir, "progress.png"), rows)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Scalar 2->4->1 XOR Curriculum")
        println(io)
        println(io, "Random initialization, no local potential, unadjusted `LocalLangevin`.")
        println(io, "The target is ramped from weak scalar clamping to the full `±1` scalar target.")
        println(io)
        println(io, "- T = `$(config.temp)`")
        println(io, "- stepsize = `$(config.stepsize)`")
        println(io, "- beta = `$(config.β)`")
        println(io, "- free/nudged = `$(config.free_relaxation)` / `$(config.nudged_relaxation)`")
        println(io, "- Minit / eval repeats = `$(config.minit)` / `$(config.eval_repeats)`")
        println(io, "- restored best epoch = `$(best_state.epoch)`")
        println(io, "- restored best MSE = `$(restored.mse)`")
        println(io, "- restored best accuracy = `$(restored.acc)`")
        println(io)
        println(io, "| stage target scale | epochs | learning rate |")
        println(io, "|---:|---:|---:|")
        for (scale, nepochs, lr) in curriculum_stages()
            println(io, "| $(scale) | $(nepochs) | $(lr) |")
        end
        println(io)
        println(io, "CSV: `metrics.csv`")
        println(io, "Plot: `progress.png`")
    end
    close_trainer!(trainer)
    println("Saved run: ", outdir)
    return (; outdir, csv_path, png_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_curriculum()
end
