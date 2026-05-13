include(joinpath(@__DIR__, "latency_utils.jl"))

const ROWS = Dict{String,Any}[]
const TARGET = joinpath(@__DIR__, "..", "simple_langevin_xor", "simple_2_4_1_langevin.jl")

measure!(ROWS, "include simple experiment") do
    include(TARGET)
end

"""Return the non-manager simple XOR profile configuration."""
function simple_profile_config()
    return SimpleXorConfig(
        epochs = parse(Int, get(ENV, "ISING_LATENCY_EPOCHS", "2")),
        log_every = 1,
        minit = parse(Int, get(ENV, "ISING_LATENCY_MINIT", "8")),
        eval_repeats = parse(Int, get(ENV, "ISING_LATENCY_EVAL_REPEATS", "16")),
        workers = parse(Int, get(ENV, "ISING_LATENCY_WORKERS", string(max(1, min(Threads.nthreads(), 8))))),
        free_relaxation = parse(Int, get(ENV, "ISING_LATENCY_FREE", "600")),
        nudged_relaxation = parse(Int, get(ENV, "ISING_LATENCY_NUDGED", "600")),
        early_relaxation = parse(Int, get(ENV, "ISING_LATENCY_EARLY", "20")),
        β = parse(FT, get(ENV, "ISING_LATENCY_BETA", "2.0")),
        lr = parse(FT, get(ENV, "ISING_LATENCY_LR", "0.002")),
        weight_decay = parse(FT, get(ENV, "ISING_LATENCY_WEIGHT_DECAY", "1e-4")),
        grad_clip = parse(FT, get(ENV, "ISING_LATENCY_GRAD_CLIP", "50")),
        temp = parse(FT, get(ENV, "ISING_LATENCY_TEMP", "0.005")),
        stepsize = parse(FT, get(ENV, "ISING_LATENCY_STEPSIZE", "0.4")),
        max_drift_fraction = parse(FT, get(ENV, "ISING_LATENCY_MAX_DRIFT", "0.6")),
    )
end

"""Profile the non-manager validation path."""
function profile_simple_eval!(rows, trainer, x, y, config; seed_offset)
    return measure!(rows, "simple eval: evaluate_simple!") do
        evaluate_simple!(trainer, x, y, config; seed_offset)
    end
end

"""Profile one non-manager training epoch in explicit subphases."""
function profile_simple_train_epoch!(rows, trainer, x, y, batch_gradient, epoch, config)
    responses = FT[]
    running = Process[]
    ntraj_ref = Ref(0)
    measure!(rows, "simple train$(epoch): zero buffers") do
        IsingLearning.zero_buffer!(batch_gradient)
        empty!(responses)
        empty!(running)
        ntraj_ref[] = 0
    end
    measure!(rows, "simple train$(epoch): worker loop") do
        for sample_idx in axes(x, 2), init_idx in 1:config.minit
            worker_idx = mod1(ntraj_ref[] + 1, length(trainer.workers))
            worker = trainer.workers[worker_idx]
            seed = config.base_seed + 1_000_000 * epoch + 10_000 * sample_idx + init_idx
            start_training_worker!(worker, view(x, :, sample_idx), view(y, :, sample_idx); seed)
            push!(running, worker)
            ntraj_ref[] += 1
            if length(running) == length(trainer.workers)
                for task_worker in running
                    finish_training_worker!(task_worker, batch_gradient, responses)
                end
                empty!(running)
            end
        end
        for task_worker in running
            finish_training_worker!(task_worker, batch_gradient, responses)
        end
    end
    return measure!(rows, "simple train$(epoch): optimizer/broadcast") do
        ntraj = ntraj_ref[]
        IsingLearning.scale_buffer!(batch_gradient, inv(FT(2) * FT(config.β) * FT(max(ntraj, 1))))
        config.weight_decay > 0 && (batch_gradient.w .+= config.weight_decay .* trainer.params.w)
        config.grad_clip > 0 && clamp!(batch_gradient.w, -config.grad_clip, config.grad_clip)
        config.grad_clip > 0 && clamp!(batch_gradient.b, -config.grad_clip, config.grad_clip)
        grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
        trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
        IsingLearning._broadcast_params!(trainer)
        (; grad_norm, response_norm = isempty(responses) ? zero(FT) : mean(responses))
    end
end

"""Run and save a non-manager simple XOR latency characterization."""
function main()
    config = simple_profile_config()
    outdir = latency_outdir("simple_langevin_latency")
    graph = measure!(ROWS, "simple graph construction") do
        simple_graph(config)
    end
    layer = measure!(ROWS, "simple layer construction") do
        SimpleLayer(graph, config)
    end
    trainer = measure!(ROWS, "simple init_simple_trainer") do
        init_simple_trainer(layer, config; graph, split = false)
    end
    x, y = measure!(ROWS, "simple simple_dataset") do
        simple_dataset()
    end
    batch_gradient = measure!(ROWS, "simple gradient_buffer") do
        IsingLearning.gradient_buffer(trainer.prototype_graph)
    end
    profile_simple_eval!(ROWS, trainer, x, y, config; seed_offset = config.base_seed + 50_000_000)
    profile_simple_train_epoch!(ROWS, trainer, x, y, batch_gradient, 1, config)

    warm_graph = measure!(ROWS, "simple warm graph construction") do
        simple_graph(config)
    end
    warm_layer = measure!(ROWS, "simple warm layer construction") do
        SimpleLayer(warm_graph, config)
    end
    warm_trainer = measure!(ROWS, "simple warm init_simple_trainer") do
        init_simple_trainer(warm_layer, config; graph = warm_graph, split = false)
    end
    warm_batch_gradient = measure!(ROWS, "simple warm gradient_buffer") do
        IsingLearning.gradient_buffer(warm_trainer.prototype_graph)
    end
    profile_simple_eval!(ROWS, warm_trainer, x, y, config; seed_offset = config.base_seed + 60_000_000)
    profile_simple_train_epoch!(ROWS, warm_trainer, x, y, warm_batch_gradient, 101, config)

    csv_path = write_latency_csv(joinpath(outdir, "simple_langevin_latency.csv"), ROWS)
    md_path = write_latency_md(
        joinpath(outdir, "README.md"),
        "Non-Manager Simple Langevin XOR Latency",
        latency_settings(
            workers = config.workers,
            minit = config.minit,
            eval_repeats = config.eval_repeats,
            free_relaxation = config.free_relaxation,
            nudged_relaxation = config.nudged_relaxation,
            temp = config.temp,
            stepsize = config.stepsize,
            beta = config.β,
        ),
        ROWS,
        [
            "`simple train: worker loop` contains process reset, launch, wait/close, and gradient collection for all sample/repeat jobs.",
            "`simple eval: evaluate_simple!` contains all validation repeats and process execution.",
            "Large compile_time values identify first-specialization latency; a second call in the same process should isolate actual runtime.",
        ],
    )
    println("Wrote CSV: ", csv_path)
    println("Wrote report: ", md_path)
    return (; rows = ROWS, csv_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
