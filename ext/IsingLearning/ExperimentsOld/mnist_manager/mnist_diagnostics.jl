using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using Statistics

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_DIAG_WORKERS", "16"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_DIAG_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_DIAG_BATCHSIZE", "256"))
const TRAIN_LIMIT = parse(Int, get(ENV, "ISING_MNIST_DIAG_TRAIN_LIMIT", "512"))
const EVAL_LIMIT = parse(Int, get(ENV, "ISING_MNIST_DIAG_EVAL_LIMIT", "128"))
const RELAXATION = parse(Int, get(ENV, "ISING_MNIST_DIAG_RELAXATION", "300"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_DIAG_STEPSIZE", "0.6"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_DIAG_TEMP", "0.005"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_DIAG_BETA", "2.0"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_DIAG_LR", "0.003"))
const OUTDIR = get(ENV, "ISING_MNIST_DIAG_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_diag")))

mkpath(OUTDIR)

function mnist_diag_layer(; lr_sign = one(Float32))
    graph = MNISTArchitecture(hidden = HIDDEN, precision = Float32)
    temp!(graph, TEMP)
    dynamics = LocalLangevin(stepsize = STEPSIZE, adjusted = false)
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        free_relaxation_steps = RELAXATION,
        nudged_relaxation_steps = RELAXATION,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = WORKERS,
        optimiser = Optimisers.Descent(lr_sign * LR),
    )
    return graph, layer, trainer
end

function l2norm(xs)
    return sqrt(sum(abs2, xs))
end

function param_norm(params)
    total = sum(abs2, params.w) + sum(abs2, params.b)
    hasproperty(params, :α) && (total += sum(abs2, params.α))
    return sqrt(total)
end

function buffer_norm(buffer)
    total = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :α) && (total += sum(abs2, buffer.α))
    return sqrt(total)
end

function output_stats(trainer, x, y)
    outputs = zeros(eltype(x), 10, size(x, 2))
    predictions = zeros(Int, size(x, 2))
    targets = zeros(Int, size(x, 2))
    for sample_idx in axes(x, 2)
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(x, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)
        out = collect(IsingLearning._validation_output(trainer))
        outputs[:, sample_idx] .= out
        predictions[sample_idx] = argmax(out) - 1
        targets[sample_idx] = argmax(view(y, :, sample_idx)) - 1
    end
    return (;
        mean_abs_output = mean(abs.(outputs)),
        max_abs_output = maximum(abs.(outputs)),
        output_mean = mean(outputs),
        output_std = std(vec(outputs)),
        pred_counts = [count(==(digit), predictions) for digit in 0:9],
        target_counts = [count(==(digit), targets) for digit in 0:9],
        accuracy = mean(predictions .== targets),
        mse = mean(sum(abs2, outputs .- y; dims = 1)),
    )
end

function run_train_probe(; lr_sign = one(Float32), label = "normal")
    graph, layer, trainer = mnist_diag_layer(; lr_sign)
    xtrain, ytrain = load_mnist_arrays(layer; split = :train, limit = TRAIN_LIMIT)
    xeval = @view xtrain[:, 1:min(EVAL_LIMIT, size(xtrain, 2))]
    yeval = @view ytrain[:, 1:min(EVAL_LIMIT, size(ytrain, 2))]
    batch_gradient = IsingLearning.gradient_buffer(graph)
    loader = MNISTDataLoader(xtrain, ytrain; batchsize = BATCHSIZE, shuffle = false, rng = Random.MersenneTwister(1234))

    before_metrics = IsingLearning.evaluate_mnist!(trainer, xeval, yeval; show_progress = false)
    before_outputs = output_stats(trainer, xeval, yeval)
    println("$(label) before metrics=", before_metrics)
    println("$(label) before outputs=", before_outputs)
    flush(stdout)

    batch_rows = NamedTuple[]
    for (batch_idx, (xbatch, ybatch)) in enumerate(loader)
        IsingLearning._reset_batch_buffers!(trainer)
        jobs = [(; x = view(xbatch, :, sample_idx), y = view(ybatch, :, sample_idx)) for sample_idx in axes(xbatch, 2)]
        run_seconds = @elapsed run!(trainer.manager, jobs)
        worker_norms = [buffer_norm(Processes.context(worker)._state.buffers) for worker in trainer.workers]
        collect_seconds = @elapsed IsingLearning._collect_batch_gradient!(trainer, batch_gradient, size(xbatch, 2))
        old_params = deepcopy(trainer.params)
        old_norm = param_norm(old_params)
        grad_norm = buffer_norm(batch_gradient)
        update_seconds = @elapsed begin
            trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
            IsingLearning._broadcast_params!(trainer)
        end
        delta_w = l2norm(trainer.params.w .- old_params.w)
        delta_b = l2norm(trainer.params.b .- old_params.b)
        row = (;
            label,
            batch = batch_idx,
            run_seconds,
            collect_seconds,
            update_seconds,
            active_workers = count(>(0), worker_norms),
            min_worker_norm = minimum(worker_norms),
            mean_worker_norm = mean(worker_norms),
            max_worker_norm = maximum(worker_norms),
            grad_norm,
            param_norm = old_norm,
            update_w_norm = delta_w,
            update_b_norm = delta_b,
            update_to_param = delta_w / max(old_norm, eps(Float32)),
        )
        push!(batch_rows, row)
        println(row)
        flush(stdout)
    end

    after_metrics = IsingLearning.evaluate_mnist!(trainer, xeval, yeval; show_progress = false)
    after_outputs = output_stats(trainer, xeval, yeval)
    println("$(label) after metrics=", after_metrics)
    println("$(label) after outputs=", after_outputs)
    flush(stdout)
    close_trainer!(trainer)
    return (; label, before_metrics, after_metrics, before_outputs, after_outputs, batch_rows)
end

normal = run_train_probe(; lr_sign = one(Float32), label = "normal_lr")
flipped = run_train_probe(; lr_sign = -one(Float32), label = "negative_lr")

open(joinpath(OUTDIR, "mnist_diagnostics_summary.txt"), "w") do io
    println(io, "normal_before=", normal.before_metrics)
    println(io, "normal_after=", normal.after_metrics)
    println(io, "normal_before_outputs=", normal.before_outputs)
    println(io, "normal_after_outputs=", normal.after_outputs)
    println(io, "negative_before=", flipped.before_metrics)
    println(io, "negative_after=", flipped.after_metrics)
    println(io, "negative_before_outputs=", flipped.before_outputs)
    println(io, "negative_after_outputs=", flipped.after_outputs)
end

println("Saved diagnostics: ", OUTDIR)
println("settings workers=", WORKERS,
    " hidden=", HIDDEN,
    " train_limit=", TRAIN_LIMIT,
    " eval_limit=", EVAL_LIMIT,
    " batchsize=", BATCHSIZE,
    " relaxation=", RELAXATION,
    " stepsize=", STEPSIZE,
    " temp=", TEMP,
    " beta=", BETA,
    " lr=", LR,
    " threads=", Threads.nthreads())
