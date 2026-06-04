using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using LinearAlgebra: dot
using Optimisers
using Random
using Statistics

const WORKERS = parse(Int, get(ENV, "ISING_MNIST_PHASE_WORKERS", "16"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_PHASE_HIDDEN", string(MNIST_DEFAULT_HIDDEN)))
const NSAMPLES = parse(Int, get(ENV, "ISING_MNIST_PHASE_SAMPLES", "64"))
const RELAXATION = parse(Int, get(ENV, "ISING_MNIST_PHASE_RELAXATION", "300"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_PHASE_STEPSIZE", "0.6"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_PHASE_TEMP", "0.005"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_PHASE_BETA", "2.0"))
const LR = parse(Float32, get(ENV, "ISING_MNIST_PHASE_LR", "0.003"))
const OUTDIR = get(ENV, "ISING_MNIST_PHASE_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_phase")))

mkpath(OUTDIR)

function l2rms(xs)
    isempty(xs) && return 0.0
    return sqrt(mean(abs2, xs))
end

function sqerr(a, b)
    return sum(abs2, a .- b)
end

function cosine(a, b)
    denom = sqrt(sum(abs2, a)) * sqrt(sum(abs2, b))
    denom == 0 && return 0.0
    return dot(a, b) / denom
end

function buffer_norm(buffer)
    total = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :α) && (total += sum(abs2, buffer.α))
    return sqrt(total)
end

function build_trainer()
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
    trainer = init_mnist_trainer(layer; graph, numthreads = WORKERS, optimiser = Optimisers.Descent(LR))
    return graph, layer, trainer
end

function worker_phase_row(layer, worker)
    ctx = StatefulAlgorithms.context(worker)
    free = ctx._state.equilibrium_state
    plus = ctx.plus_capture.captured
    minus = ctx.minus_capture.captured
    y = ctx._state.y

    input_idxs = layer.input_layer
    hidden_idxs = InteractiveIsing.layerrange(layer.model_graph[2])
    output_idxs = layer.output_layer

    free_out = @view free[output_idxs]
    plus_out = @view plus[output_idxs]
    minus_out = @view minus[output_idxs]
    target_dir = y .- free_out
    plus_dir = plus_out .- free_out
    minus_dir = minus_out .- free_out
    pm_dir = plus_out .- minus_out

    free_loss = sqerr(free_out, y)
    plus_loss = sqerr(plus_out, y)
    minus_loss = sqerr(minus_out, y)

    return (;
        digit = argmax(y) - 1,
        free_prediction = argmax(free_out) - 1,
        plus_prediction = argmax(plus_out) - 1,
        minus_prediction = argmax(minus_out) - 1,
        free_loss,
        plus_loss,
        minus_loss,
        plus_minus_free_loss = plus_loss - free_loss,
        minus_minus_free_loss = minus_loss - free_loss,
        plus_target_dot = dot(plus_dir, target_dir),
        minus_target_dot = dot(minus_dir, target_dir),
        plusminus_target_dot = dot(pm_dir, target_dir),
        plus_target_cos = cosine(plus_dir, target_dir),
        minus_target_cos = cosine(minus_dir, target_dir),
        plusminus_target_cos = cosine(pm_dir, target_dir),
        plus_output_rms = l2rms(plus_out .- free_out),
        minus_output_rms = l2rms(minus_out .- free_out),
        plusminus_output_rms = l2rms(plus_out .- minus_out),
        plus_hidden_rms = l2rms((@view plus[hidden_idxs]) .- (@view free[hidden_idxs])),
        minus_hidden_rms = l2rms((@view minus[hidden_idxs]) .- (@view free[hidden_idxs])),
        plusminus_hidden_rms = l2rms((@view plus[hidden_idxs]) .- (@view minus[hidden_idxs])),
        plus_input_rms = l2rms((@view plus[input_idxs]) .- (@view free[input_idxs])),
        minus_input_rms = l2rms((@view minus[input_idxs]) .- (@view free[input_idxs])),
        worker_buffer_norm = buffer_norm(ctx._state.buffers),
    )
end

function write_rows(path, rows)
    isempty(rows) && return nothing
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(names, ","))
        for row in rows
            println(io, join((getproperty(row, name) for name in names), ","))
        end
    end
    return path
end

function meanfield(rows, name)
    return mean(Float64(getproperty(row, name)) for row in rows)
end

function summarize(rows)
    return (;
        nsamples = length(rows),
        plus_improved_fraction = mean(row.plus_loss < row.free_loss for row in rows),
        minus_worsened_fraction = mean(row.minus_loss > row.free_loss for row in rows),
        plusminus_aligned_fraction = mean(row.plusminus_target_dot > 0 for row in rows),
        mean_free_loss = meanfield(rows, :free_loss),
        mean_plus_loss = meanfield(rows, :plus_loss),
        mean_minus_loss = meanfield(rows, :minus_loss),
        mean_plus_loss_delta = meanfield(rows, :plus_minus_free_loss),
        mean_minus_loss_delta = meanfield(rows, :minus_minus_free_loss),
        mean_plus_target_dot = meanfield(rows, :plus_target_dot),
        mean_minus_target_dot = meanfield(rows, :minus_target_dot),
        mean_plusminus_target_dot = meanfield(rows, :plusminus_target_dot),
        mean_plus_target_cos = meanfield(rows, :plus_target_cos),
        mean_minus_target_cos = meanfield(rows, :minus_target_cos),
        mean_plusminus_target_cos = meanfield(rows, :plusminus_target_cos),
        mean_plus_output_rms = meanfield(rows, :plus_output_rms),
        mean_minus_output_rms = meanfield(rows, :minus_output_rms),
        mean_plusminus_output_rms = meanfield(rows, :plusminus_output_rms),
        mean_plus_hidden_rms = meanfield(rows, :plus_hidden_rms),
        mean_minus_hidden_rms = meanfield(rows, :minus_hidden_rms),
        mean_plusminus_hidden_rms = meanfield(rows, :plusminus_hidden_rms),
        mean_worker_buffer_norm = meanfield(rows, :worker_buffer_norm),
    )
end

function run_phase_response()
    graph, layer, trainer = build_trainer()
    xtrain, ytrain = load_mnist_arrays(layer; split = :train, limit = NSAMPLES)
    rows = NamedTuple[]

    println(
        "MNIST phase response settings workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " samples=", NSAMPLES,
        " relaxation=", RELAXATION,
        " stepsize=", STEPSIZE,
        " temp=", TEMP,
        " beta=", BETA,
    )
    flush(stdout)

    sample_idx = 1
    while sample_idx <= size(xtrain, 2)
        last_idx = min(sample_idx + length(trainer.workers) - 1, size(xtrain, 2))
        chunk = sample_idx:last_idx
        IsingLearning._reset_batch_buffers!(trainer)
        jobs = [(; x = view(xtrain, :, idx), y = view(ytrain, :, idx)) for idx in chunk]
        seconds = @elapsed run!(trainer.manager, jobs)
        for worker in trainer.workers
            push!(rows, worker_phase_row(layer, worker))
        end
        rows_so_far = length(rows)
        summary = summarize(rows)
        println(
            "chunk ", first(chunk), ":", last(chunk),
            " seconds=", round(seconds; digits = 3),
            " rows=", rows_so_far,
            " plus_improved=", round(summary.plus_improved_fraction; digits = 3),
            " pm_aligned=", round(summary.plusminus_aligned_fraction; digits = 3),
            " mean_plus_delta=", round(summary.mean_plus_loss_delta; digits = 4),
            " mean_minus_delta=", round(summary.mean_minus_loss_delta; digits = 4),
        )
        flush(stdout)
        sample_idx = last_idx + 1
    end

    rows = rows[1:NSAMPLES]
    summary = summarize(rows)
    csv_path = write_rows(joinpath(OUTDIR, "mnist_phase_response.csv"), rows)
    summary_path = joinpath(OUTDIR, "mnist_phase_response_summary.txt")

    open(summary_path, "w") do io
        for name in propertynames(summary)
            println(io, name, "=", getproperty(summary, name))
        end
    end

    println("summary=", summary)
    println("saved_csv=", csv_path)
    println("saved_summary=", summary_path)
    flush(stdout)

    close_trainer!(trainer)
end

run_phase_response()
