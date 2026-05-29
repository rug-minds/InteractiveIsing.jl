using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using LinearAlgebra: dot
using Random
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing

const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_RESPONSE_HIDDEN", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_RESPONSE_OUTPUT_REPLICAS", "4"))
const NSAMPLES = parse(Int, get(ENV, "ISING_MNIST_RESPONSE_SAMPLES", "32"))
const CHECKPOINT_SWEEPS = parse.(Float64, split(get(ENV, "ISING_MNIST_RESPONSE_SWEEPS", "1,2,5,10,20,50,100"), ","))
const STEPSIZES = parse.(Float32, split(get(ENV, "ISING_MNIST_RESPONSE_STEPSIZES", "0.05,0.1,0.2,0.5"), ","))
const BETAS = parse.(Float32, split(get(ENV, "ISING_MNIST_RESPONSE_BETAS", "0.1,0.2"), ","))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_TEMP", "0.001"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_WEIGHT_SCALE", "0.005"))
const BIAS_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_BIAS_SCALE", "0.02"))
const TARGET_SCALE = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_TARGET_SCALE", "0.2"))
const TARGET_OFF = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_TARGET_OFF", string(-TARGET_SCALE)))
const TARGET_ON = parse(Float32, get(ENV, "ISING_MNIST_RESPONSE_TARGET_ON", string(TARGET_SCALE)))
const OUTDIR = get(ENV, "ISING_MNIST_RESPONSE_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_response")))

mkpath(OUTDIR)

"""
    append_row!(path, row)

Append one named-tuple row to a CSV file, writing the header on first use.
"""
function append_row!(path::AbstractString, row)
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    rms_delta(a, b)

Return root mean square distance between two state vectors or views.
"""
function rms_delta(a, b)
    return sqrt(mean(abs2, a .- b))
end

"""
    cosine(a, b)

Return cosine similarity, with zero returned for degenerate vectors.
"""
function cosine(a, b)
    denom = sqrt(sum(abs2, a)) * sqrt(sum(abs2, b))
    denom == 0 && return 0.0
    return dot(a, b) / denom
end

"""
    class_margin(scores)

Return the best-minus-second-best class score margin.
"""
function class_margin(scores)
    length(scores) >= 2 || return 0.0
    best = -Inf
    second = -Inf
    for score in scores
        score > best && ((second = best); (best = score); continue)
        score > second && (second = score)
    end
    return best - second
end

"""
    scale_targets!(y)

Map the `-1/+1` MNIST target encoding into the smaller clamp target range used
by the recent MNIST training probes.
"""
function scale_targets!(y)
    y .= ifelse.(y .> 0, TARGET_ON, TARGET_OFF)
    return y
end

"""
    init_biases!(graph, seed)

Initialize graph biases with the same small random bias scale used by the
training probes, so response diagnostics match the trained setup.
"""
function init_biases!(graph, seed::Integer)
    rng = Random.MersenneTwister(seed)
    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    b .= BIAS_SCALE .* randn(rng, eltype(graph), length(b))
    return graph
end

"""
    active_units(graph)

Return the number of non-input MNIST spins. We report relaxation in sweeps over
these active spins because the input image is clamped off in the active index set.
"""
function active_units(graph)
    return length(II.layerrange(graph[2])) + length(II.layerrange(graph[end]))
end

"""
    checkpoint_steps(graph)

Convert requested sweep counts into monotonically increasing single-spin update
counts for the current graph.
"""
function checkpoint_steps(graph)
    steps = sort!(unique!(max.(1, round.(Int, CHECKPOINT_SWEEPS .* active_units(graph)))))
    return steps
end

"""
    build_response_layer(stepsize, beta, config_id)

Build the `784 -> 120 -> 40` MNIST graph and layer used for response
measurements. The dynamics are LocalLangevin because that is the target sampler.
"""
function build_response_layer(stepsize::Float32, beta::Float32, config_id::Integer)
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(10_000 + config_id),
    )
    init_biases!(graph, 20_000 + config_id)
    temp!(graph, TEMP)
    dynamics = LocalLangevin(stepsize = stepsize, adjusted = false)
    max_steps = maximum(checkpoint_steps(graph))
    layer = MNISTLayer(
        graph = graph,
        β = beta,
        free_relaxation_steps = max_steps,
        nudged_relaxation_steps = max_steps,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return graph, layer, dynamics
end

"""
    run_to_checkpoints!(algorithm, context, steps)

Run a dynamics context up to each requested step count and return copied graph
states at those checkpoints.
"""
function run_to_checkpoints!(algorithm::A, context::C, steps::Vector{Int}) where {A,C}
    snapshots = Vector{Vector{eltype(context.model)}}(undef, length(steps))
    current_step = 0
    for (checkpoint_idx, target_step) in enumerate(steps)
        while current_step < target_step
            Processes.step!(algorithm, context)
            current_step += 1
        end
        snapshots[checkpoint_idx] = copy(state(context.model))
    end
    return snapshots
end

"""
    free_phase_snapshots!(graph, algorithm, x, steps)

Reset the graph, clamp one MNIST image, and return free-phase state snapshots.
"""
function free_phase_snapshots!(graph, algorithm, x, steps)
    resetstate!(graph)
    IsingLearning.apply_input(graph, x)
    IsingLearning.set_clamping_beta!(graph, zero(eltype(graph)))
    context = Processes.init(deepcopy(algorithm), (; model = graph))
    return run_to_checkpoints!(algorithm, context, steps)
end

"""
    nudged_phase_snapshots!(graph, algorithm, free_state, x, y, beta, steps)

Start from a free equilibrium state, apply one signed nudge, and return nudged
state snapshots.
"""
function nudged_phase_snapshots!(graph, algorithm, free_state, x, y, beta, steps)
    state(graph) .= free_state
    IsingLearning.apply_input(graph, x)
    IsingLearning.apply_targets(graph, y)
    IsingLearning.set_clamping_beta!(graph, beta)
    context = Processes.init(deepcopy(algorithm), (; model = graph))
    snapshots = run_to_checkpoints!(algorithm, context, steps)
    IsingLearning.set_clamping_beta!(graph, zero(beta))
    return snapshots
end

"""
    free_row(...)

Build one CSV row describing how close a free-phase checkpoint is to the longest
run in the same sample trajectory.
"""
function free_row(config_id, sample_idx, stepsize, beta, graph, steps, checkpoint_idx, snapshot, final_state, y)
    hidden_idxs = II.layerrange(graph[2])
    output_idxs = II.layerrange(graph[end])
    output = @view snapshot[output_idxs]
    final_output = @view final_state[output_idxs]
    scores = IsingLearning._mnist_class_scores(output)
    target_scores = IsingLearning._mnist_class_scores(y)
    return (;
        config_id,
        sample_idx,
        stepsize,
        beta,
        phase = "free",
        steps = steps[checkpoint_idx],
        sweeps = steps[checkpoint_idx] / active_units(graph),
        state_rms_to_final = rms_delta(snapshot, final_state),
        hidden_rms_to_final = rms_delta(@view(snapshot[hidden_idxs]), @view(final_state[hidden_idxs])),
        output_rms_to_final = rms_delta(output, final_output),
        output_mse_to_target = sum(abs2, output .- y),
        target_digit = argmax(target_scores) - 1,
        prediction = argmax(scores) - 1,
        score_margin = class_margin(scores),
    )
end

"""
    nudged_row(...)

Build one CSV row describing signed nudged response relative to the free state.
"""
function nudged_row(config_id, sample_idx, stepsize, beta, graph, steps, checkpoint_idx, free_state, plus_state, minus_state, y)
    hidden_idxs = II.layerrange(graph[2])
    output_idxs = II.layerrange(graph[end])
    free_output = @view free_state[output_idxs]
    plus_output = @view plus_state[output_idxs]
    minus_output = @view minus_state[output_idxs]
    target_dir = y .- free_output
    plus_dir = plus_output .- free_output
    minus_dir = minus_output .- free_output
    plusminus_dir = plus_output .- minus_output
    free_loss = sum(abs2, free_output .- y)
    plus_loss = sum(abs2, plus_output .- y)
    minus_loss = sum(abs2, minus_output .- y)
    return (;
        config_id,
        sample_idx,
        stepsize,
        beta,
        phase = "nudged",
        steps = steps[checkpoint_idx],
        sweeps = steps[checkpoint_idx] / active_units(graph),
        free_loss,
        plus_loss,
        minus_loss,
        plus_loss_delta = plus_loss - free_loss,
        minus_loss_delta = minus_loss - free_loss,
        plus_target_dot = dot(plus_dir, target_dir),
        minus_target_dot = dot(minus_dir, target_dir),
        plusminus_target_dot = dot(plusminus_dir, target_dir),
        plus_target_cos = cosine(plus_dir, target_dir),
        minus_target_cos = cosine(minus_dir, target_dir),
        plusminus_target_cos = cosine(plusminus_dir, target_dir),
        plus_output_rms = rms_delta(plus_output, free_output),
        minus_output_rms = rms_delta(minus_output, free_output),
        plusminus_output_rms = rms_delta(plus_output, minus_output),
        plus_hidden_rms = rms_delta(@view(plus_state[hidden_idxs]), @view(free_state[hidden_idxs])),
        minus_hidden_rms = rms_delta(@view(minus_state[hidden_idxs]), @view(free_state[hidden_idxs])),
        plusminus_hidden_rms = rms_delta(@view(plus_state[hidden_idxs]), @view(minus_state[hidden_idxs])),
    )
end

function mean_field(rows, name)
    return mean(Float64(getproperty(row, name)) for row in rows)
end

function fraction(rows, f)
    return mean(f(row) for row in rows)
end

"""
    write_summaries!(path, free_rows, nudged_rows)

Aggregate sample rows by config and checkpoint so the response curves are easy
to scan without plotting first.
"""
function write_summaries!(path, free_rows, nudged_rows)
    open(path, "w") do io
        println(io, "kind,config_id,stepsize,beta,sweeps,nsamples,metric,value")

        for config_id in sort(unique(row.config_id for row in free_rows))
            config_free = filter(row -> row.config_id == config_id, free_rows)
            config_nudged = filter(row -> row.config_id == config_id, nudged_rows)
            for sweeps in sort(unique(row.sweeps for row in config_free))
                chunk = filter(row -> row.sweeps == sweeps, config_free)
                first_row = first(chunk)
                for metric in (:state_rms_to_final, :hidden_rms_to_final, :output_rms_to_final, :output_mse_to_target, :score_margin)
                    println(io, join(("free", config_id, first_row.stepsize, first_row.beta, sweeps, length(chunk), metric, mean_field(chunk, metric)), ","))
                end
            end
            for sweeps in sort(unique(row.sweeps for row in config_nudged))
                chunk = filter(row -> row.sweeps == sweeps, config_nudged)
                first_row = first(chunk)
                metrics = (
                    :plus_loss_delta,
                    :minus_loss_delta,
                    :plusminus_target_dot,
                    :plus_target_cos,
                    :minus_target_cos,
                    :plus_output_rms,
                    :plus_hidden_rms,
                    :plusminus_output_rms,
                )
                for metric in metrics
                    println(io, join(("nudged", config_id, first_row.stepsize, first_row.beta, sweeps, length(chunk), metric, mean_field(chunk, metric)), ","))
                end
                println(io, join(("nudged", config_id, first_row.stepsize, first_row.beta, sweeps, length(chunk), "plus_improves_fraction", fraction(chunk, row -> row.plus_loss_delta < 0)), ","))
                println(io, join(("nudged", config_id, first_row.stepsize, first_row.beta, sweeps, length(chunk), "minus_worsens_fraction", fraction(chunk, row -> row.minus_loss_delta > 0)), ","))
            end
        end
    end
    return path
end

"""
    run_config!(config_id, stepsize, beta)

Run free and nudged response diagnostics for one LocalLangevin setting.
"""
function run_config!(config_id::Integer, stepsize::Float32, beta::Float32)
    graph, layer, dynamics = build_response_layer(stepsize, beta, config_id)
    steps = checkpoint_steps(graph)
    x, y = load_mnist_arrays(layer; split = :train, limit = NSAMPLES)
    y = scale_targets!(copy(y))

    free_csv = joinpath(OUTDIR, "mnist_relaxation_free.csv")
    nudged_csv = joinpath(OUTDIR, "mnist_relaxation_nudged.csv")
    free_rows = NamedTuple[]
    nudged_rows = NamedTuple[]

    println(
        "response config=", config_id,
        " hidden=", HIDDEN,
        " outputs=", 10 * OUTPUT_REPLICAS,
        " samples=", NSAMPLES,
        " stepsize=", stepsize,
        " beta=", beta,
        " temp=", TEMP,
        " checkpoint_sweeps=", CHECKPOINT_SWEEPS,
    )
    flush(stdout)

    for sample_idx in axes(x, 2)
        xsample = view(x, :, sample_idx)
        ysample = view(y, :, sample_idx)
        free_snapshots = free_phase_snapshots!(graph, dynamics, xsample, steps)
        free_state = last(free_snapshots)
        plus_snapshots = nudged_phase_snapshots!(graph, dynamics, free_state, xsample, ysample, beta, steps)
        minus_snapshots = nudged_phase_snapshots!(graph, dynamics, free_state, xsample, ysample, -beta, steps)

        for checkpoint_idx in eachindex(steps)
            fr = free_row(config_id, sample_idx, stepsize, beta, graph, steps, checkpoint_idx, free_snapshots[checkpoint_idx], free_state, ysample)
            nr = nudged_row(config_id, sample_idx, stepsize, beta, graph, steps, checkpoint_idx, free_state, plus_snapshots[checkpoint_idx], minus_snapshots[checkpoint_idx], ysample)
            push!(free_rows, fr)
            push!(nudged_rows, nr)
            append_row!(free_csv, fr)
            append_row!(nudged_csv, nr)
        end
    end

    write_summaries!(joinpath(OUTDIR, "mnist_relaxation_summary_config_$(config_id).csv"), free_rows, nudged_rows)
    final_free = filter(row -> row.sweeps == maximum(CHECKPOINT_SWEEPS), free_rows)
    final_nudged = filter(row -> row.sweeps == maximum(CHECKPOINT_SWEEPS), nudged_rows)
    result = (;
        config_id,
        stepsize,
        beta,
        mean_final_free_output_mse = mean_field(final_free, :output_mse_to_target),
        mean_final_free_margin = mean_field(final_free, :score_margin),
        mean_final_plus_loss_delta = mean_field(final_nudged, :plus_loss_delta),
        mean_final_minus_loss_delta = mean_field(final_nudged, :minus_loss_delta),
        mean_final_plusminus_target_dot = mean_field(final_nudged, :plusminus_target_dot),
        final_plus_improves_fraction = fraction(final_nudged, row -> row.plus_loss_delta < 0),
        final_minus_worsens_fraction = fraction(final_nudged, row -> row.minus_loss_delta > 0),
    )
    append_row!(joinpath(OUTDIR, "mnist_relaxation_config_summary.csv"), result)
    println("result=", result)
    flush(stdout)
    return result
end

function main()
    config_id = 0
    results = NamedTuple[]
    for stepsize in STEPSIZES, beta in BETAS
        config_id += 1
        push!(results, run_config!(config_id, stepsize, beta))
    end
    println("Saved MNIST relaxation response diagnostics in ", OUTDIR)
    return results
end

main()
