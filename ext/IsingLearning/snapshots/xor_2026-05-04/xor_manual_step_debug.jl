using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using SparseArrays
using LinearAlgebra

const FT = Float64

parse_list(::Type{T}, key, default) where {T} = parse.(T, split(get(ENV, key, default), ","))
parse_hidden_list(s) = parse.(Int, split(s, "+"))

const HIDDEN_SIZES = Tuple(only(parse_hidden_list.(split(get(ENV, "ISING_XOR_MANUAL_HIDDEN", "32"), ","))))
const OUTPUT_UNITS = parse(Int, get(ENV, "ISING_XOR_MANUAL_OUTPUT", "8"))
const TARGET_MODE = Symbol(get(ENV, "ISING_XOR_MANUAL_TARGET_MODE", "constant_readout_hterm"))
const EPOCHS = parse(Int, get(ENV, "ISING_XOR_MANUAL_EPOCHS", "40"))
const RELAXATION_STEPS = parse(Int, get(ENV, "ISING_XOR_MANUAL_RELAXATION", "300"))
const EVAL_REPEATS = parse(Int, get(ENV, "ISING_XOR_MANUAL_EVAL_REPEATS", "3"))
const LEARNING_RATE = parse(FT, get(ENV, "ISING_XOR_MANUAL_LR", "0.02"))
const BETA = parse(FT, get(ENV, "ISING_XOR_MANUAL_BETA", "0.1"))
const START_TEMP = parse(FT, get(ENV, "ISING_XOR_MANUAL_START_TEMP", "0.03"))
const STOP_TEMP = parse(FT, get(ENV, "ISING_XOR_MANUAL_STOP_TEMP", "1e-4"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_MANUAL_WEIGHT_SCALE", "0.03"))
const BIAS_SCALE = parse(FT, get(ENV, "ISING_XOR_MANUAL_BIAS_SCALE", "0.1"))
const WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_MANUAL_WEIGHT_SEED", "2"))
const BIAS_SEED = parse(Int, get(ENV, "ISING_XOR_MANUAL_BIAS_SEED", "11"))
const TRAIN_SEED = parse(Int, get(ENV, "ISING_XOR_MANUAL_TRAIN_SEED", "10001"))
const DYNAMICS = Symbol(get(ENV, "ISING_XOR_MANUAL_DYNAMICS", "block"))
const STATE_MODE = Symbol(get(ENV, "ISING_XOR_MANUAL_STATE_MODE", "continuous"))
const INIT_STATE = Symbol(get(ENV, "ISING_XOR_MANUAL_INIT_STATE", "random"))
const BLOCK_SIZE = parse(Int, get(ENV, "ISING_XOR_MANUAL_BLOCK_SIZE", "8"))
const GRADIENT_SIGN = parse(FT, get(ENV, "ISING_XOR_MANUAL_GRADIENT_SIGN", "1"))
const LOG_EVERY = parse(Int, get(ENV, "ISING_XOR_MANUAL_LOG_EVERY", "5"))
const WEIGHT_NORM = parse(FT, get(ENV, "ISING_XOR_MANUAL_WEIGHT_NORM", "0"))
const MINIT = parse(Int, get(ENV, "ISING_XOR_MANUAL_MINIT", "1"))

function pattern_vertical()
    p = Matrix{FT}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = col <= 2 ? -one(FT) : one(FT)
    end
    return vec(p)
end

function pattern_horizontal()
    p = Matrix{FT}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = row <= 2 ? -one(FT) : one(FT)
    end
    return vec(p)
end

function xor_input(a::Bool, b::Bool)
    pv = pattern_vertical()
    ph = pattern_horizontal()
    va = a ? pv : -pv
    hb = b ? ph : -ph
    return FT(0.5) .* (va .+ hb)
end

label_value(label::Bool) = label ? one(FT) : -one(FT)

function output_pattern(label::Bool, output_units::Integer)
    split = output_units ÷ 2
    return [label ? (idx <= split ? -one(FT) : one(FT)) :
                    (idx <= split ? one(FT) : -one(FT))
            for idx in 1:output_units]
end

readout_vector(output_units::Integer) =
    output_pattern(true, output_units) .- output_pattern(false, output_units)

readout_score(output, output_units::Integer) = dot(readout_vector(output_units), output)
target_score(label::Bool) = label_value(label)

function target_vector(label::Bool)
    if TARGET_MODE === :pattern
        return output_pattern(label, OUTPUT_UNITS)
    elseif TARGET_MODE === :readout_hterm || TARGET_MODE === :constant_readout_hterm
        target = zeros(FT, OUTPUT_UNITS)
        target[1] = target_score(label)
        return target
    else
        error("unknown target mode $TARGET_MODE")
    end
end

function xor_dataset()
    cases = ((false, false), (false, true), (true, false), (true, true))
    x = Matrix{FT}(undef, 16, length(cases))
    y = Matrix{FT}(undef, OUTPUT_UNITS, length(cases))
    labels = Bool[]
    for (col, (a, b)) in enumerate(cases)
        label = xor(a, b)
        x[:, col] .= xor_input(a, b)
        y[:, col] .= target_vector(label)
        push!(labels, label)
    end
    return x, y, labels, cases
end

function small_weight_generator()
    rng = Random.MersenneTwister(WEIGHT_SEED)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, FT))
end

function bias_generator()
    rng = Random.MersenneTwister(BIAS_SEED)
    return g -> BIAS_SCALE .* randn(rng, FT, statelen(g))
end

function xor_graph()
    layer_sizes = (16, HIDDEN_SIZES..., OUTPUT_UNITS)
    state_type = STATE_MODE === :discrete ? Discrete() : Continuous()
    layers = [Layer(layer_sizes[idx], StateSet(-one(FT), one(FT)), state_type, Coords(0, idx, 0))
              for idx in eachindex(layer_sizes)]
    wg = small_weight_generator()
    args = Any[layers[1]]
    for idx in 2:length(layers)
        push!(args, deepcopy(wg), layers[idx])
    end

    output_idxs = (sum(layer_sizes) - OUTPUT_UNITS + 1):sum(layer_sizes)
    clamping =
        TARGET_MODE === :constant_readout_hterm ?
            ConstantLinearReadoutNudge(output_idxs, readout_vector(OUTPUT_UNITS);
                β = zero(FT), target = zero(FT), free_score = zero(FT)) :
        TARGET_MODE === :readout_hterm ?
            LinearReadoutClamping(output_idxs, readout_vector(OUTPUT_UNITS);
                β = zero(FT), target = zero(FT)) :
            Clamping(β = InteractiveIsing.UniformArray(zero(FT)),
                y = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g)))

    h = Quadratic(c = FT(0.5), localpotential = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))) +
        Quartic(c = zero(FT), localpotential = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))) +
        InteractiveIsing.Bilinear() +
        InteractiveIsing.MagField(b = bias_generator()) +
        clamping

    graph = IsingGraph(args..., h; precision = FT, index_set = g -> ToggledIndexSet(g))
    diag(adj(graph)) .= zero(FT)
    return graph
end

function dynamics_algorithm()
    if DYNAMICS === :block
        return BlockLangevin(stepsize = FT(1e-3), adjusted = true, block_size = BLOCK_SIZE, group_steps = 1)
    elseif DYNAMICS === :local
        return LocalLangevin(stepsize = FT(5e-3), adjusted = true, group_steps = 1)
    elseif DYNAMICS === :metropolis
        return Metropolis()
    else
        error("unknown dynamics $DYNAMICS")
    end
end

function resolved_dynamics_stepper(worker)
    forward_routine = Processes.getalgo(worker.taskdata.func, 1)
    for child in Processes.getalgos(forward_routine)
        child isa Processes.AbstractIdentifiableAlgo && Base.getkey(child) === :dynamics && return child
    end
    error("could not find resolved @dynamics algorithm in worker process")
end

step_context!(dynamics_stepper, context) =
    Processes.step!(dynamics_stepper, context, Processes.Unstable())

function relax!(dynamics_stepper, context, n_steps::Integer)
    for _ in 1:n_steps
        context = step_context!(dynamics_stepper, context)
    end
    return context
end

function apply_input!(graph, x)
    InteractiveIsing.off!(graph.index_set, 1)
    state(graph[1]) .= x
    return graph
end

function set_graph_state!(graph, target)
    state(graph) .= target
    return graph
end

function manual_trainer()
    graph = xor_graph()
    algorithm = dynamics_algorithm()
    layer = LayeredIsingGraphLayer(
        graph;
        input_idxs = layerrange(graph[1]),
        output_idxs = layerrange(graph[end]),
        β = BETA,
        relaxation_steps = RELAXATION_STEPS,
        dynamics_algorithm = algorithm,
        validation_algorithm = deepcopy(algorithm),
    )
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = 1,
        optimiser = Optimisers.Adam(LEARNING_RATE),
    )
    return trainer
end

function set_trainer_temperature!(trainer, T)
    temp!(trainer.prototype_graph, FT(T))
    temp!(trainer.validation_graph, FT(T))
    foreach(g -> temp!(g, FT(T)), trainer.worker_graphs)
    return trainer
end

function set_target!(graph, y)
    IsingLearning.apply_targets(graph, y)
    return graph
end

set_beta!(graph, β) = IsingLearning.set_clamping_beta!(graph, β)

function reset_graph_state!(graph)
    if INIT_STATE === :ones
        state(graph) .= one(FT)
    elseif INIT_STATE === :zeros
        state(graph) .= zero(FT)
    elseif INIT_STATE === :random
        resetstate!(graph)
    else
        error("unknown INIT_STATE=$INIT_STATE")
    end
    return graph
end

function run_free_phase!(dynamics_stepper, context, x)
    graph = context.dynamics.model
    context._state.x .= x
    reset_graph_state!(graph)
    IsingLearning.apply_input(graph, context._state.x)
    context = relax!(dynamics_stepper, context, RELAXATION_STEPS)
    context._state.equilibrium_state .= state(graph)
    return context, copy(state(graph))
end

function run_nudged_phase!(dynamics_stepper, context, equilibrium_state, x, y, β)
    graph = context.dynamics.model
    context._state.x .= x
    context._state.y .= y
    set_graph_state!(graph, equilibrium_state)
    IsingLearning.apply_input(graph, context._state.x)
    set_target!(graph, context._state.y)
    set_beta!(graph, β)
    context = relax!(dynamics_stepper, context, RELAXATION_STEPS)
    return context, copy(state(graph))
end

function finite_buffer!(name, buffer)
    all(isfinite, buffer.w) || error("$name has non-finite weights")
    all(isfinite, buffer.b) || error("$name has non-finite biases")
    all(isfinite, buffer.α) || error("$name has non-finite local potentials")
    return buffer
end

function add_gradient!(dest, src)
    dest.w .+= src.w
    dest.b .+= src.b
    dest.α .+= src.α
    return dest
end

function normalize_weights!(params)
    WEIGHT_NORM > zero(FT) || return params
    isempty(params.w) && return params
    rms = sqrt(sum(abs2, params.w) / FT(length(params.w)))
    isfinite(rms) && rms > zero(FT) || return params
    params.w .*= WEIGHT_NORM / rms
    return params
end

function manual_minibatch!(dynamics_stepper, context, xbatch, ybatch, batch_gradient)
    MINIT > 0 || throw(ArgumentError("ISING_XOR_MANUAL_MINIT must be positive"))
    IsingLearning.zero_buffer!(batch_gradient)
    graph = context.dynamics.model

    for sample_idx in axes(xbatch, 2)
        for _ in 1:MINIT
            sample_gradient = IsingLearning.gradient_buffer(graph)
            context, free_state = run_free_phase!(dynamics_stepper, context, view(xbatch, :, sample_idx))
            context, plus_state = run_nudged_phase!(
                dynamics_stepper, context, free_state, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx), BETA)
            context, minus_state = run_nudged_phase!(
                dynamics_stepper, context, free_state, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx), -BETA)
            set_beta!(graph, zero(FT))

            IsingLearning.contrastive_gradient(graph, plus_state, minus_state, BETA; buffers = sample_gradient)
            add_gradient!(batch_gradient, sample_gradient)
        end
    end

    scale = inv(FT(2) * BETA * FT(size(xbatch, 2)) * FT(MINIT))
    IsingLearning.scale_buffer!(batch_gradient, scale * GRADIENT_SIGN)
    finite_buffer!("batch_gradient", batch_gradient)
    return context
end

function run_output!(dynamics_stepper, context, x; seed)
    Random.seed!(seed)
    Random.seed!(context.dynamics.rng, seed)
    graph = context.dynamics.model
    context._state.x .= x
    reset_graph_state!(graph)
    IsingLearning.apply_input(graph, context._state.x)
    context = relax!(dynamics_stepper, context, RELAXATION_STEPS)
    return context, copy(state(graph[end]))
end

function evaluate!(dynamics_stepper, context, x, labels)
    temp!(context.dynamics.model, STOP_TEMP)
    outputs = Vector{Vector{FT}}(undef, size(x, 2))
    for sample_idx in axes(x, 2)
        out = zeros(FT, OUTPUT_UNITS)
        for repeat_idx in 1:EVAL_REPEATS
            context, sample_out = run_output!(dynamics_stepper, context, view(x, :, sample_idx);
                seed = 20_000 + 1000 * repeat_idx + sample_idx)
            out .+= sample_out
        end
        outputs[sample_idx] = out ./ FT(EVAL_REPEATS)
    end
    scores = [readout_score(out, OUTPUT_UNITS) for out in outputs]
    predictions = scores .> zero(FT)
    accuracy = count(predictions .== labels) / length(labels)
    mse = sum(abs2(scores[idx] - target_score(labels[idx])) for idx in eachindex(scores)) / length(scores)
    margin = minimum(abs, scores)
    return context, (; outputs, scores, accuracy, mse, margin)
end

function train_manual!()
    Random.seed!(TRAIN_SEED)
    trainer = manual_trainer()
    worker = only(trainer.workers)
    context = Processes.context(worker)
    graph = context.dynamics.model
    dynamics_stepper = resolved_dynamics_stepper(worker)
    set_trainer_temperature!(trainer, START_TEMP)
    x, y, labels, cases = xor_dataset()

    params = trainer.params
    initial_params = deepcopy(params)
    best_params = deepcopy(params)
    opt_state = trainer.opt_state
    batch_gradient = Processes.context(worker)._state.buffers

    context, before = evaluate!(dynamics_stepper, context, x, labels)
    best = (; epoch = 0, metrics = before)

    for epoch in 1:EPOCHS
        progress = EPOCHS <= 1 ? one(FT) : FT(epoch - 1) / FT(EPOCHS - 1)
        set_trainer_temperature!(trainer, STOP_TEMP + (START_TEMP - STOP_TEMP) * (one(FT) - progress)^2)
        context = manual_minibatch!(dynamics_stepper, context, x, y, batch_gradient)
        opt_state, params = Optimisers.update(opt_state, params, batch_gradient)
        params.α .= initial_params.α
        normalize_weights!(params)
        IsingLearning.sync_graph_params!(graph, params)

        if epoch == 1 || epoch % LOG_EVERY == 0 || epoch == EPOCHS
            context, metrics = evaluate!(dynamics_stepper, context, x, labels)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b) + sum(abs2, batch_gradient.α))
            param_delta = sqrt(sum(abs2, params.w .- initial_params.w) + sum(abs2, params.b .- initial_params.b))
            weight_rms = sqrt(sum(abs2, params.w) / FT(length(params.w)))
            println(
                "epoch=$epoch mse=$(round(metrics.mse, digits=6)) acc=$(metrics.accuracy) ",
                "margin=$(round(metrics.margin, digits=6)) grad=$(round(grad_norm, digits=6)) ",
                "delta=$(round(param_delta, digits=6)) weight_rms=$(round(weight_rms, digits=6))",
            )
            if metrics.accuracy > best.metrics.accuracy ||
                    (metrics.accuracy == best.metrics.accuracy && metrics.mse < best.metrics.mse)
                best = (; epoch, metrics)
                best_params = deepcopy(params)
            end
        end
    end

    IsingLearning.sync_graph_params!(graph, best_params)
    context, after = evaluate!(dynamics_stepper, context, x, labels)

    println("before: mse=$(round(before.mse, digits=6)) acc=$(before.accuracy)")
    println("best: epoch=$(best.epoch) mse=$(round(best.metrics.mse, digits=6)) acc=$(best.metrics.accuracy)")
    println("after: mse=$(round(after.mse, digits=6)) acc=$(after.accuracy) margin=$(round(after.margin, digits=6))")
    for (case, score, out) in zip(cases, after.scores, after.outputs)
        println(
            "case=$case truth=$(xor(case...)) score=$(round(score, digits=6)) ",
            "target=$(target_score(xor(case...))) out=", round.(out; digits=4),
        )
    end
    close_trainer!(trainer)
end

    println(
        "Manual XOR debug: hidden=$HIDDEN_SIZES out=$OUTPUT_UNITS target=$TARGET_MODE ",
        "dynamics=$DYNAMICS state=$STATE_MODE epochs=$EPOCHS relax=$RELAXATION_STEPS ",
    "β=$BETA lr=$LEARNING_RATE grad_sign=$GRADIENT_SIGN init=$INIT_STATE weight_norm=$WEIGHT_NORM minit=$MINIT",
)
train_manual!()
