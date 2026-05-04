using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Optimisers
using Random
using SparseArrays
using LinearAlgebra
using Base.Threads

Random.seed!(1234)

function float_type_from_env(name::AbstractString, default::AbstractString)
    value = lowercase(get(ENV, name, default))
    value in ("float64", "64", "double") && return Float64
    value in ("float32", "32", "single") && return Float32
    error("$name must be Float64 or Float32, got `$value`")
end

const FT = float_type_from_env("ISING_XOR_FLOAT_TYPE", "Float64")
const EPOCHS = parse(Int, get(ENV, "ISING_XOR_EPOCHS", "100"))
const RELAXATION_STEPS = parse(Int, get(ENV, "ISING_XOR_RELAXATION_STEPS", "1000"))
const LEARNING_RATE = parse(FT, get(ENV, "ISING_XOR_LEARNING_RATE", "0.1"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_WEIGHT_SCALE", "0.03"))
const BETA = parse(FT, get(ENV, "ISING_XOR_BETA", "0.1"))
const START_TEMP = parse(FT, get(ENV, "ISING_XOR_START_TEMP", "0.03"))
const STOP_TEMP = parse(FT, get(ENV, "ISING_XOR_STOP_TEMP", "1e-4"))
const HIDDEN_SIZES = parse.(Int, split(get(ENV, "ISING_XOR_HIDDEN_SIZES", get(ENV, "ISING_XOR_HIDDEN_UNITS", "32")), ","))
const OUTPUT_UNITS = parse(Int, get(ENV, "ISING_XOR_OUTPUT_UNITS", "8"))
const DYNAMICS = lowercase(get(ENV, "ISING_XOR_DYNAMICS", "local"))
const ADJUSTED = parse(Bool, lowercase(get(ENV, "ISING_XOR_ADJUSTED", "true")))
const LOCAL_POTENTIAL = parse(FT, get(ENV, "ISING_XOR_LOCAL_POTENTIAL", ADJUSTED ? "0" : "1.0"))
const DOUBLE_WELL_STRENGTH = parse(FT, get(ENV, "ISING_XOR_DOUBLE_WELL_STRENGTH", ADJUSTED ? "0" : "0.01"))
const REQUESTED_THREADS = parse(Int, get(ENV, "ISING_XOR_WORKER_THREADS", "2"))
const WORKER_THREADS = max(1, min(REQUESTED_THREADS, max(1, nthreads() - 1)))
const EVALUATION_REPEATS = parse(Int, get(ENV, "ISING_XOR_EVAL_REPEATS", "5"))
OUTPUT_UNITS >= 2 || error("ISING_XOR_OUTPUT_UNITS must be at least 2")
all(>(0), HIDDEN_SIZES) || error("all hidden layer sizes must be positive")

function small_weight_generator(seed::Integer = 4321)
    rng = Random.MersenneTwister(seed)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, FT))
end

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

function xor_input(a::Bool, b::Bool, p_vertical, p_horizontal)
    va = a ? p_vertical : -p_vertical
    hb = b ? p_horizontal : -p_horizontal
    return FT(0.5) .* (va .+ hb)
end

function output_pattern(label::Bool)
    pattern = Vector{FT}(undef, OUTPUT_UNITS)
    split = OUTPUT_UNITS ÷ 2
    @inbounds for idx in eachindex(pattern)
        base = idx <= split ? one(FT) : -one(FT)
        pattern[idx] = label ? -base : base
    end
    return pattern
end

xor_target(a::Bool, b::Bool) = output_pattern(xor(a, b))

function xor_dataset()
    p_vertical = pattern_vertical()
    p_horizontal = pattern_horizontal()
    cases = ((false, false), (false, true), (true, false), (true, true))
    x = Matrix{FT}(undef, 16, length(cases))
    y = Matrix{FT}(undef, OUTPUT_UNITS, length(cases))

    for (col, (a, b)) in enumerate(cases)
        x[:, col] .= xor_input(a, b, p_vertical, p_horizontal)
        y[:, col] .= xor_target(a, b)
    end

    return x, y, cases
end

function assert_finite_namedtuple(name, nt)
    all(isfinite, nt.w) || error("$name has non-finite weights")
    all(isfinite, nt.b) || error("$name has non-finite biases")
    all(isfinite, nt.α) || error("$name has non-finite local potentials")
    return nothing
end

function assert_finite_graph(g)
    all(isfinite, state(g)) || error("graph state contains non-finite values")
    all(isfinite, SparseArrays.getnzval(adj(g))) || error("graph weights contain non-finite values")
    return nothing
end

function run_output!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    Random.seed!(seed)
    Random.seed!(worker.context.dynamics.rng, seed)
    worker.context._state.x .= x
    Processes.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    output = copy(worker.context._state.equilibrium_state[trainer.layer.output_layer])
    all(isfinite, output) || error("non-finite output detected")
    return output
end

function evaluate_xor!(trainer, x, y, cases)
    set_trainer_temperature!(trainer, STOP_TEMP)
    outputs = map(axes(x, 2)) do idx
        averaged = zeros(FT, OUTPUT_UNITS)
        for repeat_idx in 1:max(1, EVALUATION_REPEATS)
            averaged .+= run_output!(trainer, view(x, :, idx); seed = 10_000 + 1000 * repeat_idx + idx)
        end
        averaged ./= FT(max(1, EVALUATION_REPEATS))
        averaged
    end
    mse = sum(idx -> sum(abs2, outputs[idx] .- view(y, :, idx)) / length(outputs[idx]), eachindex(outputs)) / length(outputs)
    false_pattern = output_pattern(false)
    true_pattern = output_pattern(true)
    predictions = map(outputs) do out
        dot(out, true_pattern) > dot(out, false_pattern)
    end
    truth = [xor(a, b) for (a, b) in cases]
    accuracy = count(==(true), predictions .== truth) / length(truth)
    return (; outputs, mse, accuracy)
end

function annealed_temperature(epoch::Integer, epochs::Integer)
    progress = epochs <= 1 ? one(FT) : FT(epoch - 1) / FT(epochs - 1)
    return STOP_TEMP + (START_TEMP - STOP_TEMP) * (one(FT) - progress)^2
end

function set_trainer_temperature!(trainer, T::Real)
    InteractiveIsing.temp!(trainer.prototype_graph, FT(T))
    InteractiveIsing.temp!(trainer.validation_graph, FT(T))
    foreach(g -> InteractiveIsing.temp!(g, FT(T)), trainer.worker_graphs)
    return trainer
end

function set_local_potential!(graph, value::Real)
    diag(adj(graph)) .= FT(value)
    InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.Quadratic, :lp) .= FT(value)
    return graph
end

function quadratic_coefficient_for_double_well()
    # The graph diagonal contributes -0.5 * lp[i] * s[i]^2 through Bilinear.
    # This coefficient makes the combined local potential
    #   lp * DOUBLE_WELL_STRENGTH * (s^4 - 2s^2),
    # so the local minima sit at s = -1 and s = 1.
    return FT(0.5) - FT(2) * DOUBLE_WELL_STRENGTH
end

function xor_architecture()
    layer_sizes = (16, HIDDEN_SIZES..., OUTPUT_UNITS)
    layers = [Layer(layer_sizes[idx], StateSet(-one(FT), one(FT)), Continuous(), Coords(0, idx, 0)) for idx in eachindex(layer_sizes)]
    wg = small_weight_generator()
    layer_args = Any[layers[1]]
    for idx in 2:length(layers)
        push!(layer_args, deepcopy(wg), layers[idx])
    end
    bias = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))
    local_potential = g -> InteractiveIsing.filltype(Vector, LOCAL_POTENTIAL, statelen(g))
    clamping_target = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))
    clamping_beta = InteractiveIsing.UniformArray(zero(FT))
    hamiltonian =
        Quadratic(c = quadratic_coefficient_for_double_well(), localpotential = local_potential) +
        Quartic(c = DOUBLE_WELL_STRENGTH, localpotential = local_potential) +
        InteractiveIsing.Bilinear() +
        InteractiveIsing.MagField(b = bias) +
        Clamping(β = clamping_beta, y = clamping_target)
    graph = IsingGraph(layer_args..., hamiltonian; precision = FT, index_set = g -> ToggledIndexSet(g))
    diag(adj(graph)) .= LOCAL_POTENTIAL
    return graph
end

function freeze_local_potential!(trainer, reference_params)
    trainer.params.α .= reference_params.α
    IsingLearning._broadcast_params!(trainer)
    return trainer
end

function langevin_dynamics()
    if DYNAMICS == "global"
        return GlobalLangevin(stepsize = FT(1e-3), adjusted = ADJUSTED, group_steps = 1)
    elseif DYNAMICS == "local"
        return LocalLangevin(stepsize = ADJUSTED ? FT(5e-3) : FT(1e-2), adjusted = ADJUSTED, group_steps = 1)
    else
        error("ISING_XOR_DYNAMICS must be `global` or `local`, got `$DYNAMICS`")
    end
end

function validation_langevin_dynamics()
    if DYNAMICS == "local"
        return LocalLangevin(stepsize = ADJUSTED ? FT(5e-3) : FT(1e-2), adjusted = ADJUSTED, group_steps = 1, order = :deterministic)
    else
        return deepcopy(langevin_dynamics())
    end
end

graph = xor_architecture()
InteractiveIsing.temp!(graph, START_TEMP)
dynamics = langevin_dynamics()
layer = LayeredIsingGraphLayer(
    xor_architecture;
    input_idxs = layerrange(graph[1]),
    output_idxs = layerrange(graph[end]),
    β = BETA,
    fullsweeps = 1,
    relaxation_steps = RELAXATION_STEPS,
    dynamics_algorithm = dynamics,
    validation_algorithm = validation_langevin_dynamics(),
)

x, y, cases = xor_dataset()
trainer = init_mnist_trainer(
    layer;
    graph,
    numthreads = WORKER_THREADS,
    optimiser = Optimisers.Adam(LEARNING_RATE),
)
batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
initial_params = deepcopy(trainer.params)

try
    println(
        "Running XOR pattern smoke test: ",
        (epochs = EPOCHS, worker_threads = WORKER_THREADS, relaxation_steps = RELAXATION_STEPS,
         learning_rate = LEARNING_RATE, weight_scale = WEIGHT_SCALE, β = BETA,
         start_T = START_TEMP, stop_T = STOP_TEMP, float_type = FT,
         hidden_sizes = HIDDEN_SIZES, output_units = OUTPUT_UNITS,
         local_potential = LOCAL_POTENTIAL, double_well_strength = DOUBLE_WELL_STRENGTH,
         dynamics = DYNAMICS, adjusted = ADJUSTED),
    )

    before = evaluate_xor!(trainer, x, y, cases)
    best_mse = before.mse
    best_accuracy = before.accuracy
    best_epoch = 0
    best_params = deepcopy(trainer.params)
    println("Before training: mse=$(before.mse), accuracy=$(before.accuracy)")
    for (case, output) in zip(cases, before.outputs)
        println("  input=$case target=$(xor_target(case...)) output=$output")
    end

    for epoch in 1:EPOCHS
        set_trainer_temperature!(trainer, annealed_temperature(epoch, EPOCHS))
        IsingLearning._run_minibatch!(trainer, x, y, batch_gradient)
        freeze_local_potential!(trainer, initial_params)
        assert_finite_namedtuple("params", trainer.params)
        assert_finite_namedtuple("batch_gradient", batch_gradient)
        foreach(assert_finite_graph, trainer.worker_graphs)

        if epoch == 1 || epoch % max(1, EPOCHS ÷ 10) == 0 || epoch == EPOCHS
            metrics = evaluate_xor!(trainer, x, y, cases)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b) + sum(abs2, batch_gradient.α))
            param_delta = sqrt(
                sum(abs2, trainer.params.w .- initial_params.w) +
                sum(abs2, trainer.params.b .- initial_params.b) +
                sum(abs2, trainer.params.α .- initial_params.α)
            )
            println(
                "epoch=$epoch mse=$(metrics.mse) accuracy=$(metrics.accuracy) ",
                "grad_norm=$grad_norm param_delta=$param_delta",
            )
            if metrics.accuracy > best_accuracy || (metrics.accuracy == best_accuracy && metrics.mse < best_mse)
                best_mse = metrics.mse
                best_accuracy = metrics.accuracy
                best_epoch = epoch
                best_params = deepcopy(trainer.params)
            end
        end
    end

    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    after = evaluate_xor!(trainer, x, y, cases)
    println("After training: mse=$(after.mse), accuracy=$(after.accuracy), restored_best_epoch=$best_epoch, best_logged_mse=$best_mse, best_logged_accuracy=$best_accuracy")
    for (case, output) in zip(cases, after.outputs)
        println("  input=$case target=$(xor_target(case...)) output=$output")
    end
finally
    close_trainer!(trainer)
end
