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
const SWEEP_RELAXATION_STEPS = parse.(Int, split(get(ENV, "ISING_XOR_RELAXATION_SWEEP", "34,100,340,1000,2000"), ","))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_WEIGHT_SCALE", "0.03"))
const BETA = parse(FT, get(ENV, "ISING_XOR_BETA", "0.1"))
const TEMPERATURE = parse(FT, get(ENV, "ISING_XOR_RELAXATION_T", "1e-4"))
const HIDDEN_SIZES = parse.(Int, split(get(ENV, "ISING_XOR_HIDDEN_SIZES", get(ENV, "ISING_XOR_HIDDEN_UNITS", "32")), ","))
const OUTPUT_UNITS = parse(Int, get(ENV, "ISING_XOR_OUTPUT_UNITS", "8"))
const LOCAL_POTENTIAL = parse(FT, get(ENV, "ISING_XOR_LOCAL_POTENTIAL", "1.0"))
const DOUBLE_WELL_STRENGTH = parse(FT, get(ENV, "ISING_XOR_DOUBLE_WELL_STRENGTH", "0.1"))
const DYNAMICS = lowercase(get(ENV, "ISING_XOR_DYNAMICS", "local"))
const REQUESTED_THREADS = parse(Int, get(ENV, "ISING_XOR_WORKER_THREADS", "1"))
const WORKER_THREADS = max(1, min(REQUESTED_THREADS, max(1, nthreads() - 1)))
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

function xor_architecture()
    layer_sizes = (16, HIDDEN_SIZES..., OUTPUT_UNITS)
    layers = [Layer(layer_sizes[idx], Continuous(), Coords(0, idx, 0)) for idx in eachindex(layer_sizes)]
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
        Quadratic(c = -FT(2) * DOUBLE_WELL_STRENGTH, localpotential = local_potential) +
        Quartic(c = DOUBLE_WELL_STRENGTH, localpotential = local_potential) +
        InteractiveIsing.Bilinear() +
        InteractiveIsing.MagField(b = bias) +
        Clamping(β = clamping_beta, y = clamping_target)
    graph = IsingGraph(layer_args..., hamiltonian; precision = FT, index_set = g -> ToggledIndexSet(g))
    diag(adj(graph)) .= LOCAL_POTENTIAL
    return graph
end

function langevin_dynamics()
    if DYNAMICS == "global"
        return GlobalLangevin(stepsize = FT(1e-3), adjusted = false, group_steps = 1)
    elseif DYNAMICS == "local"
        return LocalLangevin(stepsize = FT(1e-2), adjusted = false, group_steps = 1)
    else
        error("ISING_XOR_DYNAMICS must be `global` or `local`, got `$DYNAMICS`")
    end
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
    outputs = [run_output!(trainer, view(x, :, idx); seed = 10_000 + idx) for idx in axes(x, 2)]
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

function print_float_summary(trainer)
    graph = trainer.prototype_graph
    println(
        "float summary: ",
        (state = eltype(state(graph)),
         weights = eltype(SparseArrays.getnzval(adj(graph))),
         bias = eltype(trainer.params.b),
         quadratic = eltype(trainer.params.α)),
    )
    return nothing
end

x, y, cases = xor_dataset()
dynamics = langevin_dynamics()
active_units = sum(HIDDEN_SIZES) + OUTPUT_UNITS

println(
    "Running XOR relaxation sweep: ",
    (steps = SWEEP_RELAXATION_STEPS, dynamics = DYNAMICS, active_units, temperature = TEMPERATURE,
     β = BETA, float_type = FT, hidden_sizes = HIDDEN_SIZES, output_units = OUTPUT_UNITS,
     double_well_strength = DOUBLE_WELL_STRENGTH),
)

reference_outputs = nothing
previous_outputs = nothing

for relaxation_steps in SWEEP_RELAXATION_STEPS
    graph = xor_architecture()
    InteractiveIsing.temp!(graph, TEMPERATURE)
    layer = LayeredIsingGraphLayer(
        xor_architecture;
        input_idxs = layerrange(graph[1]),
        output_idxs = layerrange(graph[end]),
        β = BETA,
        fullsweeps = 1,
        relaxation_steps,
        dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = WORKER_THREADS,
        optimiser = Optimisers.Adam(zero(FT)),
    )

    try
        relaxation_steps == first(SWEEP_RELAXATION_STEPS) && print_float_summary(trainer)
        metrics = evaluate_xor!(trainer, x, y, cases)
        output_matrix = reduce(hcat, metrics.outputs)
        global reference_outputs
        global previous_outputs
        if reference_outputs === nothing
            reference_outputs = output_matrix
        end
        output_shift = maximum(abs.(output_matrix .- reference_outputs))
        previous_shift = previous_outputs === nothing ? zero(output_shift) : maximum(abs.(output_matrix .- previous_outputs))
        previous_outputs = output_matrix
        updates_per_spin = DYNAMICS == "local" ? FT(relaxation_steps) / FT(active_units) : FT(relaxation_steps)
        println(
            "relaxation_steps=$relaxation_steps updates_per_active_spin=$updates_per_spin ",
            "mse=$(metrics.mse) accuracy=$(metrics.accuracy) ",
            "max_output_shift_from_first=$output_shift max_output_shift_from_previous=$previous_shift",
        )
        for (case, output) in zip(cases, metrics.outputs)
            println("  input=$case target=$(xor_target(case...)) output=$output")
        end
    finally
        close_trainer!(trainer)
    end
end
