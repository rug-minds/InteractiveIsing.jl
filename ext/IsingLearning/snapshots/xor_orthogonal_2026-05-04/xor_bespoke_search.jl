using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Optimisers
using Random
using SparseArrays
using LinearAlgebra

const FT = Float64

const DEFAULT_EPOCHS = parse(Int, get(ENV, "ISING_XOR_SEARCH_EPOCHS", "80"))
const LOG_EVERY = parse(Int, get(ENV, "ISING_XOR_SEARCH_LOG_EVERY", "20"))
const EVALUATION_REPEATS = parse(Int, get(ENV, "ISING_XOR_SEARCH_EVAL_REPEATS", "5"))
const MINIT = parse(Int, get(ENV, "ISING_XOR_SEARCH_MINIT", "1"))
const WORKER_THREADS = 1

parse_list(::Type{T}, key, default) where {T} = parse.(T, split(get(ENV, key, default), ","))
parse_hidden_list(s) = parse.(Int, split(s, "+"))

const HIDDEN_CONFIGS = parse_hidden_list.(split(get(ENV, "ISING_XOR_SEARCH_HIDDEN", "32"), ","))
const OUTPUT_CONFIGS = parse_list(Int, "ISING_XOR_SEARCH_OUTPUT", "2")
const OUTPUT_CODES = Symbol.(split(get(ENV, "ISING_XOR_SEARCH_OUTPUT_CODE", "orthogonal"), ","))
const TARGET_MODES = Symbol.(split(get(ENV, "ISING_XOR_SEARCH_TARGET_MODE", "readout_hterm"), ","))
const READOUT_TARGETS = parse_list(FT, "ISING_XOR_SEARCH_READOUT_TARGET", "1")
const ADJUSTED = parse(Bool, lowercase(get(ENV, "ISING_XOR_SEARCH_ADJUSTED", "true")))
const LOCAL_POTENTIALS = parse_list(FT, "ISING_XOR_SEARCH_LOCAL_POTENTIAL", ADJUSTED ? "0" : "1.0")
const DOUBLE_WELLS = parse_list(FT, "ISING_XOR_SEARCH_DOUBLE_WELL", ADJUSTED ? "0" : "0.01")
const LEARNING_RATES = parse_list(FT, "ISING_XOR_SEARCH_LR", "0.005,0.01,0.03")
const BETAS = parse_list(FT, "ISING_XOR_SEARCH_BETA", "0.05,0.1")
const RELAXATION_STEPS_LIST = parse_list(Int, "ISING_XOR_SEARCH_RELAXATION", "1000")
const DYNAMICS_CONFIGS = Symbol.(split(get(ENV, "ISING_XOR_SEARCH_DYNAMICS", "local"), ","))
const BLOCK_SIZES = parse_list(Int, "ISING_XOR_SEARCH_BLOCK_SIZE", "8")
const WEIGHT_SEEDS = parse_list(Int, "ISING_XOR_SEARCH_WEIGHT_SEED", "4321")
const BIAS_SEEDS = parse_list(Int, "ISING_XOR_SEARCH_BIAS_SEED", "8765")
const STATE_MODES = Symbol.(split(get(ENV, "ISING_XOR_SEARCH_STATE_MODE", "continuous"), ","))
const START_TEMP = parse(FT, get(ENV, "ISING_XOR_SEARCH_START_TEMP", "0.03"))
const STOP_TEMP = parse(FT, get(ENV, "ISING_XOR_SEARCH_STOP_TEMP", "1e-4"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_SEARCH_WEIGHT_SCALE", "0.03"))
const SKIP_WEIGHT_SCALES = parse_list(FT, "ISING_XOR_SEARCH_SKIP_WEIGHT_SCALE", "0")
const WEIGHT_NORMS = parse_list(FT, "ISING_XOR_SEARCH_WEIGHT_NORM", "0")
const BIAS_SCALES = parse_list(FT, "ISING_XOR_SEARCH_BIAS_SCALE", "0")
const BASE_SEED = parse(Int, get(ENV, "ISING_XOR_SEARCH_BASE_SEED", "10000"))
# Debug-only override. The mathematically correct EP descent gradient for the
# core code uses +1 here; -1 is useful only for probing finite-relaxation
# transients.
const GRADIENT_SIGN = parse(FT, get(ENV, "ISING_XOR_SEARCH_GRADIENT_SIGN", "1"))
const PRINT_OUTPUTS = parse(Bool, lowercase(get(ENV, "ISING_XOR_SEARCH_PRINT_OUTPUTS", "false")))
const INPUT_BIAS = parse(Bool, lowercase(get(ENV, "ISING_XOR_SEARCH_INPUT_BIAS", "false")))

input_units() = INPUT_BIAS ? 17 : 16

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

function output_pattern(label::Bool, output_units::Integer, output_code::Symbol = OUTPUT_CODES[1])
    if output_code === :antipodal
        pattern = Vector{FT}(undef, output_units)
        split = output_units ÷ 2
        @inbounds for idx in eachindex(pattern)
            base = idx <= split ? one(FT) : -one(FT)
            pattern[idx] = label ? -base : base
        end
        return pattern
    elseif output_code === :orthogonal
        iseven(output_units) ||
            throw(ArgumentError("orthogonal output code requires an even output size, got $output_units"))
        pattern = ones(FT, output_units)
        if label
            split = output_units ÷ 2
            @inbounds for idx in (split + 1):output_units
                pattern[idx] = -one(FT)
            end
        end
        return pattern
    else
        error("unknown output code $output_code")
    end
end

label_value(label::Bool) = label ? one(FT) : -one(FT)

function readout_vector(output_units::Integer, output_code::Symbol = OUTPUT_CODES[1])
    raw = output_pattern(true, output_units, output_code) .-
        output_pattern(false, output_units, output_code)
    return output_code === :orthogonal ? raw ./ FT(output_units) : raw
end

function readout_score(output, output_units::Integer, output_code::Symbol = OUTPUT_CODES[1])
    return dot(readout_vector(output_units, output_code), output)
end

function label_target_score(label::Bool, config)
    if config.target_mode === :pattern
        return readout_score(
            output_pattern(label, config.output_units, config.output_code),
            config.output_units,
            config.output_code,
        )
    elseif config.target_mode === :readout ||
            config.target_mode === :readout_hterm ||
            config.target_mode === :constant_readout_hterm
        return label_value(label) * config.readout_target
    else
        error("unknown target_mode $(config.target_mode)")
    end
end

function output_clamping_target(label::Bool, config)
    if config.target_mode === :pattern
        return output_pattern(label, config.output_units, config.output_code)
    elseif config.target_mode === :readout
        w = readout_vector(config.output_units, config.output_code)
        return label_target_score(label, config) .* w ./ dot(w, w)
    elseif config.target_mode === :readout_hterm || config.target_mode === :constant_readout_hterm
        target = zeros(FT, config.output_units)
        target[1] = label_target_score(label, config)
        return target
    else
        error("unknown target_mode $(config.target_mode)")
    end
end

xor_target(a::Bool, b::Bool, config) = output_clamping_target(xor(a, b), config)

function xor_dataset(config)
    p_vertical = pattern_vertical()
    p_horizontal = pattern_horizontal()
    cases = ((false, false), (false, true), (true, false), (true, true))
    x = Matrix{FT}(undef, input_units(), length(cases))
    y = Matrix{FT}(undef, config.output_units, length(cases))

    for (col, (a, b)) in enumerate(cases)
        x[1:16, col] .= xor_input(a, b, p_vertical, p_horizontal)
        INPUT_BIAS && (x[17, col] = one(FT))
        y[:, col] .= xor_target(a, b, config)
    end

    return x, y, cases
end

function small_weight_generator(weight_scale::FT, seed::Integer)
    rng = Random.MersenneTwister(seed)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> weight_scale * randn(rng, FT))
end

function quadratic_coefficient_for_double_well(double_well::FT)
    # The graph diagonal contributes -0.5 * lp[i] * s[i]^2 through Bilinear.
    # Pick the explicit quadratic term so the combined local potential is
    #   lp * double_well * (s^4 - 2s^2),
    # which has stationary minima at s = -1 and s = 1.
    return FT(0.5) - FT(2) * double_well
end

function initial_bias_generator(config)
    rng = Random.MersenneTwister(config.bias_seed)
    return g -> config.bias_scale .* randn(rng, FT, statelen(g))
end

function xor_architecture(config)
    layer_sizes = (input_units(), config.hidden_sizes..., config.output_units)
    state_type = config.state_mode === :discrete ? Discrete() : Continuous()
    layers = [Layer(layer_sizes[idx], StateSet(-one(FT), one(FT)), state_type, Coords(0, idx, 0)) for idx in eachindex(layer_sizes)]
    wg = small_weight_generator(config.weight_scale, config.weight_seed)
    layer_args = Any[layers[1]]
    for idx in 2:length(layers)
        push!(layer_args, deepcopy(wg), layers[idx])
    end

    bias = config.bias_scale == zero(FT) ?
        (g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))) :
        initial_bias_generator(config)
    local_potential = g -> InteractiveIsing.filltype(Vector, config.local_potential, statelen(g))
    clamping_target = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))
    clamping_beta = InteractiveIsing.UniformArray(zero(FT))
    total_units = sum(layer_sizes)
    output_idxs = (total_units - config.output_units + 1):total_units
    clamping =
        config.target_mode === :readout_hterm ?
        LinearReadoutClamping(output_idxs, readout_vector(config.output_units, config.output_code); β = zero(FT), target = zero(FT)) :
        config.target_mode === :constant_readout_hterm ?
        ConstantLinearReadoutNudge(output_idxs, readout_vector(config.output_units, config.output_code); β = zero(FT), target = zero(FT), free_score = zero(FT)) :
        Clamping(β = clamping_beta, y = clamping_target)
    quadratic_c = quadratic_coefficient_for_double_well(config.double_well)
    hamiltonian =
        Quadratic(c = quadratic_c, localpotential = local_potential) +
        Quartic(c = config.double_well, localpotential = local_potential) +
        InteractiveIsing.Bilinear() +
        InteractiveIsing.MagField(b = bias) +
        clamping

    graph = IsingGraph(layer_args..., hamiltonian; precision = FT, index_set = g -> ToggledIndexSet(g))
    diag(adj(graph)) .= config.local_potential
    if config.skip_weight_scale > zero(FT)
        rng = Random.MersenneTwister(config.weight_seed + 10_000)
        for col in layerrange(graph[1]), row in layerrange(graph[end])
            w = config.skip_weight_scale * randn(rng, FT)
            adj(graph)[row, col] = w
            adj(graph)[col, row] = w
        end
    end
    return graph
end

function langevin_dynamics(config)
    if config.dynamics == :local
        return LocalLangevin(stepsize = config.stepsize, adjusted = config.adjusted, group_steps = 1)
    elseif config.dynamics == :block
        return BlockLangevin(stepsize = config.stepsize, adjusted = config.adjusted, block_size = config.block_size, group_steps = 1)
    elseif config.dynamics == :global
        return GlobalLangevin(stepsize = config.stepsize, adjusted = config.adjusted, group_steps = 1)
    elseif config.dynamics == :metropolis
        return Metropolis()
    else
        error("unknown dynamics $(config.dynamics)")
    end
end

function validation_langevin_dynamics(config)
    if config.dynamics == :local
        return LocalLangevin(stepsize = config.stepsize, adjusted = config.adjusted, group_steps = 1, order = :deterministic)
    elseif config.dynamics == :metropolis
        return Metropolis()
    elseif config.dynamics == :global
        return deepcopy(langevin_dynamics(config))
    elseif config.dynamics == :block
        return deepcopy(langevin_dynamics(config))
    else
        return deepcopy(langevin_dynamics(config))
    end
end

function set_trainer_temperature!(trainer, T::Real)
    InteractiveIsing.temp!(trainer.prototype_graph, FT(T))
    InteractiveIsing.temp!(trainer.validation_graph, FT(T))
    foreach(g -> InteractiveIsing.temp!(g, FT(T)), trainer.worker_graphs)
    return trainer
end

function annealed_temperature(epoch::Integer, epochs::Integer)
    progress = epochs <= 1 ? one(FT) : FT(epoch - 1) / FT(epochs - 1)
    return STOP_TEMP + (START_TEMP - STOP_TEMP) * (one(FT) - progress)^2
end

function run_output!(trainer, x; seed::Integer)
    worker = trainer.validation_worker
    Random.seed!(seed)
    Random.seed!(StatefulAlgorithms.context(worker).dynamics.rng, seed)
    IsingLearning._write_input!(worker, x)
    @assert all(StatefulAlgorithms.context(worker)._state.x .== x)
    StatefulAlgorithms.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)
    output = copy(StatefulAlgorithms.context(worker)._state.equilibrium_state[trainer.layer.output_layer])
    all(isfinite, output) || error("non-finite output")
    return output
end

function evaluate_xor!(trainer, x, y, cases, config; seed_offset::Integer = 10_000)
    set_trainer_temperature!(trainer, STOP_TEMP)
    outputs = map(axes(x, 2)) do idx
        averaged = zeros(FT, config.output_units)
        for repeat_idx in 1:max(1, EVALUATION_REPEATS)
            averaged .+= run_output!(trainer, view(x, :, idx); seed = seed_offset + 1000 * repeat_idx + idx)
        end
        averaged ./= FT(max(1, EVALUATION_REPEATS))
        averaged
    end
    spin_mse = sum(idx -> sum(abs2, outputs[idx] .- view(y, :, idx)) / length(outputs[idx]), eachindex(outputs)) / length(outputs)
    truth = [xor(a, b) for (a, b) in cases]
    score_mse = sum(eachindex(outputs)) do idx
        score = readout_score(outputs[idx], config.output_units, config.output_code)
        target = label_target_score(truth[idx], config)
        abs2(score - target)
    end / length(outputs)
    predictions = map(outputs) do out
        readout_score(out, config.output_units, config.output_code) > zero(FT)
    end
    accuracy = count(==(true), predictions .== truth) / length(truth)
    margin = minimum(abs(readout_score(outputs[idx], config.output_units, config.output_code)) for idx in eachindex(outputs))
    mse = (config.target_mode === :readout ||
           config.target_mode === :readout_hterm ||
           config.target_mode === :constant_readout_hterm) ? score_mse : spin_mse
    return (; outputs, mse, spin_mse, score_mse, accuracy, margin)
end

function finite_namedtuple!(name, nt)
    all(isfinite, nt.w) || error("$name has non-finite weights")
    all(isfinite, nt.b) || error("$name has non-finite biases")
    all(isfinite, nt.α) || error("$name has non-finite local potentials")
    return nothing
end

function normalize_weights!(params, target_rms::FT)
    target_rms > zero(FT) || return params
    isempty(params.w) && return params
    rms = sqrt(sum(abs2, params.w) / FT(length(params.w)))
    isfinite(rms) && rms > zero(FT) || return params
    params.w .*= target_rms / rms
    return params
end

function run_minibatch_checked!(trainer, xbatch, ybatch, batch_gradient, config, epoch)
    MINIT > 0 || throw(ArgumentError("ISING_XOR_SEARCH_MINIT must be positive"))
    IsingLearning._reset_batch_buffers!(trainer)
    worker = only(trainer.workers)

    for sample_idx in axes(xbatch, 2)
        for init_idx in 1:MINIT
            StatefulAlgorithms.isdone(worker) && close(worker)
            IsingLearning._write_example!(worker, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx))
            @assert all(StatefulAlgorithms.context(worker)._state.x .== view(xbatch, :, sample_idx))
            @assert all(StatefulAlgorithms.context(worker)._state.y .== view(ybatch, :, sample_idx))
            Random.seed!(StatefulAlgorithms.context(worker).dynamics.rng, config.seed + 100_000 * epoch + 1000 * sample_idx + init_idx)
            StatefulAlgorithms.reset!(worker)
            run(worker)
            wait(worker)
            close(worker)

            all(state(StatefulAlgorithms.context(worker).dynamics.model[1]) .== view(xbatch, :, sample_idx)) ||
                error("input layer write was not preserved after sample $sample_idx")
            all(isfinite, StatefulAlgorithms.context(worker).plus_capture.captured) || error("non-finite plus phase")
            all(isfinite, StatefulAlgorithms.context(worker).minus_capture.captured) || error("non-finite minus phase")
        end
    end

    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, size(xbatch, 2) * MINIT)
    batch_gradient.w .*= GRADIENT_SIGN
    batch_gradient.b .*= GRADIENT_SIGN
    batch_gradient.α .*= GRADIENT_SIGN
    finite_namedtuple!("batch_gradient", batch_gradient)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    normalize_weights!(trainer.params, config.weight_norm)
    IsingLearning._broadcast_params!(trainer)
    return nothing
end

function freeze_local_potential!(trainer, reference_params)
    trainer.params.α .= reference_params.α
    IsingLearning._broadcast_params!(trainer)
    return trainer
end

function train_config(config)
    Random.seed!(config.seed)
    graph = xor_architecture(config)
    InteractiveIsing.temp!(graph, START_TEMP)
    dynamics = langevin_dynamics(config)
    layer = LayeredIsingGraphLayer(
        () -> xor_architecture(config);
        input_idxs = layerrange(graph[1]),
        output_idxs = layerrange(graph[end]),
        β = config.beta,
        fullsweeps = 1,
        relaxation_steps = config.relaxation_steps,
        dynamics_algorithm = dynamics,
        validation_algorithm = validation_langevin_dynamics(config),
    )

    x, y, cases = xor_dataset(config)
    trainer = init_mnist_trainer(
        layer;
        graph,
        numthreads = WORKER_THREADS,
        optimiser = Optimisers.Adam(config.learning_rate),
    )
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    best_params = deepcopy(trainer.params)
    before = evaluate_xor!(trainer, x, y, cases, config)
    best = (; epoch = 0, metrics = before)

    for epoch in 1:config.epochs
        set_trainer_temperature!(trainer, annealed_temperature(epoch, config.epochs))
        run_minibatch_checked!(trainer, x, y, batch_gradient, config, epoch)
        freeze_local_potential!(trainer, initial_params)
        finite_namedtuple!("params", trainer.params)
        trainer.params.α == initial_params.α || error("local potential changed despite freeze")

        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_xor!(trainer, x, y, cases, config)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b) + sum(abs2, batch_gradient.α))
            param_delta = sqrt(
                sum(abs2, trainer.params.w .- initial_params.w) +
                sum(abs2, trainer.params.b .- initial_params.b)
            )
            weight_rms = sqrt(sum(abs2, trainer.params.w) / FT(length(trainer.params.w)))
            println(
                "  epoch=$epoch mse=$(round(metrics.mse, digits=6)) acc=$(metrics.accuracy) ",
                "margin=$(round(metrics.margin, digits=6)) grad=$(round(grad_norm, digits=6)) ",
                "delta=$(round(param_delta, digits=6)) weight_rms=$(round(weight_rms, digits=6))",
            )
            if metrics.accuracy > best.metrics.accuracy ||
                    (metrics.accuracy == best.metrics.accuracy && metrics.mse < best.metrics.mse)
                best = (; epoch, metrics)
                best_params = deepcopy(trainer.params)
            end
        end
    end

    trainer.params = best_params
    IsingLearning._broadcast_params!(trainer)
    after = evaluate_xor!(trainer, x, y, cases, config)
    close_trainer!(trainer)
    return (; config, before, best, after)
end

function build_configs()
    configs = NamedTuple[]
    idx = 0
    for hidden_sizes in HIDDEN_CONFIGS
        for output_units in OUTPUT_CONFIGS
            for output_code in OUTPUT_CODES
                for target_mode in TARGET_MODES
                    for readout_target in READOUT_TARGETS
                        for local_potential in LOCAL_POTENTIALS
                            for double_well in DOUBLE_WELLS
                                for bias_scale in BIAS_SCALES
                                    for skip_weight_scale in SKIP_WEIGHT_SCALES
                                        for weight_norm in WEIGHT_NORMS
                                            for learning_rate in LEARNING_RATES
                                                for beta in BETAS
                                                    for relaxation_steps in RELAXATION_STEPS_LIST
                                                        for dynamics in DYNAMICS_CONFIGS
                                                            for block_size in BLOCK_SIZES
                                                                for state_mode in STATE_MODES
                                                                    for weight_seed in WEIGHT_SEEDS
                                                                        for bias_seed in BIAS_SEEDS
                                                                            idx += 1
                                                                            stepsize =
                                                                                dynamics === :block ? FT(1e-3) :
                                                                                ADJUSTED ? FT(5e-3) :
                                                                                FT(1e-2)
                                                                            yield_config = (;
                                                                                hidden_sizes = Tuple(hidden_sizes),
                                                                                output_units,
                                                                                output_code,
                                                                                target_mode,
                                                                                readout_target,
                                                                                local_potential,
                                                                                double_well,
                                                                                bias_scale,
                                                                                skip_weight_scale,
                                                                                weight_norm,
                                                                                learning_rate,
                                                                                beta,
                                                                                relaxation_steps,
                                                                                weight_scale = WEIGHT_SCALE,
                                                                                weight_seed,
                                                                                bias_seed,
                                                                                state_mode,
                                                                                seed = BASE_SEED + idx,
                                                                                epochs = DEFAULT_EPOCHS,
                                                                                log_every = LOG_EVERY,
                                                                                dynamics,
                                                                                block_size,
                                                                                adjusted = ADJUSTED,
                                                                                stepsize,
                                                                            )
                                                                            push!(configs, yield_config)
                                                                        end
                                                                    end
                                                                end
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return configs
end

configs = build_configs()
println("Running bespoke XOR search with $(length(configs)) config(s)")

results = NamedTuple[]
for (idx, config) in enumerate(configs)
    println(
        "config $idx/$(length(configs)): ",
        (hidden = config.hidden_sizes, out = config.output_units,
         output_code = config.output_code,
         target_mode = config.target_mode, readout_target = config.readout_target,
         lp = config.local_potential,
         dw = config.double_well, bias = config.bias_scale, skip = config.skip_weight_scale,
         weight_norm = config.weight_norm,
         lr = config.learning_rate, beta = config.beta,
         relaxation = config.relaxation_steps, dynamics = config.dynamics,
         block_size = config.block_size, adjusted = config.adjusted, stepsize = config.stepsize,
         state = config.state_mode, weight_seed = config.weight_seed, bias_seed = config.bias_seed,
         minit = MINIT),
    )
    result = train_config(config)
    push!(results, result)
    println(
        "  before: mse=$(round(result.before.mse, digits=6)) acc=$(result.before.accuracy) ",
        "best: epoch=$(result.best.epoch) mse=$(round(result.best.metrics.mse, digits=6)) acc=$(result.best.metrics.accuracy) ",
        "after: mse=$(round(result.after.mse, digits=6)) spin_mse=$(round(result.after.spin_mse, digits=6)) ",
        "score_mse=$(round(result.after.score_mse, digits=6)) acc=$(result.after.accuracy) ",
        "margin=$(round(result.after.margin, digits=6))",
    )
    if PRINT_OUTPUTS
        for (case, out) in zip(((false, false), (false, true), (true, false), (true, true)), result.after.outputs)
            score = readout_score(out, result.config.output_units, result.config.output_code)
            target = label_target_score(xor(case...), result.config)
            println(
                "    case=$case truth=$(xor(case...)) ",
                "score=$(round(score, digits=6)) target=$(round(target, digits=6)) ",
                "out=", round.(out; digits=4),
            )
        end
    end
end

sort!(results; by = r -> (-r.after.accuracy, r.after.mse))
println("Top configs:")
for (rank, result) in enumerate(first(results, min(10, length(results))))
    c = result.config
    println(
        "#$rank hidden=$(c.hidden_sizes) out=$(c.output_units) target=$(c.target_mode) readout_target=$(c.readout_target) lp=$(c.local_potential) ",
        "output_code=$(c.output_code) ",
        "dw=$(c.double_well) bias=$(c.bias_scale) skip=$(c.skip_weight_scale) weight_norm=$(c.weight_norm) lr=$(c.learning_rate) beta=$(c.beta) relax=$(c.relaxation_steps) ",
        "dynamics=$(c.dynamics) block=$(c.block_size) adjusted=$(c.adjusted) stepsize=$(c.stepsize) ",
        "state=$(c.state_mode) weight_seed=$(c.weight_seed) bias_seed=$(c.bias_seed) ",
        "before_mse=$(round(result.before.mse, digits=6)) before_acc=$(result.before.accuracy) ",
        "best_epoch=$(result.best.epoch) best_mse=$(round(result.best.metrics.mse, digits=6)) best_acc=$(result.best.metrics.accuracy) ",
        "after_mse=$(round(result.after.mse, digits=6)) after_spin_mse=$(round(result.after.spin_mse, digits=6)) ",
        "after_score_mse=$(round(result.after.score_mse, digits=6)) after_acc=$(result.after.accuracy) ",
        "margin=$(round(result.after.margin, digits=6))",
    )
end
