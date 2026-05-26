using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using LuxCore
using Optimisers
using Random
using Serialization
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float64
const XOR_CASES = ((false, false), (false, true), (true, false), (true, true))

Base.@kwdef struct MajorityVoteBaselineConfig{T<:AbstractFloat,S<:AbstractString}
    epochs::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_EPOCHS", "2500"))
    workers::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_WORKERS", "32"))
    log_every::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_LOG_EVERY", "25"))
    repeats_per_case::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_REPEATS", "32"))
    chunks_per_case::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_CHUNKS_PER_CASE", "0"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_EVAL_REPEATS", "64"))
    free_steps::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_FREE_STEPS", "200"))
    nudged_steps::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_NUDGED_STEPS", "80"))
    block_size::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_BLOCK_SIZE", "8"))
    β::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_BETA", "0.5"))
    lr::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_LR", "0.05"))
    temp::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_TEMP", "0.001"))
    stepsize::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_STEPSIZE", "0.05"))
    weight_scale::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_WEIGHT_SCALE", "0.05"))
    bias_scale::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_BIAS_SCALE", "0.1"))
    weight_decay::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_WEIGHT_DECAY", "1e-4"))
    target_mse::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_TARGET_MSE", "0.1"))
    seed::Int = parse(Int, get(ENV, "ISING_XOR_MAJORITY_SEED", "13"))
    init_mode::Symbol = Symbol(get(ENV, "ISING_XOR_MAJORITY_INIT", "random"))
    training_rule::Symbol = Symbol(get(ENV, "ISING_XOR_MAJORITY_RULE", "analytic_teacher"))
    target_free_sign::Symbol = Symbol(get(ENV, "ISING_XOR_MAJORITY_TARGET_FREE_SIGN", "free_minus_target"))
    state_mode::Symbol = Symbol(get(ENV, "ISING_XOR_MAJORITY_STATE_MODE", "discrete"))
    dynamics_mode::Symbol = Symbol(get(ENV, "ISING_XOR_MAJORITY_DYNAMICS_MODE", "metropolis"))
    teacher_input_gain::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_TEACHER_INPUT_GAIN", "16.0"))
    teacher_output_gain::T = parse(FT, get(ENV, "ISING_XOR_MAJORITY_TEACHER_OUTPUT_GAIN", "4.0"))
    outdir::S = get(
        ENV,
        "ISING_XOR_MAJORITY_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "two_input_2x2_hidden_majority_vote_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

struct RandomWeightInitializer{R,T<:AbstractFloat}
    rng::R
    scale::T
end

struct RandomBiasInitializer{R,T<:AbstractFloat}
    rng::R
    scale::T
end

struct MajorityVoteJob{X<:AbstractVector,Y<:AbstractVector}
    case_idx::Int
    x::X
    y::Y
    repeats::Int
end

mutable struct MajorityVoteManagerState{L,G,P,B,O}
    layer::L
    prototype_graph::G
    params::Base.RefValue{P}
    teacher_params::Base.RefValue{P}
    batch_gradient::B
    total_repeats::Base.RefValue{Int}
    opt_state::O
end

"""Initialize one all-to-all edge weight."""
function (initializer::RandomWeightInitializer{R,T})(; dr, c1, c2, dc) where {R,T<:AbstractFloat}
    return initializer.scale * randn(initializer.rng, T)
end

"""Initialize the trainable magnetic field for the full graph."""
function (initializer::RandomBiasInitializer{R,T})(graph::G) where {R,T<:AbstractFloat,G}
    return initializer.scale .* randn(initializer.rng, T, II.statelen(graph))
end

"""Allocate a zero target vector for the clamping term."""
function zero_target(graph::G) where {G}
    return II.filltype(Vector, zero(eltype(graph)), II.statelen(graph))
end

"""Allocate a zero clamping mask for the clamping term."""
function zero_mask(graph::G) where {G}
    return II.filltype(Vector, zero(eltype(graph)), II.statelen(graph))
end

"""Create the toggled active set used by continuous XOR experiments."""
function toggled_index_set(graph::G) where {G}
    return II.ToggledIndexSet(graph)
end

"""Return two-spin XOR samples and four replicated scalar targets."""
function xor_dataset(::Type{T} = FT) where {T<:AbstractFloat}
    x = Matrix{T}(undef, 2, length(XOR_CASES))
    y = Matrix{T}(undef, 4, length(XOR_CASES))
    targets = Vector{T}(undef, length(XOR_CASES))
    for (col, (a, b)) in enumerate(XOR_CASES)
        target = xor(a, b) ? one(T) : -one(T)
        x[:, col] .= (a ? one(T) : -one(T), b ? one(T) : -one(T))
        y[:, col] .= target
        targets[col] = target
    end
    return x, y, targets
end

"""Return the hidden-unit index for each XOR input pattern."""
function hidden_pattern_table(graph::G) where {G}
    hidden_idxs = collect(II.layerrange(graph[2]))
    patterns = [FT[a ? 1 : -1, b ? 1 : -1] for (a, b) in XOR_CASES]
    targets = FT[xor(a, b) ? 1 : -1 for (a, b) in XOR_CASES]
    return hidden_idxs, patterns, targets
end

"""Construct an explicit parameter vector that realizes XOR by four pattern detectors."""
function analytic_majority_params(graph::G, layer::L, ps::P, config::C) where {G,L<:LayeredIsingGraphLayer,P,C<:MajorityVoteBaselineConfig}
    input_idxs = collect(layer.input_layer)
    output_idxs = collect(layer.output_layer)
    hidden_idxs, patterns, targets = hidden_pattern_table(graph)
    input_pos = Dict(idx => pos for (pos, idx) in pairs(input_idxs))
    hidden_pos = Dict(idx => pos for (pos, idx) in pairs(hidden_idxs))
    output_set = Set(output_idxs)
    adjmat = II.adj(graph).sp
    teacher_w = zeros(eltype(ps.w), length(ps.w))
    teacher_b = zeros(eltype(ps.b), length(ps.b))

    for (pos, hidden_idx) in pairs(hidden_idxs)
        teacher_b[hidden_idx] = config.teacher_input_gain
    end

    # Directly write the CSC nonzero vector so the teacher matches Lux's
    # parameter layout exactly.
    for col in axes(adjmat, 2)
        for ptr in adjmat.colptr[col]:(adjmat.colptr[col + 1] - 1)
            row = adjmat.rowval[ptr]
            if haskey(input_pos, row) && haskey(hidden_pos, col)
                teacher_w[ptr] = -config.teacher_input_gain * patterns[hidden_pos[col]][input_pos[row]]
            elseif haskey(hidden_pos, row) && haskey(input_pos, col)
                teacher_w[ptr] = -config.teacher_input_gain * patterns[hidden_pos[row]][input_pos[col]]
            elseif haskey(hidden_pos, row) && col in output_set
                teacher_w[ptr] = -config.teacher_output_gain * targets[hidden_pos[row]]
            elseif row in output_set && haskey(hidden_pos, col)
                teacher_w[ptr] = -config.teacher_output_gain * targets[hidden_pos[col]]
            end
        end
    end

    hasproperty(ps, :α) || return (; w = teacher_w, b = teacher_b)
    return (; w = teacher_w, b = teacher_b, α = copy(ps.α))
end

"""Initialize one free-phase trajectory for the configured basin."""
function initialize_majority_state_body!(isinggraph::G, init_mode) where {G}
    if init_mode === :random
        II.resetstate!(isinggraph)
    elseif init_mode === :zero
        fill!(II.state(isinggraph), zero(eltype(isinggraph)))
    elseif init_mode === :negative
        fill!(II.state(isinggraph), -one(eltype(isinggraph)))
    elseif init_mode === :positive
        fill!(II.state(isinggraph), one(eltype(isinggraph)))
    else
        throw(ArgumentError("unknown XOR majority init mode `$(init_mode)`; use `negative`, `positive`, `zero`, or `random`"))
    end
    return nothing
end

Processes.@ProcessAlgorithm function InitializeMajorityState!(isinggraph::G, init_mode) where {G}
    initialize_majority_state_body!(isinggraph, init_mode)
    return nothing
end

"""Accumulate a target-minus-free energy contrast into the shared gradient buffer."""
function accumulate_target_minus_free_gradient_body!(isinggraph::G, target_state::S, free_state::F, buffers::B) where {G,S<:AbstractVector,F<:AbstractVector,B}
    bilinear = isinggraph.hamiltonian[II.Bilinear]
    magfield = isinggraph.hamiltonian[II.MagField]
    II.parameter_derivative(bilinear, target_state, dJ = buffers.w, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(bilinear, free_state, dJ = buffers.w, buffermode = II.SubtractBuffer())
    II.parameter_derivative(magfield, target_state, db = buffers.b, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(magfield, free_state, db = buffers.b, buffermode = II.SubtractBuffer())
    return buffers
end

"""Accumulate a free-minus-target energy contrast into the shared gradient buffer."""
function accumulate_free_minus_target_gradient_body!(isinggraph::G, target_state::S, free_state::F, buffers::B) where {G,S<:AbstractVector,F<:AbstractVector,B}
    bilinear = isinggraph.hamiltonian[II.Bilinear]
    magfield = isinggraph.hamiltonian[II.MagField]
    II.parameter_derivative(bilinear, free_state, dJ = buffers.w, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(bilinear, target_state, dJ = buffers.w, buffermode = II.SubtractBuffer())
    II.parameter_derivative(magfield, free_state, db = buffers.b, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(magfield, target_state, db = buffers.b, buffermode = II.SubtractBuffer())
    return buffers
end

Processes.@ProcessAlgorithm function AccumulateTargetMinusFreeGradient!(isinggraph, target_state, free_state, buffers)
    accumulate_target_minus_free_gradient_body!(isinggraph, target_state, free_state, buffers)
    return nothing
end

Processes.@ProcessAlgorithm function AccumulateFreeMinusTargetGradient!(isinggraph, target_state, free_state, buffers)
    accumulate_free_minus_target_gradient_body!(isinggraph, target_state, free_state, buffers)
    return nothing
end

Processes.@ProcessAlgorithm function TouchMajorityJob!(x, y)
    return nothing
end

"""Build the all-to-all `2 -> 2x2 -> 4` majority-vote XOR graph."""
function xor_graph(config::C) where {C<:MajorityVoteBaselineConfig}
    rng_w = Random.MersenneTwister(config.seed)
    rng_b = Random.MersenneTwister(config.seed + 1)
    weights = II.AllToAllWeightGenerator(RandomWeightInitializer(rng_w, config.weight_scale))
    bias = RandomBiasInitializer(rng_b, config.bias_scale)
    state_type = if config.state_mode === :discrete
        II.Discrete()
    elseif config.state_mode === :continuous
        II.Continuous()
    else
        throw(ArgumentError("unknown XOR majority state mode `$(config.state_mode)`; use `discrete` or `continuous`"))
    end

    input = II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), state_type, II.Coords(0, 1, 0); periodic = false)
    hidden = II.Layer(2, 2, II.StateSet(-one(FT), one(FT)), state_type, II.Coords(0, 2, 0); periodic = false)
    output = II.Layer(2, 2, II.StateSet(-one(FT), one(FT)), state_type, II.Coords(0, 3, 0); periodic = false)
    hamiltonian = II.Bilinear() + II.MagField(b = bias) + II.Clamping(β = II.UniformArray(zero(FT)), y = zero_target, mask = zero_mask)

    graph = II.IsingGraph(
        input,
        weights,
        hidden,
        deepcopy(weights),
        output,
        hamiltonian;
        precision = FT,
        index_set = toggled_index_set,
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Create the process-backed Lux layer used by training and evaluation."""
function xor_layer(graph::G, config::C) where {G,C<:MajorityVoteBaselineConfig}
    dynamics = if config.dynamics_mode === :metropolis
        II.Metropolis()
    elseif config.dynamics_mode === :langevin
        II.BlockLangevin(
            stepsize = config.stepsize,
            adjusted = false,
            block_size = config.block_size,
            group_steps = 1,
        )
    else
        throw(ArgumentError("unknown XOR majority dynamics mode `$(config.dynamics_mode)`; use `metropolis` or `langevin`"))
    end
    return LayeredIsingGraphLayer(
        graph;
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_steps,
        free_relaxation_steps = config.free_steps,
        nudged_relaxation_steps = config.nudged_steps,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Build one routable free/plus/minus contrastive repeat."""
function majority_single_contrastive_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:MajorityVoteBaselineConfig}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    β = layer.β
    negative_β = -β
    zero_β = zero(β)
    free_steps = layer.free_relaxation_steps
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    init_value = config.init_mode

    return Processes.@Routine begin
        @state x
        @state y
        @state buffers
        @state init_mode = init_value
        @state equilibrium_state = zeros(n_units)
        @state plus_state = zeros(n_units)
        @state minus_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Free phase.
        InitializeMajorityState!(isinggraph = dynamics.model, init_mode = init_mode)
        IsingLearning.apply_input(dynamics.model, x)
        model = @repeat free_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(graph -> II.state(graph), model))

        # Positive nudge from the captured free state.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, β)
        model = @repeat nudged_steps dynamics()
        IsingLearning.copyvector!(plus_state, @transform(graph -> II.state(graph), model))

        # Negative nudge from the same free state.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, negative_β)
        model = @repeat nudged_steps dynamics()
        IsingLearning.copyvector!(minus_state, @transform(graph -> II.state(graph), model))

        IsingLearning.set_clamping_beta!(dynamics.model, zero_β)
        IsingLearning.contrastive_gradient(dynamics.model, plus_state, minus_state, β; buffers = buffers)
    end
end

"""Build the repeated contrastive routine executed once per manager job."""
function majority_contrastive_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:MajorityVoteBaselineConfig}
    contrastive_algorithm = if config.training_rule === :ep
        majority_single_contrastive_algorithm(layer, config)
    elseif config.training_rule === :target_free
        majority_single_target_free_algorithm(layer, config)
    elseif config.training_rule === :analytic_teacher
        majority_single_teacher_algorithm()
    else
        throw(ArgumentError("unknown XOR majority training rule `$(config.training_rule)`; use `ep`, `target_free`, or `analytic_teacher`"))
    end
    repeat_count = repeats_per_majority_job(config)
    return Processes.Routine(contrastive_algorithm, (repeat_count,))
end

"""Build one routable no-op job used by the analytic teacher rule."""
function majority_single_teacher_algorithm()
    return Processes.@Routine begin
        @state x
        @state y
        TouchMajorityJob!(x = x, y = y)
    end
end

"""Build one routable free/target contrastive repeat."""
function majority_single_target_free_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:MajorityVoteBaselineConfig}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    β = layer.β
    zero_β = zero(β)
    free_steps = layer.free_relaxation_steps
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    init_value = config.init_mode
    config.target_free_sign === :free_minus_target && return majority_single_free_minus_target_algorithm(layer, config)
    config.target_free_sign === :target_minus_free && return majority_single_target_minus_free_algorithm(layer, config)
    throw(ArgumentError("unknown target-free sign `$(config.target_free_sign)`; use `target_minus_free` or `free_minus_target`"))
end

"""Build one routable free/target repeat using the free-minus-target sign."""
function majority_single_free_minus_target_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:MajorityVoteBaselineConfig}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    β = layer.β
    zero_β = zero(β)
    free_steps = layer.free_relaxation_steps
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    init_value = config.init_mode

    return Processes.@Routine begin
        @state x
        @state y
        @state buffers
        @state init_mode = init_value
        @state equilibrium_state = zeros(n_units)
        @state plus_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Free phase.
        InitializeMajorityState!(isinggraph = dynamics.model, init_mode = init_mode)
        IsingLearning.apply_input(dynamics.model, x)
        model = @repeat free_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(graph -> II.state(graph), model))

        # Target phase from the captured free state.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, β)
        model = @repeat nudged_steps dynamics()
        IsingLearning.copyvector!(plus_state, @transform(graph -> II.state(graph), model))

        IsingLearning.set_clamping_beta!(dynamics.model, zero_β)
        AccumulateFreeMinusTargetGradient!(isinggraph = dynamics.model, target_state = plus_state, free_state = equilibrium_state, buffers = buffers)
    end
end

"""Build one routable free/target repeat using the target-minus-free sign."""
function majority_single_target_minus_free_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:MajorityVoteBaselineConfig}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    β = layer.β
    zero_β = zero(β)
    free_steps = layer.free_relaxation_steps
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    init_value = config.init_mode

    return Processes.@Routine begin
        @state x
        @state y
        @state buffers
        @state init_mode = init_value
        @state equilibrium_state = zeros(n_units)
        @state plus_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Free phase.
        InitializeMajorityState!(isinggraph = dynamics.model, init_mode = init_mode)
        IsingLearning.apply_input(dynamics.model, x)
        model = @repeat free_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(graph -> II.state(graph), model))

        # Target phase from the captured free state.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, β)
        model = @repeat nudged_steps dynamics()
        IsingLearning.copyvector!(plus_state, @transform(graph -> II.state(graph), model))

        IsingLearning.set_clamping_beta!(dynamics.model, zero_β)
        AccumulateTargetMinusFreeGradient!(isinggraph = dynamics.model, target_state = plus_state, free_state = equilibrium_state, buffers = buffers)
    end
end

"""Return the mutable training context stored in one worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Return the worker graph owned by the free-phase dynamics subcontext."""
function worker_model(worker::W) where {W}
    return Processes.context(worker).dynamics.model
end

"""Create a worker graph with fresh state and shared static parameter arrays."""
function worker_graph(prototype::G, ps::P, config::C) where {G,P,C<:MajorityVoteBaselineConfig}
    IsingLearning.sync_params!(prototype, ps)
    base_bias = II.Force(II.getparam(prototype.hamiltonian, II.MagField, :b))
    hamiltonian = II.Bilinear() + II.MagField(b = base_bias) + II.Clamping(β = II.UniformArray(zero(eltype(prototype))), y = zero_target, mask = zero_mask)
    graph = II.IsingGraph(
        getfield(prototype, :layers)...,
        hamiltonian;
        precision = eltype(prototype),
        adj = II.adj(prototype),
        index_set = toggled_index_set,
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Return the equal number of process repeats assigned to each manager job."""
function repeats_per_majority_job(config::C) where {C<:MajorityVoteBaselineConfig}
    chunks = chunks_per_majority_case(config)
    return config.repeats_per_case ÷ chunks
end

"""Return the number of equal-repeat manager jobs to create for each XOR case."""
function chunks_per_majority_case(config::C) where {C<:MajorityVoteBaselineConfig}
    requested = config.chunks_per_case > 0 ?
        max(1, min(config.repeats_per_case, config.chunks_per_case)) :
        max(1, min(config.repeats_per_case, cld(config.workers, length(XOR_CASES))))
    chunks = requested
    while config.repeats_per_case % chunks != 0
        chunks -= 1
    end
    return chunks
end

"""Build one reusable process worker from the composable contrastive routine."""
function repeated_worker(layer::L, graph::G, config::C) where {L<:LayeredIsingGraphLayer,G,C<:MajorityVoteBaselineConfig}
    algorithm = Processes.resolve(majority_contrastive_algorithm(layer, config))
    state = II.state(graph)
    if config.training_rule === :analytic_teacher
        return Processes.Process(
            algorithm,
            Processes.Init(:_state;
                x = zeros(eltype(graph), length(layer.input_layer)),
                y = zeros(eltype(graph), length(layer.output_layer)),
                buffers = IsingLearning.layer_gradient_buffer(graph),
                equilibrium_state = copy(state),
                plus_state = similar(state),
                minus_state = similar(state),
            );
            repeat = 1,
        )
    end
    return Processes.Process(
        algorithm,
        Processes.Init(:_state;
            model = graph,
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            buffers = IsingLearning.layer_gradient_buffer(graph),
            equilibrium_state = copy(state),
            plus_state = similar(state),
            minus_state = similar(state),
        ),
        Processes.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Allocate a gradient buffer matching the current parameter tree."""
function parameter_buffer(ps::P) where {P}
    buffer = (;
        w = zeros(eltype(ps.w), length(ps.w)),
        b = zeros(eltype(ps.b), length(ps.b)),
    )
    hasproperty(ps, :α) || return buffer
    return merge(buffer, (; α = zeros(eltype(ps.α), length(ps.α))))
end

"""Set every array in a gradient buffer to zero."""
function clear_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Add one worker-local gradient buffer into the batch buffer."""
function add_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

"""Scale one gradient buffer in place."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Write the analytic-teacher Adam gradient into the batch buffer."""
function analytic_teacher_gradient!(buffer::B, ps::P, teacher::P) where {B,P}
    buffer.w .= ps.w .- teacher.w
    buffer.b .= ps.b .- teacher.b
    hasproperty(buffer, :α) && (buffer.α .= ps.α .- teacher.α)
    return buffer
end

"""Return the batch gradient normalization for the configured training rule."""
function gradient_scale(config::C, total::Integer) where {C<:MajorityVoteBaselineConfig}
    if config.training_rule === :ep
        return inv(FT(2) * FT(config.β) * FT(total))
    elseif config.training_rule === :target_free
        return inv(FT(total))
    end
    throw(ArgumentError("unknown XOR majority training rule `$(config.training_rule)`; use `ep`, `target_free`, or `analytic_teacher`"))
end

"""Clear batch and worker buffers before a new minibatch."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    manager.config.training_rule === :analytic_teacher && return manager
    for worker in Processes.workers(manager)
        clear_buffer!(worker_context(worker).buffers)
    end
    return manager
end

"""Flush worker-local contrastive buffers into one averaged batch gradient."""
function flush_majority_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    if manager.config.training_rule === :analytic_teacher
        return analytic_teacher_gradient!(manager.state.batch_gradient, manager.state.params[], manager.state.teacher_params[])
    end
    for worker in Processes.workers(manager)
        ctx = worker_context(worker)
        add_buffer!(manager.state.batch_gradient, ctx.buffers)
        clear_buffer!(ctx.buffers)
    end
    total = manager.state.total_repeats[]
    total > 0 || throw(ArgumentError("cannot flush an XOR batch with zero repeats"))
    scale_buffer!(manager.state.batch_gradient, gradient_scale(manager.config, total))
    manager.config.weight_decay > zero(FT) && (manager.state.batch_gradient.w .+= manager.config.weight_decay .* manager.state.params[].w)
    return manager.state.batch_gradient
end

"""Install updated shared parameters once after a manager batch."""
function sync_worker_params!(manager::M, ps::P) where {M<:Processes.ProcessManager,P}
    if manager.config.training_rule !== :analytic_teacher
        for worker in Processes.workers(manager)
            IsingLearning.sync_params!(worker_model(worker), ps)
        end
    end
    IsingLearning.sync_params!(manager.state.prototype_graph, ps)
    return manager
end

"""Create one worker from the manager-owned layer and prototype graph."""
function majority_makeworker(idx::Integer, manager::M) where {M<:Processes.ProcessManager}
    graph = worker_graph(manager.state.prototype_graph, manager.state.params[], manager.config)
    return repeated_worker(manager.state.layer, graph, manager.config)
end

"""Write one XOR job into a worker before the manager runs it."""
function majority_prepare!(slot, job::J, manager::M) where {J<:MajorityVoteJob,M<:Processes.ProcessManager}
    ctx = worker_context(slot.worker)
    ctx.x .= job.x
    ctx.y .= job.y
    Processes.resetworker!(slot)
    return nothing
end

"""Manager recipe flush hook."""
function majority_flush!(manager::M) where {M<:Processes.ProcessManager}
    return flush_majority_buffers!(manager)
end

"""Create the `ProcessManager` that owns the baseline training workers."""
function majority_manager(layer::L, graph::G, ps::P, config::C) where {L<:LayeredIsingGraphLayer,G,P,C<:MajorityVoteBaselineConfig}
    optimiser = Optimisers.Adam(config.lr)
    teacher = analytic_majority_params(graph, layer, ps, config)
    state = MajorityVoteManagerState(layer, graph, Ref(ps), Ref(teacher), parameter_buffer(ps), Ref(0), Optimisers.setup(optimiser, ps))
    recipe = (;
        makeworker = majority_makeworker,
        prepare! = majority_prepare!,
        flush! = majority_flush!,
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = MajorityVoteJob{Vector{FT},Vector{FT}},
    )
end

"""
Create equal-repeat manager jobs for one full XOR minibatch.

The contrastive routine stores the repeat count in its outer `Routine`, while
each manager job only selects one XOR case and one equal-sized chunk of that
case. By default this makes enough chunks to occupy the configured workers:
with 4 XOR cases and 32 workers, `chunks_per_case == 8`, so each epoch has 32
jobs. If the requested repeat count is not divisible by that chunk count, the
chunk count is reduced to the nearest divisor so every worker execution has the
same repeat count.
"""
function majority_jobs(config::C, x::X, y::Y) where {C<:MajorityVoteBaselineConfig,X<:AbstractMatrix,Y<:AbstractMatrix}
    chunks_per_case = chunks_per_majority_case(config)
    repeats = repeats_per_majority_job(config)
    jobs = MajorityVoteJob{Vector{FT},Vector{FT}}[]
    for case_idx in axes(x, 2)
        for chunk in 1:chunks_per_case
            push!(jobs, MajorityVoteJob(case_idx, copy(view(x, :, case_idx)), copy(view(y, :, case_idx)), repeats))
        end
    end
    return jobs
end

"""Run one manager minibatch and synchronize updated parameters."""
function run_majority_batch!(manager::M, jobs::J) where {M<:Processes.ProcessManager,J}
    clear_manager_buffers!(manager)
    manager.state.total_repeats[] = sum(job.repeats for job in jobs)
    Processes.run!(manager, jobs, Processes.Dynamic())
    manager.state.opt_state, ps_new = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = ps_new
    sync_worker_params!(manager, ps_new)
    return ps_new
end

"""Evaluate majority-vote scores, MSE, margins, and accuracy over XOR cases."""
function evaluate_majority(layer::L, ps::P, st::S, x::X, y::Y, targets::Z, config::C) where {L<:LayeredIsingGraphLayer,P,S,X,Y,Z,C<:MajorityVoteBaselineConfig}
    outputs = zeros(FT, size(y))
    vote_scores = zeros(FT, length(targets))
    graph = st.graph
    dynamics = deepcopy(layer.validation_algorithm)
    dynamics_context = Processes.init(dynamics, (; model = graph))
    IsingLearning.sync_params!(graph, ps)
    for repeat_idx in 1:config.eval_repeats
        for sample_idx in axes(x, 2)
            initialize_majority_state_body!(graph, config.init_mode)
            IsingLearning.apply_input(graph, view(x, :, sample_idx))
            for step_idx in 1:layer.relaxation_steps
                Processes.step!(dynamics, dynamics_context)
            end
            out = IsingLearning.graph_view(graph, layer.output_layer)
            outputs[:, sample_idx] .+= out
            vote_scores[sample_idx] += mean(out) >= zero(FT) ? one(FT) : -one(FT)
        end
    end
    outputs ./= FT(config.eval_repeats)
    vote_scores ./= FT(config.eval_repeats)
    spin_scores = vec(mean(outputs; dims = 1))
    predictions = ifelse.(vote_scores .>= zero(FT), one(FT), -one(FT))
    margins = targets .* vote_scores
    return (;
        mse = mean(abs2, vote_scores .- targets),
        output_mse = mean(abs2, outputs .- y),
        spin_score_mse = mean(abs2, spin_scores .- targets),
        accuracy = mean(predictions .== targets),
        all_correct = all(predictions .== targets),
        min_margin = minimum(margins),
        mean_margin = mean(margins),
        scores = copy(vote_scores),
        spin_scores = copy(spin_scores),
        outputs = copy(outputs),
        predictions = copy(predictions),
    )
end

"""Format one numeric vector as a pipe-separated CSV cell."""
function format_vector(values::V; digits::Int = 4) where {V<:AbstractVector}
    rounded = round.(Float64.(values); digits)
    return join(rounded, "|")
end

"""Format one numeric matrix as a pipe-separated CSV cell."""
function format_matrix(values::M; digits::Int = 4) where {M<:AbstractMatrix}
    rounded = round.(Float64.(vec(values)); digits)
    return join(rounded, "|")
end

"""Append one row to the metrics CSV."""
function append_metrics!(path::P, row::R) where {P<:AbstractString,R}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    io = open(path, "a")
    needs_header && println(io, join(names, ","))
    println(io, join((getproperty(row, name) for name in names), ","))
    close(io)
    return path
end

"""Write a concise README for one baseline run."""
function write_run_readme!(path::P, config::C, jobs::J, final_metrics::M) where {P<:AbstractString,C<:MajorityVoteBaselineConfig,J,M}
    io = open(path, "w")
    println(io, "# Two-Input 2x2 Hidden Majority-Vote XOR Baseline")
    println(io)
    println(io, "Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.")
    println(io)
    println(io, "- epochs: `$(config.epochs)`")
    println(io, "- workers: `$(config.workers)`")
    println(io, "- Julia threads observed: `$(Threads.nthreads())`")
    println(io, "- repeats per XOR case: `$(config.repeats_per_case)`")
    println(io, "- jobs per epoch: `$(length(jobs))`")
    println(io, "- free/nudged steps: `$(config.free_steps)` / `$(config.nudged_steps)`")
    println(io, "- state mode: `$(config.state_mode)`")
    println(io, "- dynamics: `$(config.dynamics_mode)`")
    println(io, "- training rule: `$(config.training_rule)`")
    config.training_rule === :target_free && println(io, "- target-free sign: `$(config.target_free_sign)`")
    println(io, "- optimizer: `Adam`")
    println(io, "- learning rate: `$(config.lr)`")
    println(io, "- beta: `$(config.β)`")
    println(io, "- final majority-vote MSE: `$(final_metrics.mse)`")
    println(io, "- final analog spin-score MSE: `$(final_metrics.spin_score_mse)`")
    println(io, "- final accuracy: `$(final_metrics.accuracy)`")
    close(io)
    return path
end

"""Train the baseline and write metrics plus serialized final parameters."""
function run_experiment(config::C = MajorityVoteBaselineConfig()) where {C<:MajorityVoteBaselineConfig}
    config.workers > 0 || throw(ArgumentError("ISING_XOR_MAJORITY_WORKERS must be positive"))
    config.repeats_per_case > 0 || throw(ArgumentError("ISING_XOR_MAJORITY_REPEATS must be positive"))
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    mkpath(config.outdir)

    graph = xor_graph(config)
    layer = xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 10), layer)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 11), layer)
    x, y, targets = xor_dataset(FT)
    jobs = majority_jobs(config, x, y)
    manager = majority_manager(layer, graph, ps, config)
    metrics_path = joinpath(config.outdir, "metrics.csv")

    final_metrics = nothing
    reached_target = false
    for epoch in 0:config.epochs
        if epoch == 0 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_majority(layer, manager.state.params[], st, x, y, targets, config)
            final_metrics = metrics
            reached_target = metrics.mse <= config.target_mse && metrics.all_correct
            append_metrics!(
                metrics_path,
                (;
                    epoch,
                    mse = metrics.mse,
                    output_mse = metrics.output_mse,
                    spin_score_mse = metrics.spin_score_mse,
                    accuracy = metrics.accuracy,
                    all_correct = metrics.all_correct,
                    min_margin = metrics.min_margin,
                    mean_margin = metrics.mean_margin,
                    scores = format_vector(metrics.scores),
                    spin_scores = format_vector(metrics.spin_scores),
                    predictions = format_vector(metrics.predictions; digits = 0),
                    outputs = format_matrix(metrics.outputs),
                ),
            )
            println("epoch=", epoch, " mse=", round(metrics.mse; digits = 6), " accuracy=", metrics.accuracy, " min_margin=", round(metrics.min_margin; digits = 4))
        end
        (epoch == config.epochs || reached_target) && break
        run_majority_batch!(manager, jobs)
    end

    io = open(joinpath(config.outdir, "final_params.bin"), "w")
    serialize(io, (; ps = manager.state.params[], config, targets))
    close(io)
    write_run_readme!(joinpath(config.outdir, "README.md"), config, jobs, final_metrics)
    close(manager)
    println("saved ", config.outdir)
    return config.outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment()
end
