using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CairoMakie
using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using LuxCore
using Optimisers
using Random
using Serialization
using Statistics

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float32

const CNN_XOR_CASES = ((false, false), (false, true), (true, false), (true, true))
const CNN_HIDDEN1_SCALE_REF = Ref(FT(0.025))
const CNN_HIDDEN2_SCALE_REF = Ref(FT(0.025))
const CNN_OUTPUT_SCALE_REF = Ref(FT(0.05))
const CNN_HIDDEN1_RNG_REF = Ref(Random.MersenneTwister(1))
const CNN_HIDDEN2_RNG_REF = Ref(Random.MersenneTwister(2))

Base.@kwdef struct LocalCNNXORConfig
    name::String = "r1_1_r2_1"
    epochs::Int = parse(Int, get(ENV, "ISING_XOR_CNN_EPOCHS", "100"))
    workers::Int = parse(Int, get(ENV, "ISING_XOR_CNN_WORKERS", "32"))
    log_every::Int = parse(Int, get(ENV, "ISING_XOR_CNN_LOG_EVERY", "10"))
    snapshot_every::Int = parse(Int, get(ENV, "ISING_XOR_CNN_SNAPSHOT_EVERY", "20"))
    plot::Bool = parse(Bool, lowercase(get(ENV, "ISING_XOR_CNN_PLOT", "true")))
    repeats_per_case::Int = parse(Int, get(ENV, "ISING_XOR_CNN_REPEATS", "64"))
    chunks_per_case::Int = parse(Int, get(ENV, "ISING_XOR_CNN_CHUNKS_PER_CASE", "0"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_XOR_CNN_EVAL_REPEATS", "64"))
    input_side::Int = parse(Int, get(ENV, "ISING_XOR_CNN_INPUT_SIDE", "8"))
    hidden1_side::Int = parse(Int, get(ENV, "ISING_XOR_CNN_HIDDEN1_SIDE", "8"))
    hidden2_side::Int = parse(Int, get(ENV, "ISING_XOR_CNN_HIDDEN2_SIDE", "4"))
    output_side::Int = parse(Int, get(ENV, "ISING_XOR_CNN_OUTPUT_SIDE", "4"))
    output_mode::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_CNN_OUTPUT_MODE", "two_class")))
    layer1_radius::Int = parse(Int, get(ENV, "ISING_XOR_CNN_LAYER1_RADIUS", get(ENV, "ISING_XOR_CNN_RADIUS", "1")))
    layer2_radius::Int = parse(Int, get(ENV, "ISING_XOR_CNN_LAYER2_RADIUS", "1"))
    hidden_periodic::Bool = parse(Bool, lowercase(get(ENV, "ISING_XOR_CNN_HIDDEN_PERIODIC", "false")))
    free_sweeps::Int = parse(Int, get(ENV, "ISING_XOR_CNN_FREE_SWEEPS", "20"))
    nudged_sweeps::Int = parse(Int, get(ENV, "ISING_XOR_CNN_NUDGED_SWEEPS", "20"))
    β::FT = parse(FT, get(ENV, "ISING_XOR_CNN_BETA", "1.0"))
    lr::FT = parse(FT, get(ENV, "ISING_XOR_CNN_LR", "0.002"))
    lr_decay::FT = parse(FT, get(ENV, "ISING_XOR_CNN_LR_DECAY", "0.995"))
    lr_min::FT = parse(FT, get(ENV, "ISING_XOR_CNN_LR_MIN", "0.0002"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_XOR_CNN_WEIGHT_DECAY", "0.0001"))
    optimizer::String = lowercase(get(ENV, "ISING_XOR_CNN_OPTIMIZER", "adam"))
    dynamics::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_CNN_DYNAMICS", "block")))
    block_size::Int = parse(Int, get(ENV, "ISING_XOR_CNN_BLOCK_SIZE", "8"))
    temp::FT = parse(FT, get(ENV, "ISING_XOR_CNN_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_XOR_CNN_STEPSIZE", "0.1"))
    max_drift_fraction::FT = parse(FT, get(ENV, "ISING_XOR_CNN_MAX_DRIFT", "1.0"))
    inter_weight_scale::FT = parse(FT, get(ENV, "ISING_XOR_CNN_INTER_SCALE", "0.2"))
    hidden_internal_scale::FT = parse(FT, get(ENV, "ISING_XOR_CNN_HIDDEN_INTERNAL_SCALE", "0.025"))
    output_internal_scale::FT = parse(FT, get(ENV, "ISING_XOR_CNN_OUTPUT_INTERNAL_SCALE", "0.0"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_XOR_CNN_BIAS_SCALE", "0.02"))
    init_mode::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_CNN_INIT_MODE", "zero")))
    seed::Int = parse(Int, get(ENV, "ISING_XOR_CNN_SEED", "4103"))
    outdir::String = get(
        ENV,
        "ISING_XOR_CNN_OUTDIR",
        joinpath(
            @__DIR__,
            "experiments",
            "current",
            "input_8x8_hidden1_8x8_hidden2_4x4_output_4x4_two_class",
        ),
    )
end

struct LocalCNNXORJob{X<:AbstractVector,Y<:AbstractVector}
    case_idx::Int
    x::X
    y::Y
    repeats::Int
end

struct AveragedCNNContrastiveStep{S} <: Processes.ProcessAlgorithm
    base::S
end

struct CNNContrastiveStep{D,N,T,I} <: Processes.ProcessAlgorithm
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    β::T
    input_dim::Int
    output_dim::Int
    free_relaxation_steps::Int
    nudged_relaxation_steps::Int
    init_mode::I
end

mutable struct LocalCNNManagerState{P,B,O}
    params::Base.RefValue{P}
    batch_gradient::B
    total_repeats::Base.RefValue{Int}
    opt_state::O
end

"""Parse a comma-separated integer list, returning `default` when empty."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Parse a comma-separated boolean list, returning `default` when empty."""
function parse_bool_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Bool}}
    isempty(strip(value)) && return default
    return [parse(Bool, lowercase(strip(part))) for part in split(value, ",") if !isempty(strip(part))]
end

"""Return a copy of `config` with selected fields replaced."""
function copy_config(config::C; kwargs...) where {C<:LocalCNNXORConfig}
    fields = Dict{Symbol,Any}(name => getfield(config, name) for name in fieldnames(C))
    for (key, value) in kwargs
        fields[key] = value
    end
    return LocalCNNXORConfig(; fields...)
end

"""Number of graph spins used to convert sweeps into process steps."""
function cnn_nunits(config::C) where {C<:LocalCNNXORConfig}
    return config.input_side^2 + config.hidden1_side^2 + config.hidden2_side^2 + config.output_side^2
end

"""Create a centered square topology so differently sized layers line up."""
function centered_topology(side::Integer, reference_side::Integer; periodic::Bool)
    origin = (Float64(reference_side - side) / 2, Float64(reference_side - side) / 2)
    return II.SquareTopology((Int(side), Int(side)); origin, periodic)
end

"""Return true when two world coordinates fall inside the local square fanout."""
function inside_local_window(c1::C1, c2::C2, radius::Integer) where {C1,C2}
    return max(abs(c1[1] - c2[1]), abs(c1[2] - c2[2])) <= Float64(radius) + 1e-6
end

"""Generate random local inter-layer couplings within one square fanout window."""
function inter_layer_generator(radius::Integer, scale::T, rng::R) where {T<:Real,R<:Random.AbstractRNG}
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> inside_local_window(c1, c2, radius) ? FT(scale) * randn(rng, FT) : zero(FT),
        rng,
    )
end

"""Named hidden-1 callback used by `WeightGenerator`."""
function hidden1_internal_weight(; dr, c1, c2, dc)
    return CNN_HIDDEN1_SCALE_REF[] * randn(CNN_HIDDEN1_RNG_REF[], FT)
end

"""Named hidden-2 callback used by `WeightGenerator`."""
function hidden2_internal_weight(; dr, c1, c2, dc)
    return CNN_HIDDEN2_SCALE_REF[] * randn(CNN_HIDDEN2_RNG_REF[], FT)
end

"""Named output callback used by `WeightGenerator` for coherent voting."""
function output_internal_weight(; dr, c1, c2, dc)
    return CNN_OUTPUT_SCALE_REF[]
end

"""Generate recurrent local hidden-1 couplings."""
function hidden1_internal_generator(radius::Integer, scale::T, rng::R) where {T<:Real,R<:Random.AbstractRNG}
    CNN_HIDDEN1_SCALE_REF[] = FT(scale)
    CNN_HIDDEN1_RNG_REF[] = rng
    return II.WeightGenerator(hidden1_internal_weight, Int(radius), rng; symmetric = true)
end

"""Generate recurrent local hidden-2 couplings."""
function hidden2_internal_generator(radius::Integer, scale::T, rng::R) where {T<:Real,R<:Random.AbstractRNG}
    CNN_HIDDEN2_SCALE_REF[] = FT(scale)
    CNN_HIDDEN2_RNG_REF[] = rng
    return II.WeightGenerator(hidden2_internal_weight, Int(radius), rng; symmetric = true)
end

"""Generate ferromagnetic local output couplings for coherent majority voting."""
function output_internal_generator(scale::T) where {T<:Real}
    CNN_OUTPUT_SCALE_REF[] = FT(scale)
    return II.WeightGenerator(output_internal_weight, 1, Random.MersenneTwister(1); symmetric = true)
end

"""Return the vertical bipolar pattern for a square layer."""
function vertical_pattern(side::Integer, ::Type{T} = FT) where {T<:AbstractFloat}
    pattern = Matrix{T}(undef, Int(side), Int(side))
    split = Int(side) ÷ 2
    for col in axes(pattern, 2), row in axes(pattern, 1)
        pattern[row, col] = row <= split ? -one(T) : one(T)
    end
    return vec(pattern)
end

"""Return the horizontal bipolar pattern for a square layer."""
function horizontal_pattern(side::Integer, ::Type{T} = FT) where {T<:AbstractFloat}
    pattern = Matrix{T}(undef, Int(side), Int(side))
    split = Int(side) ÷ 2
    for col in axes(pattern, 2), row in axes(pattern, 1)
        pattern[row, col] = col <= split ? -one(T) : one(T)
    end
    return vec(pattern)
end

"""Return the false/true distributed output targets for pattern readout."""
function output_pattern_targets(config::C, ::Type{T} = FT) where {C<:LocalCNNXORConfig,T<:AbstractFloat}
    return vertical_pattern(config.output_side, T), horizontal_pattern(config.output_side, T)
end

"""Return direct two-class replicated output targets for false and true."""
function output_twoclass_targets(config::C, ::Type{T} = FT) where {C<:LocalCNNXORConfig,T<:AbstractFloat}
    iseven(config.output_side^2) || throw(ArgumentError("two_class output mode needs an even number of output units"))
    split = config.output_side^2 ÷ 2
    false_target = Vector{T}(undef, config.output_side^2)
    true_target = similar(false_target)
    false_target[1:split] .= one(T)
    false_target[(split + 1):end] .= -one(T)
    true_target[1:split] .= -one(T)
    true_target[(split + 1):end] .= one(T)
    return false_target, true_target
end

"""Write the output target vector for one XOR case."""
function write_cnn_target!(dest::D, config::C, label::Bool) where {D<:AbstractVector,C<:LocalCNNXORConfig}
    if config.output_mode === :majority
        dest .= label ? one(eltype(dest)) : -one(eltype(dest))
    elseif config.output_mode === :pattern
        false_pattern, true_pattern = output_pattern_targets(config, eltype(dest))
        dest .= label ? true_pattern : false_pattern
    elseif config.output_mode === :two_class
        false_target, true_target = output_twoclass_targets(config, eltype(dest))
        dest .= label ? true_target : false_target
    else
        throw(ArgumentError("unknown XOR CNN output mode `$(config.output_mode)`; use `majority`, `pattern`, or `two_class`"))
    end
    return dest
end

"""Return static 8x8 checkerboard input patterns and configured output targets."""
function cnn_xor_dataset(config::C, ::Type{T} = FT) where {C<:LocalCNNXORConfig,T<:AbstractFloat}
    x = Matrix{T}(undef, config.input_side^2, length(CNN_XOR_CASES))
    y = Matrix{T}(undef, config.output_side^2, length(CNN_XOR_CASES))

    for (case_idx, (a, b)) in enumerate(CNN_XOR_CASES)
        a_value = a ? one(T) : -one(T)
        b_value = b ? one(T) : -one(T)
        input_pattern = Matrix{T}(undef, config.input_side, config.input_side)
        for col in axes(input_pattern, 2), row in axes(input_pattern, 1)
            input_pattern[row, col] = isodd(row + col) ? a_value : b_value
        end
        x[:, case_idx] .= vec(input_pattern)
        write_cnn_target!(view(y, :, case_idx), config, xor(a, b))
    end
    return x, y
end

"""Build the local CNN-like graph: `8x8 -> 8x8 -> 4x4 -> 4x4` by default."""
function cnn_xor_graph(config::C) where {C<:LocalCNNXORConfig}
    layer1_radius = config.layer1_radius
    layer2_radius = config.layer2_radius
    reference_side = config.input_side
    rng_inter_01 = Random.MersenneTwister(config.seed)
    rng_inter_12 = Random.MersenneTwister(config.seed + 1)
    rng_inter_2o = Random.MersenneTwister(config.seed + 2)
    rng_hidden_1 = Random.MersenneTwister(config.seed + 3)
    rng_hidden_2 = Random.MersenneTwister(config.seed + 4)
    rng_b = Random.MersenneTwister(config.seed + 5)

    input = II.Layer(
        config.input_side,
        config.input_side,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        centered_topology(config.input_side, reference_side; periodic = false),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    hidden1 = II.Layer(
        config.hidden1_side,
        config.hidden1_side,
        II.StateSet(-one(FT), one(FT)),
        hidden1_internal_generator(layer1_radius, config.hidden_internal_scale, rng_hidden_1),
        II.Continuous(),
        centered_topology(config.hidden1_side, reference_side; periodic = config.hidden_periodic),
        II.Coords(0, 100, 0);
        periodic = config.hidden_periodic,
    )
    hidden2 = II.Layer(
        config.hidden2_side,
        config.hidden2_side,
        II.StateSet(-one(FT), one(FT)),
        hidden2_internal_generator(layer2_radius, config.hidden_internal_scale, rng_hidden_2),
        II.Continuous(),
        centered_topology(config.hidden2_side, reference_side; periodic = config.hidden_periodic),
        II.Coords(0, 220, 0);
        periodic = config.hidden_periodic,
    )
    output = II.Layer(
        config.output_side,
        config.output_side,
        II.StateSet(-one(FT), one(FT)),
        output_internal_generator(config.output_internal_scale),
        II.Continuous(),
        centered_topology(config.output_side, reference_side; periodic = false),
        II.Coords(0, 330, 0);
        periodic = false,
    )

    bias = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    graph = II.IsingGraph(
        input,
        inter_layer_generator(layer1_radius, config.inter_weight_scale, rng_inter_01),
        hidden1,
        inter_layer_generator(layer2_radius, config.inter_weight_scale, rng_inter_12),
        hidden2,
        inter_layer_generator(layer2_radius, config.inter_weight_scale, rng_inter_2o),
        output,
        II.Bilinear() + II.MagField(b = bias) + II.Clamping(β = II.UniformArray(zero(FT)), y = target, mask = mask);
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Create the sampler used for free, nudged, and validation trajectories."""
function cnn_xor_dynamics(config::C) where {C<:LocalCNNXORConfig}
    config.dynamics === :local && return II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    config.dynamics === :block && return II.BlockLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = config.max_drift_fraction,
        adjusted = false,
        block_size = config.block_size,
        group_steps = 1,
    )
    throw(ArgumentError("unknown XOR CNN dynamics `$(config.dynamics)`; use `local` or `block`"))
end

"""Initialize a graph state according to the configured free-phase basin."""
function initialize_cnn_state!(model::G, init_mode::I) where {G,I}
    if init_mode === :random
        II.resetstate!(model)
    elseif init_mode === :zero
        fill!(II.state(model), zero(eltype(model)))
    else
        throw(ArgumentError("unknown XOR CNN init mode `$(init_mode)`; use `random` or `zero`"))
    end
    return model
end

"""Wrap the graph with the standard Learning contrastive layer."""
function cnn_xor_layer(graph::G, config::C) where {G,C<:LocalCNNXORConfig}
    dynamics = cnn_xor_dynamics(config)
    nsteps_free = config.free_sweeps * II.nstates(graph)
    nsteps_nudged = config.nudged_sweeps * II.nstates(graph)
    return LayeredIsingGraphLayer(
        graph;
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = nsteps_free,
        free_relaxation_steps = nsteps_free,
        nudged_relaxation_steps = nsteps_nudged,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Create the local contrastive step with configurable free-state init."""
function CNNContrastiveStep(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:LocalCNNXORConfig}
    return CNNContrastiveStep(
        deepcopy(layer.dynamics_algorithm),
        deepcopy(layer.nudged_dynamics_algorithm),
        layer.β,
        length(layer.input_layer),
        length(layer.output_layer),
        layer.free_relaxation_steps,
        layer.nudged_relaxation_steps,
        config.init_mode,
    )
end

"""Initialize reusable storage for one configurable-init contrastive worker."""
function Processes.init(step::CNNContrastiveStep, context)
    model = context.model
    T = eltype(model)
    x = get(context, :x, zeros(T, step.input_dim))
    y = get(context, :y, zeros(T, step.output_dim))
    buffers = get(context, :buffers, IsingLearning.layer_gradient_buffer(model))
    equilibrium_state = get(context, :equilibrium_state, copy(II.state(model)))
    plus_state = get(context, :plus_state, similar(equilibrium_state))
    minus_state = get(context, :minus_state, similar(equilibrium_state))
    free_context = Processes.init(step.dynamics_algorithm, (; model))
    nudged_context = Processes.init(step.nudged_dynamics_algorithm, (; model))
    return (; model, x, y, buffers, equilibrium_state, plus_state, minus_state, free_context, nudged_context)
end

"""Run free, positive-nudged, and negative-nudged phases for one sample."""
function Processes.step!(step::CNNContrastiveStep, context)
    model = context.model
    β = step.β

    initialize_cnn_state!(model, step.init_mode)
    IsingLearning.apply_input(model, context.x)
    IsingLearning.relax_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
    context.equilibrium_state .= II.state(model)

    II.state(model) .= context.equilibrium_state
    IsingLearning.apply_input(model, context.x)
    IsingLearning.apply_targets(model, context.y)
    IsingLearning.set_clamping_beta!(model, β)
    IsingLearning.relax_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.plus_state .= II.state(model)

    II.state(model) .= context.equilibrium_state
    IsingLearning.apply_input(model, context.x)
    IsingLearning.apply_targets(model, context.y)
    IsingLearning.set_clamping_beta!(model, -β)
    IsingLearning.relax_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.minus_state .= II.state(model)

    IsingLearning.set_clamping_beta!(model, zero(β))
    IsingLearning.contrastive_gradient(model, context.plus_state, context.minus_state, β; buffers = context.buffers)
    return nothing
end

"""Keep the configurable-init contrastive context reusable."""
function Processes.cleanup(step::CNNContrastiveStep, context)
    return nothing
end

"""Initialize the worker-local input-averaging wrapper."""
function Processes.init(step::AveragedCNNContrastiveStep{S}, context) where {S}
    base_context = Processes.init(step.base, context)
    repeats = get(context, :repeats, Ref(1))
    repeats_ref = repeats isa Base.RefValue ? repeats : Ref(Int(repeats))
    return (; base_context, repeats = repeats_ref)
end

"""Run several random-start contrastive repeats inside one worker execution."""
function Processes.step!(step::AveragedCNNContrastiveStep{S}, context) where {S}
    @inbounds for _ in 1:context.repeats[]
        Processes.step!(step.base, context.base_context)
    end
    return nothing
end

"""Keep the averaged worker context reusable."""
function Processes.cleanup(step::AveragedCNNContrastiveStep{S}, context) where {S}
    return nothing
end

"""Return the mutable state subcontext of one manager-owned worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Create a worker graph with fresh state and shared static parameter arrays."""
function worker_graph(prototype::G, ps::P, config::C) where {G,P,C<:LocalCNNXORConfig}
    IsingLearning.sync_params!(prototype, ps)
    base_bias = II.Force(II.getparam(prototype.hamiltonian, II.MagField, :b))
    target = g -> II.filltype(Vector, zero(eltype(g)), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(eltype(g)), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = base_bias) + II.Clamping(β = II.UniformArray(zero(eltype(prototype))), y = target, mask = mask)
    graph = II.IsingGraph(
        getfield(prototype, :layers)...,
        hamiltonian;
        precision = eltype(prototype),
        adj = II.adj(prototype),
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Build one reusable Process worker for the local CNN-like contrastive step."""
function averaged_worker(layer::L, graph::G, config::C) where {L<:LayeredIsingGraphLayer,G,C<:LocalCNNXORConfig}
    step = AveragedCNNContrastiveStep(CNNContrastiveStep(layer, config))
    return Processes.Process(
        :_state => step,
        Processes.Init(:_state;
            model = graph,
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            repeats = Ref(1),
        );
        repeat = 1,
    )
end

"""Allocate a gradient buffer with the same shape as the Lux parameter tree."""
function parameter_buffer(ps::P) where {P}
    buffer = (;
        w = zeros(eltype(ps.w), length(ps.w)),
        b = zeros(eltype(ps.b), length(ps.b)),
    )
    hasproperty(ps, :α) || return buffer
    return merge(buffer, (; α = zeros(eltype(ps.α), length(ps.α))))
end

"""Zero all arrays in a gradient buffer."""
function clear_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Accumulate one gradient buffer into another."""
function add_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

"""Scale every gradient-buffer array in place."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Clear the batch buffer and every worker-local contrastive buffer."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        clear_buffer!(worker_context(worker).base_context.buffers)
    end
    return manager
end

"""Merge all worker-local buffers once after a minibatch drains."""
function flush_cnn_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in Processes.workers(manager)
        ctx = worker_context(worker).base_context
        add_buffer!(manager.state.batch_gradient, ctx.buffers)
        clear_buffer!(ctx.buffers)
    end
    total = manager.state.total_repeats[]
    total > 0 || throw(ArgumentError("cannot flush a batch with zero repeats"))
    scale_buffer!(manager.state.batch_gradient, inv(FT(2) * FT(manager.config.β) * FT(total)))
    return manager.state.batch_gradient
end

"""Add L2 weight decay to trainable couplings before the optimizer step."""
function add_weight_decay!(gradient::G, ps::P, λ::T) where {G,P,T<:Real}
    λ > zero(T) || return gradient
    gradient.w .+= λ .* ps.w
    return gradient
end

"""Return the optimizer learning rate for a one-indexed update number."""
function learning_rate_at(config::C, update_idx::Integer) where {C<:LocalCNNXORConfig}
    exponent = max(Int(update_idx) - 1, 0)
    scheduled = config.lr * config.lr_decay^exponent
    return max(config.lr_min, scheduled)
end

"""Adjust the Optimisers.jl state in place while preserving Adam moments."""
function adjust_learning_rate!(manager::M, config::C, update_idx::Integer) where {M<:Processes.ProcessManager,C<:LocalCNNXORConfig}
    η = learning_rate_at(config, update_idx)
    Optimisers.adjust!(manager.state.opt_state, η)
    return η
end

"""Install updated shared parameters once after a manager batch."""
function sync_worker_params!(manager::M, ps::P) where {M<:Processes.ProcessManager,P}
    worker = first(Processes.workers(manager))
    IsingLearning.sync_params!(worker_context(worker).base_context.model, ps)
    return manager
end

"""Create a ProcessManager that owns reusable averaged contrastive workers."""
function cnn_xor_manager(layer::L, graph::G, ps::P, config::C) where {L<:LayeredIsingGraphLayer,G,P,C<:LocalCNNXORConfig}
    optimizer_name = lowercase(config.optimizer)
    optimiser = optimizer_name == "adam" ? Optimisers.Adam(config.lr) :
        optimizer_name in ("descent", "sgd") ? Optimisers.Descent(config.lr) :
        throw(ArgumentError("unknown XOR optimizer `$(config.optimizer)`; use `adam` or `descent`"))
    state = LocalCNNManagerState(Ref(ps), parameter_buffer(ps), Ref(0), Optimisers.setup(optimiser, ps))
    recipe = (;
        makeworker = (idx, manager) -> averaged_worker(layer, worker_graph(graph, manager.state.params[], manager.config), manager.config),
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.base_context.x .= job.x
            ctx.base_context.y .= job.y
            ctx.repeats[] = job.repeats
            Processes.resetworker!(slot)
            return nothing
        end,
        flush! = manager -> flush_cnn_buffers!(manager),
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = LocalCNNXORJob{Vector{FT},Vector{FT}},
    )
end

"""Create manager jobs, defaulting to enough chunks to fill the workers.

Each job stores the number of random-start repeats to run. The worker-side
`AveragedCNNContrastiveStep` executes those repeats inside one
`ProcessAlgorithm` step, so a 32-worker run gets 32 scheduled jobs without
turning each random init into its own manager job.
"""
function cnn_xor_jobs(config::C, x::X, y::Y) where {C<:LocalCNNXORConfig,X<:AbstractMatrix,Y<:AbstractMatrix}
    chunks_per_case = config.chunks_per_case > 0 ?
        max(1, min(config.repeats_per_case, config.chunks_per_case)) :
        max(1, min(config.repeats_per_case, cld(config.workers, length(CNN_XOR_CASES))))
    jobs = LocalCNNXORJob{Vector{FT},Vector{FT}}[]
    for case_idx in axes(x, 2)
        base = config.repeats_per_case ÷ chunks_per_case
        extra = config.repeats_per_case % chunks_per_case
        for chunk in 1:chunks_per_case
            repeats = base + (chunk <= extra ? 1 : 0)
            repeats == 0 && continue
            push!(jobs, LocalCNNXORJob(case_idx, copy(view(x, :, case_idx)), copy(view(y, :, case_idx)), repeats))
        end
    end
    return jobs
end

"""Run one full XOR minibatch and synchronize updated parameters."""
function run_cnn_batch!(manager::M, jobs::J, config::C, update_idx::Integer) where {M<:Processes.ProcessManager,J,C<:LocalCNNXORConfig}
    clear_manager_buffers!(manager)
    manager.state.total_repeats[] = sum(job.repeats for job in jobs)
    Processes.run!(manager, jobs, Processes.Dynamic())
    add_weight_decay!(manager.state.batch_gradient, manager.state.params[], config.weight_decay)
    gradient_norm = parameter_norm(manager.state.batch_gradient)
    old_params = manager.state.params[]
    current_lr = adjust_learning_rate!(manager, config, update_idx)
    manager.state.opt_state, ps_new = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = ps_new
    sync_worker_params!(manager, ps_new)
    return (;
        ps = ps_new,
        gradient_norm,
        parameter_norm = parameter_norm(ps_new),
        update_norm = parameter_delta_norm(old_params, ps_new),
        learning_rate = current_lr,
    )
end

"""Return the L2 norm of a Lux parameter or gradient tree used here."""
function parameter_norm(ps::P) where {P}
    total = sum(abs2, ps.w) + sum(abs2, ps.b)
    hasproperty(ps, :α) && (total += sum(abs2, ps.α))
    return sqrt(Float64(total))
end

"""Return the L2 norm of the update between two parameter trees."""
function parameter_delta_norm(old::O, new::N) where {O,N}
    total = sum(abs2, new.w .- old.w) + sum(abs2, new.b .- old.b)
    hasproperty(new, :α) && (total += sum(abs2, new.α .- old.α))
    return sqrt(Float64(total))
end

"""Compute scalar scores from output states for the configured readout."""
function output_scores(outputs::O, config::C) where {O<:AbstractMatrix,C<:LocalCNNXORConfig}
    if config.output_mode === :majority
        return vec(mean(outputs; dims = 1))
    elseif config.output_mode === :pattern
        false_pattern, true_pattern = output_pattern_targets(config, eltype(outputs))
        scores = Vector{eltype(outputs)}(undef, size(outputs, 2))
        for case_idx in axes(outputs, 2)
            out = view(outputs, :, case_idx)
            true_score = -mean(abs2, out .- true_pattern)
            false_score = -mean(abs2, out .- false_pattern)
            scores[case_idx] = true_score - false_score
        end
        return scores
    elseif config.output_mode === :two_class
        split = size(outputs, 1) ÷ 2
        false_scores = vec(mean(view(outputs, 1:split, :); dims = 1))
        true_scores = vec(mean(view(outputs, (split + 1):size(outputs, 1), :); dims = 1))
        return true_scores .- false_scores
    else
        throw(ArgumentError("unknown XOR CNN output mode `$(config.output_mode)`; use `majority`, `pattern`, or `two_class`"))
    end
end

"""Run one validation relaxation and return the configured output view."""
function cnn_forward_output!(
    layer::L,
    graph::G,
    algorithm::A,
    context::K,
    x::X,
    ps::P,
    config::C,
) where {L<:LayeredIsingGraphLayer,G,A,K,X<:AbstractVector,P,C<:LocalCNNXORConfig}
    IsingLearning.sync_params!(graph, ps)
    initialize_cnn_state!(graph, config.init_mode)
    IsingLearning.apply_input(graph, x)
    IsingLearning.relax_context!(algorithm, context, layer.free_relaxation_steps)
    return IsingLearning.graph_view(graph, layer.output_layer)
end

"""Evaluate readout score, physical-output MSE, margins, and accuracy."""
function evaluate_cnn_xor(layer::L, ps::P, st::S, x::X, y::Y, config::C) where {L<:LayeredIsingGraphLayer,P,S,X<:AbstractMatrix,Y<:AbstractMatrix,C<:LocalCNNXORConfig}
    outputs = zeros(FT, size(y, 1), size(y, 2))
    graph = st.graph
    algorithm = layer.validation_algorithm
    context = Processes.init(algorithm, (; model = graph))
    for _ in 1:config.eval_repeats
        for case_idx in axes(x, 2)
            outputs[:, case_idx] .+= cnn_forward_output!(
                layer,
                graph,
                algorithm,
                context,
                view(x, :, case_idx),
                ps,
                config,
            )
        end
    end
    outputs ./= FT(config.eval_repeats)

    scores = output_scores(outputs, config)
    targets = FT[xor(a, b) ? 1 : -1 for (a, b) in CNN_XOR_CASES]
    predictions = scores .> zero(FT)
    target_bits = targets .> zero(FT)
    margins = targets .* scores
    per_case_mse = vec(mean(abs2, outputs .- y; dims = 1))
    return (;
        mse = mean(abs2, scores .- targets),
        output_mse = mean(abs2, outputs .- y),
        accuracy = mean(predictions .== target_bits),
        all_correct = all(predictions .== target_bits),
        min_margin = minimum(margins),
        mean_margin = mean(margins),
        margins = copy(margins),
        per_case_mse = copy(per_case_mse),
        scores = copy(scores),
        predictions = Int.(predictions),
        outputs = copy(outputs),
    )
end

"""Append one named-tuple row to a simple CSV file."""
function append_metrics!(path::P, row) where {P<:AbstractString}
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        names = propertynames(row)
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return the `8x8 -> 4x4` architecture grid over layer-1 and layer-2 radii."""
function experiment_configs(base::C) where {C<:LocalCNNXORConfig}
    layer1_radii = parse_int_list(get(ENV, "ISING_XOR_CNN_LAYER1_RADII", "1,2,3,4,5"), collect(1:5))
    layer2_radii = parse_int_list(get(ENV, "ISING_XOR_CNN_LAYER2_RADII", "1,2"), collect(1:2))
    periodic_values = parse_bool_list(get(ENV, "ISING_XOR_CNN_PERIODIC", "false"), [false])
    limit = parse(Int, get(ENV, "ISING_XOR_CNN_LIMIT", "0"))

    configs = LocalCNNXORConfig[]
    for periodic in periodic_values, layer1_radius in layer1_radii, layer2_radius in layer2_radii
        name = periodic ? "r1_$(layer1_radius)_r2_$(layer2_radius)_periodic" : "r1_$(layer1_radius)_r2_$(layer2_radius)"
        seed = base.seed + 10_000 * layer1_radius + 101 * layer2_radius + (periodic ? 1_000_000 : 0)
        push!(
            configs,
            copy_config(
                base;
                name,
                hidden1_side = 8,
                hidden2_side = 4,
                layer1_radius,
                layer2_radius,
                hidden_periodic = periodic,
                seed,
            ),
        )
    end
    if limit > 0
        resize!(configs, min(limit, length(configs)))
    end
    return configs
end

"""Return the learning-summary plot path inside the checkerboard run folder."""
function checkerboard_summary_plot_path(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    return joinpath(outdir, "learning_summary.png")
end

"""Save a checkerboard result plot as the requested PNG file."""
function save_checkerboard_plot(path::P, fig) where {P<:AbstractString}
    save(path, fig)
    return path
end

"""Return line style and width for a layer-1 radius group."""
function checkerboard_line_style(layer1_radius::Integer)
    layer1_radius <= 2 && return (:solid, 2.0)
    layer1_radius <= 4 && return (:dash, 2.8)
    return (:dot, 3.6)
end

"""Plot aggregate learning curves, accuracy, margins, and best-margin comparison."""
function plot_results(outdir::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    isempty(rows) && return nothing

    fig = Figure(size = (1500, 950))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "score MSE", title = "Learning curves")
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_margin = Axis(fig[3, 1], xlabel = "epoch", ylabel = "min margin", title = "Worst-case margin")
    ax_best = Axis(fig[1:3, 2], xlabel = "configuration", ylabel = "best min margin", title = "Best Worst-Case Margin")

    configs = unique(row.config for row in rows)
    palette = Makie.wong_colors()
    for (idx, config_name) in enumerate(configs)
        subset = [row for row in rows if row.config == config_name]
        color = palette[mod1(idx, length(palette))]
        layer1_radius = isempty(subset) ? 0 : subset[1].layer1_radius
        style, width = checkerboard_line_style(layer1_radius)
        lines!(ax_mse, [row.epoch for row in subset], [row.mse for row in subset], color = color, label = config_name, linestyle = style, linewidth = width)
        lines!(ax_acc, [row.epoch for row in subset], [row.accuracy for row in subset], color = color, linestyle = style, linewidth = width)
        lines!(ax_margin, [row.epoch for row in subset], [row.min_margin for row in subset], color = color, linestyle = style, linewidth = width)
    end

    sorted = sort(summary_rows; by = row -> row.best_min_margin, rev = true)
    xvals = 1:length(sorted)
    barplot!(ax_best, xvals, [row.best_min_margin for row in sorted], color = :steelblue)
    ax_best.xticks = (xvals, [row.config for row in sorted])
    ax_best.xticklabelrotation = pi / 3
    axislegend(ax_mse, position = :rt, nbanks = 2)

    path = checkerboard_summary_plot_path(outdir)
    save_checkerboard_plot(path, fig)
    return path
end

"""Plot the learning history for one `r1/r2` configuration inside its run folder."""
function plot_config_results(run_dir::P, rows::R, config::C) where {P<:AbstractString,R<:AbstractVector,C<:LocalCNNXORConfig}
    isempty(rows) && return nothing

    fig = Figure(size = (1200, 850))
    title = "$(config.name): r1=$(config.layer1_radius), r2=$(config.layer2_radius)"
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "score MSE", title = title)
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "Truth-table accuracy")
    ax_margin = Axis(fig[3, 1], xlabel = "epoch", ylabel = "min margin", title = "Worst-case margin")

    epochs = [row.epoch for row in rows]
    lines!(ax_mse, epochs, [row.mse for row in rows], color = :steelblue, linewidth = 2.6)
    lines!(ax_acc, epochs, [row.accuracy for row in rows], color = :seagreen, linewidth = 2.6)
    lines!(ax_margin, epochs, [row.min_margin for row in rows], color = :firebrick, linewidth = 2.6)
    hlines!(ax_acc, [1.0], color = (:gray30, 0.45), linestyle = :dash)
    hlines!(ax_margin, [0.0], color = (:gray30, 0.45), linestyle = :dash)

    path = joinpath(run_dir, "learning_curve.png")
    save_checkerboard_plot(path, fig)
    return path
end

"""Write one serialized snapshot for later inspection or restart."""
function save_snapshot!(run_dir::P, epoch::Integer, ps::S, config::C, metrics::M) where {P<:AbstractString,S,C<:LocalCNNXORConfig,M}
    snapshot_dir = joinpath(run_dir, "snapshots")
    mkpath(snapshot_dir)
    path = joinpath(snapshot_dir, "epoch_$(lpad(string(epoch), 6, '0')).bin")
    open(path, "w") do io
        serialize(io, (; epoch = Int(epoch), ps, config, metrics, cases = CNN_XOR_CASES))
    end
    return path
end

"""Load initial parameters for a config from a previous checkerboard run when requested."""
function maybe_resume_params(ps::P, config::C) where {P,C<:LocalCNNXORConfig}
    resume_dir = get(ENV, "ISING_XOR_CNN_RESUME_DIR", "")
    isempty(resume_dir) && return ps
    resume_file = get(ENV, "ISING_XOR_CNN_RESUME_FILE", "best_margin_params.bin")
    path = joinpath(resume_dir, config.name, resume_file)
    isfile(path) || return ps
    payload = open(deserialize, path)
    hasproperty(payload, :ps) || throw(ArgumentError("resume file `$path` does not contain a `ps` field"))
    return payload.ps
end

"""Run one configured `r1/r2` architecture variant and return metric rows."""
function run_config!(config::C, root_outdir::P) where {C<:LocalCNNXORConfig,P<:AbstractString}
    run_dir = joinpath(root_outdir, config.name)
    isdir(run_dir) && rm(run_dir; recursive = true, force = true)
    mkpath(run_dir)

    graph = cnn_xor_graph(config)
    layer = cnn_xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 20), layer)
    ps = maybe_resume_params(ps, config)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 21), layer)
    x, y = cnn_xor_dataset(config, FT)
    jobs = cnn_xor_jobs(config, x, y)
    manager = cnn_xor_manager(layer, graph, ps, config)

    metrics_path = joinpath(run_dir, "metrics.csv")
    isfile(metrics_path) && rm(metrics_path)
    best_mse = Inf
    best_epoch = 0
    best_accuracy = 0.0
    best_min_margin = -Inf
    best_margin_epoch = 0
    first_all_correct_epoch = -1
    rows = NamedTuple[]
    started = time()
    last_update = (; gradient_norm = NaN, parameter_norm = parameter_norm(manager.state.params[]), update_norm = NaN, learning_rate = config.lr)

    try
        for epoch in 0:config.epochs
            should_snapshot = config.snapshot_every > 0 && epoch % config.snapshot_every == 0
            should_record = epoch == 0 || epoch % config.log_every == 0 || should_snapshot || epoch == config.epochs
            if should_record
                metrics = evaluate_cnn_xor(layer, manager.state.params[], st, x, y, config)
                if metrics.mse < best_mse
                    best_mse = metrics.mse
                    best_epoch = epoch
                    best_accuracy = metrics.accuracy
                    open(joinpath(run_dir, "best_params.bin"), "w") do io
                        serialize(io, (; ps = manager.state.params[], config, metrics, cases = CNN_XOR_CASES))
                    end
                end
                if metrics.min_margin > best_min_margin
                    best_min_margin = metrics.min_margin
                    best_margin_epoch = epoch
                    open(joinpath(run_dir, "best_margin_params.bin"), "w") do io
                        serialize(io, (; ps = manager.state.params[], config, metrics, cases = CNN_XOR_CASES))
                    end
                end
                if metrics.all_correct && first_all_correct_epoch < 0
                    first_all_correct_epoch = epoch
                end
                should_snapshot && save_snapshot!(run_dir, epoch, manager.state.params[], config, metrics)
                row = (;
                    config = config.name,
                    input_side = config.input_side,
                    hidden1_side = config.hidden1_side,
                    hidden2_side = config.hidden2_side,
                    output_side = config.output_side,
                    layer1_radius = config.layer1_radius,
                    layer2_radius = config.layer2_radius,
                    periodic = config.hidden_periodic,
                    epoch,
                    mse = metrics.mse,
                    output_mse = metrics.output_mse,
                    accuracy = metrics.accuracy,
                    all_correct = metrics.all_correct,
                    min_margin = metrics.min_margin,
                    mean_margin = metrics.mean_margin,
                    best_mse,
                    best_epoch,
                    best_min_margin,
                    best_margin_epoch,
                    first_all_correct_epoch,
                    gradient_norm = last_update.gradient_norm,
                    parameter_norm = last_update.parameter_norm,
                    update_norm = last_update.update_norm,
                    learning_rate = last_update.learning_rate,
                    elapsed_seconds = round(time() - started; digits = 3),
                    predictions = join(metrics.predictions, "|"),
                    scores = join(round.(metrics.scores; digits = 4), "|"),
                    margins = join(round.(metrics.margins; digits = 4), "|"),
                    per_case_mse = join(round.(metrics.per_case_mse; digits = 4), "|"),
                )
                push!(rows, row)
                append_metrics!(metrics_path, row)
                println(
                    config.name,
                    " epoch=", epoch,
                    " mse=", round(metrics.mse; digits = 6),
                    " acc=", round(metrics.accuracy; digits = 3),
                    " min_margin=", round(metrics.min_margin; digits = 4),
                    " best_margin=", round(best_min_margin; digits = 4),
                )
            end
            epoch == config.epochs && break
            last_update = run_cnn_batch!(manager, jobs, config, epoch + 1)
        end
    finally
        close(manager)
    end

    open(joinpath(run_dir, "final_params.bin"), "w") do io
        serialize(io, (; ps = manager.state.params[], config, cases = CNN_XOR_CASES))
    end
    config.plot && plot_config_results(run_dir, rows, config)
    return (;
        rows,
        summary = (;
            config = config.name,
            input_side = config.input_side,
            hidden1_side = config.hidden1_side,
            hidden2_side = config.hidden2_side,
            output_side = config.output_side,
            layer1_radius = config.layer1_radius,
            layer2_radius = config.layer2_radius,
            periodic = config.hidden_periodic,
            best_mse,
            best_accuracy,
            best_epoch,
            best_min_margin,
            best_margin_epoch,
            first_all_correct_epoch,
            run_dir,
        ),
    )
end

"""Write the aggregate CSV files for the grid."""
function write_aggregate_csvs!(outdir::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    metrics_path = joinpath(outdir, "metrics.csv")
    summary_path = joinpath(outdir, "summary.csv")
    for row in rows
        append_metrics!(metrics_path, row)
    end
    for row in summary_rows
        append_metrics!(summary_path, row)
    end
    return (; metrics_path, summary_path)
end

"""Run the `8x8 -> 8x8 -> 4x4 -> 4x4` checkerboard radius-grid experiment."""
function main()
    base = LocalCNNXORConfig()
    base.workers > 0 || throw(ArgumentError("ISING_XOR_CNN_WORKERS must be positive"))
    base.repeats_per_case > 0 || throw(ArgumentError("ISING_XOR_CNN_REPEATS must be positive"))
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers
    base.output_side > 0 || throw(ArgumentError("output_side must be positive"))

    configs = experiment_configs(base)
    isempty(configs) && throw(ArgumentError("no local CNN XOR configs selected"))
    mkpath(base.outdir)
    for filename in ("metrics.csv", "summary.csv")
        path = joinpath(base.outdir, filename)
        isfile(path) && rm(path)
    end

    all_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    println("Running $(length(configs)) checkerboard 8x8-to-4x4 CNN-like XOR config(s)")
    for (idx, config) in enumerate(configs)
        println("[$idx/$(length(configs))] $(config.name)")
        result = run_config!(config, base.outdir)
        append!(all_rows, result.rows)
        push!(summary_rows, result.summary)
    end

    csvs = write_aggregate_csvs!(base.outdir, all_rows, summary_rows)
    plot_path = base.plot ? plot_results(base.outdir, all_rows, summary_rows) : nothing
    println("saved metrics ", csvs.metrics_path)
    println("saved summary ", csvs.summary_path)
    !isnothing(plot_path) && println("saved plot ", plot_path)
    return (; outdir = base.outdir, rows = all_rows, summary_rows, plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
