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
const StatefulAlgorithms = II.StatefulAlgorithms
const FT = Float32

const EDGE_XOR_CASES = ((false, false), (false, true), (true, false), (true, true))
const EDGE_HIDDEN_SCALE_REF = Ref(FT(0.04))
const EDGE_OUTPUT_SCALE_REF = Ref(FT(0.06))
const EDGE_HIDDEN_RNG_REF = Ref(Random.MersenneTwister(1))

Base.@kwdef struct EdgeApplicationConfig
    name::String = "nn3"
    seed_index::Int = 1
    epochs::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_EPOCHS", "160"))
    workers::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_WORKERS", "32"))
    log_every::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_LOG_EVERY", "10"))
    snapshot_every::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_SNAPSHOT_EVERY", "40"))
    plot::Bool = parse(Bool, lowercase(get(ENV, "ISING_XOR_EDGE_PLOT", "true")))
    repeats_per_case::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_REPEATS", "64"))
    chunks_per_case::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_CHUNKS_PER_CASE", "0"))
    eval_repeats::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_EVAL_REPEATS", "64"))
    side::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_SIDE", "16"))
    output_mode::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_EDGE_OUTPUT_MODE", "two_class")))
    local_nn::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_NN", "3"))
    edge_fanout::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_FANOUT", "1"))
    free_sweeps::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_FREE_SWEEPS", "20"))
    nudged_sweeps::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_NUDGED_SWEEPS", "20"))
    β::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_BETA", "1.0"))
    lr::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_LR", "0.002"))
    lr_decay::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_LR_DECAY", "0.995"))
    lr_min::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_LR_MIN", "0.0002"))
    weight_decay::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_WEIGHT_DECAY", "0.0001"))
    optimizer::String = lowercase(get(ENV, "ISING_XOR_EDGE_OPTIMIZER", "adam"))
    dynamics::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_EDGE_DYNAMICS", "block")))
    block_size::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_BLOCK_SIZE", "8"))
    temp::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_TEMP", "0.001"))
    stepsize::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_STEPSIZE", "0.1"))
    max_drift_fraction::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_MAX_DRIFT", "1.0"))
    input_scale::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_INPUT_SCALE", "0.4"))
    hidden_internal_scale::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_HIDDEN_SCALE", "0.2"))
    output_scale::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_OUTPUT_SCALE", "0.4"))
    output_internal_scale::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_OUTPUT_INTERNAL_SCALE", "0.0"))
    bias_scale::FT = parse(FT, get(ENV, "ISING_XOR_EDGE_BIAS_SCALE", "0.05"))
    init_mode::Symbol = Symbol(lowercase(get(ENV, "ISING_XOR_EDGE_INIT_MODE", "zero")))
    seed::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_SEED", "7103"))
    resume_dir::String = get(ENV, "ISING_XOR_EDGE_RESUME_DIR", "")
    resume_file::String = get(ENV, "ISING_XOR_EDGE_RESUME_FILE", "final_params.bin")
    start_epoch::Int = parse(Int, get(ENV, "ISING_XOR_EDGE_START_EPOCH", "0"))
    outdir::String = get(
        ENV,
        "ISING_XOR_EDGE_OUTDIR",
        joinpath(
            @__DIR__,
            "experiments",
            "current",
            "separate_input_output_lines_side16_nn1to10_seeds1to5_e160",
        ),
    )
end

struct EdgeApplicationJob{X<:AbstractVector,Y<:AbstractVector}
    case_idx::Int
    x::X
    y::Y
    repeats::Int
end

mutable struct EdgeManagerState{P,B,O}
    params::Base.RefValue{P}
    batch_gradient::B
    total_repeats::Base.RefValue{Int}
    opt_state::O
end

"""Return the number of graph spins used to convert sweeps into process steps."""
function edge_nunits(config::C) where {C<:EdgeApplicationConfig}
    return 2 * config.side + config.side^2
end

"""Parse a comma-separated integer list, returning `default` if the value is empty."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Return a fixed-width seed suffix for readable run folder names."""
function seed_label(seed_index::Integer)
    return "seed" * lpad(string(seed_index), 2, '0')
end

"""Return a copy of `config` with selected fields replaced."""
function copy_config(config::C; kwargs...) where {C<:EdgeApplicationConfig}
    fields = Dict{Symbol,Any}(name => getfield(config, name) for name in fieldnames(C))
    for (key, value) in kwargs
        fields[key] = value
    end
    return EdgeApplicationConfig(; fields...)
end

"""Create a square topology with an explicit world-coordinate origin."""
function edge_topology(size::Tuple{Int,Int}; origin::Tuple{Float64,Float64}, periodic::Bool = false)
    return II.SquareTopology(size; origin, periodic)
end

"""Return true when two world coordinates are within a Chebyshev edge fanout."""
function inside_edge_fanout(c1::C1, c2::C2, fanout::Integer) where {C1,C2}
    return max(abs(c1[1] - c2[1]), abs(c1[2] - c2[2])) <= Float64(fanout) + 1e-6
end

"""Named callback for trainable local hidden-layer couplings."""
function edge_hidden_internal_weight(; dr, c1, c2, dc)
    return EDGE_HIDDEN_SCALE_REF[] * randn(EDGE_HIDDEN_RNG_REF[], FT)
end

"""Named callback for ferromagnetic output-line vote coupling."""
function edge_output_internal_weight(; dr, c1, c2, dc)
    return EDGE_OUTPUT_SCALE_REF[]
end

"""Build the hidden-layer recurrent local `WeightGenerator`."""
function edge_hidden_generator(config::C) where {C<:EdgeApplicationConfig}
    EDGE_HIDDEN_SCALE_REF[] = config.hidden_internal_scale
    EDGE_HIDDEN_RNG_REF[] = Random.MersenneTwister(config.seed + 1)
    return II.WeightGenerator(edge_hidden_internal_weight, config.local_nn, EDGE_HIDDEN_RNG_REF[]; symmetric = true)
end

"""Build the output-line coupling generator used for majority-vote coherence."""
function edge_output_internal_generator(config::C) where {C<:EdgeApplicationConfig}
    EDGE_OUTPUT_SCALE_REF[] = config.output_internal_scale
    return II.WeightGenerator(edge_output_internal_weight, 1, Random.MersenneTwister(config.seed + 2); symmetric = true)
end

"""Connect the checkerboard input line to the left edge of the 16x16 layer."""
function edge_input_generator(config::C) where {C<:EdgeApplicationConfig}
    rng = Random.MersenneTwister(config.seed + 3)
    fanout = config.edge_fanout
    scale = config.input_scale
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> inside_edge_fanout(c1, c2, fanout) ? scale * abs(randn(rng, FT)) : zero(FT),
        rng,
    )
end

"""Connect the right edge of the 16x16 layer to the majority-vote output line."""
function edge_readout_generator(config::C) where {C<:EdgeApplicationConfig}
    rng = Random.MersenneTwister(config.seed + 4)
    fanout = config.edge_fanout
    scale = config.output_scale
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> inside_edge_fanout(c1, c2, fanout) ? scale * randn(rng, FT) : zero(FT),
        rng,
    )
end

"""Write the configured output-line target for one XOR label."""
function write_edge_target!(dest::D, config::C, label::Bool) where {D<:AbstractVector,C<:EdgeApplicationConfig}
    if config.output_mode === :majority
        dest .= label ? one(eltype(dest)) : -one(eltype(dest))
    elseif config.output_mode === :two_class
        iseven(length(dest)) || throw(ArgumentError("two_class edge output needs an even output-line length"))
        split = length(dest) ÷ 2
        dest[1:split] .= label ? -one(eltype(dest)) : one(eltype(dest))
        dest[(split + 1):end] .= label ? one(eltype(dest)) : -one(eltype(dest))
    else
        throw(ArgumentError("unknown edge output mode `$(config.output_mode)`; use `majority` or `two_class`"))
    end
    return dest
end

"""Return static checkerboard edge inputs and configured output-line targets."""
function edge_xor_dataset(config::C, ::Type{T} = FT) where {C<:EdgeApplicationConfig,T<:AbstractFloat}
    x = Matrix{T}(undef, config.side, length(EDGE_XOR_CASES))
    y = Matrix{T}(undef, config.side, length(EDGE_XOR_CASES))
    for (case_idx, (a, b)) in enumerate(EDGE_XOR_CASES)
        a_value = a ? one(T) : -one(T)
        b_value = b ? one(T) : -one(T)
        for row in 1:config.side
            x[row, case_idx] = isodd(row) ? a_value : b_value
        end
        write_edge_target!(view(y, :, case_idx), config, xor(a, b))
    end
    return x, y
end

"""Build `input edge -> 16x16 propagation layer -> output edge` graph."""
function edge_xor_graph(config::C) where {C<:EdgeApplicationConfig}
    side = config.side
    rng_b = Random.MersenneTwister(config.seed + 5)

    input = II.Layer(
        side,
        1,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        edge_topology((side, 1); origin = (0.0, 0.0), periodic = false),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    hidden = II.Layer(
        side,
        side,
        II.StateSet(-one(FT), one(FT)),
        edge_hidden_generator(config),
        II.Continuous(),
        edge_topology((side, side); origin = (0.0, 0.0), periodic = false),
        II.Coords(0, 80, 0);
        periodic = false,
    )
    output = II.Layer(
        side,
        1,
        II.StateSet(-one(FT), one(FT)),
        edge_output_internal_generator(config),
        II.Continuous(),
        edge_topology((side, 1); origin = (0.0, Float64(side - 1)), periodic = false),
        II.Coords(0, 260, 0);
        periodic = false,
    )

    bias = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    graph = II.IsingGraph(
        input,
        edge_input_generator(config),
        hidden,
        edge_readout_generator(config),
        output,
        II.Bilinear() + II.MagField(b = bias) + II.Clamping(β = II.UniformArray(zero(FT)), y = target, mask = mask);
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Create the sampler used for free, nudged, and validation trajectories."""
function edge_xor_dynamics(config::C) where {C<:EdgeApplicationConfig}
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
    throw(ArgumentError("unknown edge dynamics `$(config.dynamics)`; use `local` or `block`"))
end

"""Initialize a graph state according to the configured free-phase basin."""
function initialize_edge_state!(model::G, init_mode::I) where {G,I}
    if init_mode === :random
        II.resetstate!(model)
    elseif init_mode === :zero
        fill!(II.state(model), zero(eltype(model)))
    else
        throw(ArgumentError("unknown edge init mode `$(init_mode)`; use `random` or `zero`"))
    end
    return model
end

"""Wrap the edge graph in the standard contrastive Learning layer."""
function edge_xor_layer(graph::G, config::C) where {G,C<:EdgeApplicationConfig}
    dynamics = edge_xor_dynamics(config)
    n_free = config.free_sweeps * II.nstates(graph)
    n_nudged = config.nudged_sweeps * II.nstates(graph)
    return LayeredIsingGraphLayer(
        graph;
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = n_free,
        free_relaxation_steps = n_free,
        nudged_relaxation_steps = n_nudged,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Accumulate one plus/minus contrastive gradient into a reusable buffer."""
function accumulate_layer_contrastive_gradient!(
    isinggraph::G,
    plus_state::P,
    minus_state::M,
    beta::T,
    buffers::B,
) where {G,P<:AbstractVector,M<:AbstractVector,T<:Real,B}
    IsingLearning.contrastive_gradient(isinggraph, plus_state, minus_state, beta; buffers)
    return buffers
end

"""Build one reusable free-phase routine for an edge XOR worker."""
function edge_free_phase_algorithm(dynamics_algorithm::D, init_mode, steps::I) where {D,I<:Integer}
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @state x
        @state equilibrium_state

        initialize_edge_state!(dynamics.model, init_mode)
        IsingLearning.apply_input(dynamics.model, x)
        @repeat steps dynamics()
        CopyGraphState!(equilibrium_state, dynamics.model)
    end
end

"""Build one reusable positive-nudge phase routine for an edge XOR worker."""
function edge_plus_phase_algorithm(nudged_dynamics_algorithm::D, beta::T, steps::I) where {D,T<:Real,I<:Integer}
    return StatefulAlgorithms.@Routine begin
        @alias nudged_dynamics = nudged_dynamics_algorithm
        @state x
        @state y
        @state equilibrium_state
        @state plus_state

        SetGraphState!(nudged_dynamics.model, equilibrium_state)
        IsingLearning.apply_input(nudged_dynamics.model, x)
        IsingLearning.apply_targets(nudged_dynamics.model, y)
        IsingLearning.set_clamping_beta!(nudged_dynamics.model, beta)
        @repeat steps nudged_dynamics()
        CopyGraphState!(plus_state, nudged_dynamics.model)
    end
end

"""Build one reusable negative-nudge phase routine for an edge XOR worker."""
function edge_minus_phase_algorithm(nudged_dynamics_algorithm::D, beta::T, steps::I) where {D,T<:Real,I<:Integer}
    return StatefulAlgorithms.@Routine begin
        @alias nudged_dynamics = nudged_dynamics_algorithm
        @state x
        @state y
        @state equilibrium_state
        @state minus_state

        SetGraphState!(nudged_dynamics.model, equilibrium_state)
        IsingLearning.apply_input(nudged_dynamics.model, x)
        IsingLearning.apply_targets(nudged_dynamics.model, y)
        IsingLearning.set_clamping_beta!(nudged_dynamics.model, -beta)
        @repeat steps nudged_dynamics()
        CopyGraphState!(minus_state, nudged_dynamics.model)
    end
end

"""Build one reusable contrastive-sample routine for an edge XOR worker."""
function edge_sample_algorithm(
    free_phase,
    plus_phase,
    minus_phase,
    beta::T,
) where {T<:Real}
    return StatefulAlgorithms.@Routine begin
        @alias free_phase = free_phase
        @alias plus_phase = plus_phase
        @alias minus_phase = minus_phase
        @state buffers
        @state plus_state
        @state minus_state

        free_phase()
        plus_phase()
        minus_phase()
        IsingLearning.set_clamping_beta!(minus_phase.nudged_dynamics.model, zero(beta))
        accumulate_layer_contrastive_gradient!(minus_phase.nudged_dynamics.model, plus_state, minus_state, beta, buffers)
    end
end

"""Build the reusable LoopAlgorithm used by each edge XOR worker."""
function edge_worker_algorithm(layer::L, config::C) where {L<:LayeredIsingGraphLayer,C<:EdgeApplicationConfig}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    nudged_dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    free_steps = layer.free_relaxation_steps
    nudged_steps = layer.nudged_relaxation_steps
    β = layer.β
    init_mode = config.init_mode
    free_phase = edge_free_phase_algorithm(dynamics_algorithm, init_mode, free_steps)
    plus_phase = edge_plus_phase_algorithm(nudged_dynamics_algorithm, β, nudged_steps)
    minus_phase = edge_minus_phase_algorithm(nudged_dynamics_algorithm, β, nudged_steps)
    sample_once = edge_sample_algorithm(free_phase, plus_phase, minus_phase, β)
    return StatefulAlgorithms.@Routine begin
        @alias sample_once = sample_once
        @state repeats

        @repeat repeats[] sample_once()
    end
end

"""Return the mutable training context stored in one worker."""
function worker_context(worker::W) where {W}
    return StatefulAlgorithms.context(worker)._state
end

"""Create a worker graph with fresh state and shared static parameter arrays."""
function worker_graph(prototype::G, ps::P, config::C) where {G,P,C<:EdgeApplicationConfig}
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

"""Build one reusable LoopAlgorithm process worker."""
function edge_worker(layer::L, graph::G, config::C) where {L<:LayeredIsingGraphLayer,G,C<:EdgeApplicationConfig}
    graph_state = II.state(graph)
    algorithm = StatefulAlgorithms.resolve(edge_worker_algorithm(layer, config))
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            buffers = IsingLearning.layer_gradient_buffer(graph),
            equilibrium_state = similar(graph_state),
            plus_state = similar(graph_state),
            minus_state = similar(graph_state),
            repeats = Ref(1),
        ),
        StatefulAlgorithms.Init(:dynamics; model = graph),
        StatefulAlgorithms.Init(:nudged_dynamics; model = graph);
        repeat = 1,
    )
end

"""Allocate a gradient buffer matching the parameter tree."""
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

"""Scale every array in a gradient buffer."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Clear manager and worker-local buffers before a minibatch."""
function clear_manager_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in StatefulAlgorithms.workers(manager)
        clear_buffer!(worker_context(worker).buffers)
    end
    return manager
end

"""Flush worker-local buffers into one averaged gradient."""
function flush_edge_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in StatefulAlgorithms.workers(manager)
        ctx = worker_context(worker)
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
function learning_rate_at(config::C, update_idx::Integer) where {C<:EdgeApplicationConfig}
    exponent = max(Int(update_idx) - 1, 0)
    scheduled = config.lr * config.lr_decay^exponent
    return max(config.lr_min, scheduled)
end

"""Adjust the Optimisers.jl state in place while preserving Adam moments."""
function adjust_learning_rate!(manager::M, config::C, update_idx::Integer) where {M<:StatefulAlgorithms.ProcessManager,C<:EdgeApplicationConfig}
    η = learning_rate_at(config, update_idx)
    Optimisers.adjust!(manager.state.opt_state, η)
    return η
end

"""Install updated shared parameters once after a manager batch."""
function sync_worker_params!(manager::M, ps::P) where {M<:StatefulAlgorithms.ProcessManager,P}
    worker = first(StatefulAlgorithms.workers(manager))
    IsingLearning.sync_params!(StatefulAlgorithms.context(worker).dynamics.model, ps)
    return manager
end

"""Create the `ProcessManager` that owns reusable edge-application workers."""
function edge_xor_manager(layer::L, graph::G, ps::P, config::C) where {L<:LayeredIsingGraphLayer,G,P,C<:EdgeApplicationConfig}
    optimizer_name = lowercase(config.optimizer)
    optimiser = optimizer_name == "adam" ? Optimisers.Adam(config.lr) :
        optimizer_name in ("descent", "sgd") ? Optimisers.Descent(config.lr) :
        throw(ArgumentError("unknown XOR optimizer `$(config.optimizer)`; use `adam` or `descent`"))
    state = EdgeManagerState(Ref(ps), parameter_buffer(ps), Ref(0), Optimisers.setup(optimiser, ps))
    recipe = (;
        makeworker = (idx, manager) -> edge_worker(layer, worker_graph(graph, manager.state.params[], manager.config), manager.config),
        loadjob! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x .= job.x
            ctx.y .= job.y
            ctx.repeats[] = job.repeats
            StatefulAlgorithms.resetworker!(slot)
            return nothing
        end,
        sync_to_state! = manager -> flush_edge_buffers!(manager),
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        sync_policy = StatefulAlgorithms.SyncAtEnd(),
        execution = StatefulAlgorithms.ChannelWorkers(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = EdgeApplicationJob{Vector{FT},Vector{FT}},
    )
end

"""Create manager jobs, defaulting to enough chunks to fill the workers.

Each job stores the number of random-start repeats to run. The worker-side
LoopAlgorithm executes those repeats inside one scheduled manager job, so a
32-worker run gets 32 scheduled jobs without turning each random init into
its own manager job.
"""
function edge_xor_jobs(config::C, x::X, y::Y) where {C<:EdgeApplicationConfig,X<:AbstractMatrix,Y<:AbstractMatrix}
    chunks_per_case = config.chunks_per_case > 0 ?
        max(1, min(config.repeats_per_case, config.chunks_per_case)) :
        max(1, min(config.repeats_per_case, cld(config.workers, length(EDGE_XOR_CASES))))
    jobs = EdgeApplicationJob{Vector{FT},Vector{FT}}[]
    for case_idx in axes(x, 2)
        base = config.repeats_per_case ÷ chunks_per_case
        extra = config.repeats_per_case % chunks_per_case
        for chunk in 1:chunks_per_case
            repeats = base + (chunk <= extra ? 1 : 0)
            repeats == 0 && continue
            push!(jobs, EdgeApplicationJob(case_idx, copy(view(x, :, case_idx)), copy(view(y, :, case_idx)), repeats))
        end
    end
    return jobs
end

"""Run one XOR minibatch and synchronize updated parameters."""
function run_edge_batch!(manager::M, jobs::J, config::C, update_idx::Integer) where {M<:StatefulAlgorithms.ProcessManager,J,C<:EdgeApplicationConfig}
    clear_manager_buffers!(manager)
    manager.state.total_repeats[] = sum(job.repeats for job in jobs)
    StatefulAlgorithms.run!(manager, jobs)
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

"""Compute scalar scores from output-line states for the configured readout."""
function edge_output_scores(outputs::O, config::C) where {O<:AbstractMatrix,C<:EdgeApplicationConfig}
    if config.output_mode === :majority
        return vec(mean(outputs; dims = 1))
    elseif config.output_mode === :two_class
        split = size(outputs, 1) ÷ 2
        false_scores = vec(mean(view(outputs, 1:split, :); dims = 1))
        true_scores = vec(mean(view(outputs, (split + 1):size(outputs, 1), :); dims = 1))
        return true_scores .- false_scores
    else
        throw(ArgumentError("unknown edge output mode `$(config.output_mode)`; use `majority` or `two_class`"))
    end
end

"""Run one validation relaxation and return the configured output view."""
function edge_forward_output!(
    layer::L,
    graph::G,
    algorithm::A,
    context::K,
    x::X,
    ps::P,
    config::C,
) where {L<:LayeredIsingGraphLayer,G,A,K,X<:AbstractVector,P,C<:EdgeApplicationConfig}
    IsingLearning.sync_params!(graph, ps)
    initialize_edge_state!(graph, config.init_mode)
    IsingLearning.apply_input(graph, x)
    IsingLearning.relax_context!(algorithm, context, layer.free_relaxation_steps)
    return IsingLearning.graph_view(graph, layer.output_layer)
end

"""Evaluate readout score, physical-output MSE, margins, and XOR accuracy."""
function evaluate_edge_xor(layer::L, ps::P, st::S, x::X, y::Y, config::C) where {L<:LayeredIsingGraphLayer,P,S,X<:AbstractMatrix,Y<:AbstractMatrix,C<:EdgeApplicationConfig}
    outputs = zeros(FT, size(y, 1), size(y, 2))
    graph = st.graph
    algorithm = layer.validation_algorithm
    context = StatefulAlgorithms.init(algorithm, (; model = graph))
    for _ in 1:config.eval_repeats
        for case_idx in axes(x, 2)
            outputs[:, case_idx] .+= edge_forward_output!(
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

    scores = edge_output_scores(outputs, config)
    targets = FT[xor(a, b) ? 1 : -1 for (a, b) in EDGE_XOR_CASES]
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

"""Append one named-tuple row to a CSV file."""
function append_metrics!(path::P, row) where {P<:AbstractString}
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        names = propertynames(row)
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return selected NN and seed-repeat configs for an edge robustness run."""
function edge_configs(base::C, nns::N, seed_indices::S) where {C<:EdgeApplicationConfig,N<:AbstractVector{Int},S<:AbstractVector{Int}}
    configs = EdgeApplicationConfig[]
    for nn in nns, seed_index in seed_indices
        name = "nn$(nn)_$(seed_label(seed_index))"
        seed = base.seed + 10_000 * seed_index + 101 * nn
        push!(configs, copy_config(base; name, local_nn = nn, seed_index, seed))
    end
    return configs
end

"""Return the default NN/seed configs selected by environment variables."""
function experiment_configs(base::C) where {C<:EdgeApplicationConfig}
    nns = parse_int_list(get(ENV, "ISING_XOR_EDGE_NNS", "1,2,3,4,5,6,7,8,9,10"), collect(1:10))
    seed_indices = parse_int_list(get(ENV, "ISING_XOR_EDGE_SEEDS", "1,2,3,4,5"), collect(1:5))
    limit = parse(Int, get(ENV, "ISING_XOR_EDGE_LIMIT", "0"))
    configs = edge_configs(base, nns, seed_indices)
    if limit > 0
        resize!(configs, min(limit, length(configs)))
    end
    return configs
end

"""Return the learning-summary plot path inside the edge run folder."""
function edge_summary_plot_path(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    return joinpath(outdir, "learning_summary.png")
end

"""Save an edge result plot as the requested PNG file."""
function save_edge_plot(path::P, fig) where {P<:AbstractString}
    save(path, fig)
    return path
end

"""Return line style and width for an edge NN group."""
function edge_line_style(nn::Integer)
    nn <= 3 && return (:solid, 2.0)
    nn <= 7 && return (:dash, 2.8)
    return (:dot, 3.6)
end

"""Plot learning curves, accuracy, margins, and the best margin comparison."""
function plot_results(outdir::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    isempty(rows) && return nothing
    fig = Figure(size = (1500, 950))
    ax_mse = Axis(fig[1, 1], xlabel = "epoch", ylabel = "score MSE", title = "Edge propagation learning")
    ax_acc = Axis(fig[2, 1], xlabel = "epoch", ylabel = "accuracy", title = "Accuracy")
    ax_margin = Axis(fig[3, 1], xlabel = "epoch", ylabel = "min margin", title = "Worst-case margin")
    ax_best = Axis(fig[1:3, 2], xlabel = "configuration", ylabel = "best min margin", title = "Best Worst-Case Margin")

    palette = Makie.wong_colors()
    configs = unique(row.config for row in rows)
    for (idx, config_name) in enumerate(configs)
        subset = [row for row in rows if row.config == config_name]
        color = palette[mod1(idx, length(palette))]
        nn = isempty(subset) ? 0 : subset[1].nn
        style, width = edge_line_style(nn)
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

    path = edge_summary_plot_path(outdir)
    save_edge_plot(path, fig)
    return path
end

"""Plot the learning history for one NN/seed configuration inside its folder."""
function plot_config_results(run_dir::P, rows::R, config::C) where {P<:AbstractString,R<:AbstractVector,C<:EdgeApplicationConfig}
    isempty(rows) && return nothing

    fig = Figure(size = (1200, 850))
    title = "NN $(config.local_nn), $(seed_label(config.seed_index))"
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
    save_edge_plot(path, fig)
    return path
end

"""Return one robustness row per NN by aggregating seed summaries."""
function nn_robustness_rows(summary_rows::S) where {S<:AbstractVector}
    nns = sort(unique(row.nn for row in summary_rows))
    rows = NamedTuple[]
    for nn in nns
        subset = [row for row in summary_rows if row.nn == nn]
        solved = [row for row in subset if row.best_accuracy == 1.0 && row.best_min_margin > 0 && row.best_mse < 0.1]
        push!(
            rows,
            (;
                nn,
                runs = length(subset),
                solved = length(solved),
                solved_fraction = length(solved) / length(subset),
                best_mse = minimum(row.best_mse for row in subset),
                mean_best_mse = mean(row.best_mse for row in subset),
                best_min_margin = maximum(row.best_min_margin for row in subset),
                mean_best_min_margin = mean(row.best_min_margin for row in subset),
            ),
        )
    end
    return rows
end

"""Plot NN robustness across repeated seeds."""
function plot_robustness_results(outdir::P, robustness_rows::R) where {P<:AbstractString,R<:AbstractVector}
    isempty(robustness_rows) && return nothing

    fig = Figure(size = (1450, 900))
    xvals = [row.nn for row in robustness_rows]
    ax_success = Axis(fig[1, 1], xlabel = "NN", ylabel = "solved seed fraction", title = "Robustness across seeds")
    ax_mse = Axis(fig[2, 1], xlabel = "NN", ylabel = "mean best MSE", title = "Mean best MSE")
    ax_margin = Axis(fig[3, 1], xlabel = "NN", ylabel = "mean best min margin", title = "Mean best margin")
    barplot!(ax_success, xvals, [row.solved_fraction for row in robustness_rows], color = :seagreen)
    barplot!(ax_mse, xvals, [row.mean_best_mse for row in robustness_rows], color = :steelblue)
    barplot!(ax_margin, xvals, [row.mean_best_min_margin for row in robustness_rows], color = :firebrick)
    path = joinpath(outdir, "robustness_summary.png")
    save_edge_plot(path, fig)
    return path
end

"""Write one serialized snapshot for later inspection or restart."""
function save_snapshot!(run_dir::P, epoch::Integer, ps::S, config::C, metrics::M) where {P<:AbstractString,S,C<:EdgeApplicationConfig,M}
    snapshot_dir = joinpath(run_dir, "snapshots")
    mkpath(snapshot_dir)
    path = joinpath(snapshot_dir, "epoch_$(lpad(string(epoch), 6, '0')).bin")
    open(path, "w") do io
        serialize(io, (; epoch = Int(epoch), ps, config, metrics, cases = EDGE_XOR_CASES))
    end
    return path
end

"""Load resumed parameters for one NN/seed config when a resume directory is set."""
function maybe_resume_params(ps::P, config::C) where {P,C<:EdgeApplicationConfig}
    isempty(config.resume_dir) && return ps
    path = joinpath(config.resume_dir, config.name, config.resume_file)
    isfile(path) || throw(ArgumentError("resume file `$path` does not exist"))
    payload = open(deserialize, path)
    hasproperty(payload, :ps) || throw(ArgumentError("resume file `$path` does not contain a `ps` field"))
    return payload.ps
end

"""Run one NN configuration and return metric rows plus summary."""
function run_config!(config::C, root_outdir::P) where {C<:EdgeApplicationConfig,P<:AbstractString}
    run_dir = joinpath(root_outdir, config.name)
    isdir(run_dir) && rm(run_dir; recursive = true, force = true)
    mkpath(run_dir)

    graph = edge_xor_graph(config)
    layer = edge_xor_layer(graph, config)
    ps = LuxCore.initialparameters(Random.MersenneTwister(config.seed + 20), layer)
    ps = maybe_resume_params(ps, config)
    st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 21), layer)
    x, y = edge_xor_dataset(config, FT)
    jobs = edge_xor_jobs(config, x, y)
    manager = edge_xor_manager(layer, graph, ps, config)

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
            absolute_epoch = config.start_epoch + epoch
            should_snapshot = config.snapshot_every > 0 && epoch % config.snapshot_every == 0
            should_record = epoch == 0 || epoch % config.log_every == 0 || should_snapshot || epoch == config.epochs
            if should_record
                metrics = evaluate_edge_xor(layer, manager.state.params[], st, x, y, config)
                if metrics.mse < best_mse
                    best_mse = metrics.mse
                    best_epoch = absolute_epoch
                    best_accuracy = metrics.accuracy
                    open(joinpath(run_dir, "best_params.bin"), "w") do io
                        serialize(io, (; ps = manager.state.params[], config, metrics, cases = EDGE_XOR_CASES))
                    end
                end
                if metrics.min_margin > best_min_margin
                    best_min_margin = metrics.min_margin
                    best_margin_epoch = absolute_epoch
                    open(joinpath(run_dir, "best_margin_params.bin"), "w") do io
                        serialize(io, (; ps = manager.state.params[], config, metrics, cases = EDGE_XOR_CASES))
                    end
                end
                if metrics.all_correct && first_all_correct_epoch < 0
                    first_all_correct_epoch = absolute_epoch
                end
                should_snapshot && save_snapshot!(run_dir, absolute_epoch, manager.state.params[], config, metrics)
                row = (;
                    config = config.name,
                    nn = config.local_nn,
                    seed_index = config.seed_index,
                    seed = config.seed,
                    epoch = absolute_epoch,
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
            last_update = run_edge_batch!(manager, jobs, config, epoch + 1)
        end
    finally
        close(manager)
    end

    open(joinpath(run_dir, "final_params.bin"), "w") do io
        serialize(io, (; ps = manager.state.params[], config, cases = EDGE_XOR_CASES))
    end
    config.plot && plot_config_results(run_dir, rows, config)
    return (;
        rows,
        summary = (;
            config = config.name,
            nn = config.local_nn,
            seed_index = config.seed_index,
            seed = config.seed,
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

"""Write aggregate metrics and summary CSV files."""
function write_aggregate_csvs!(outdir::P, rows::R, summary_rows::S) where {P<:AbstractString,R<:AbstractVector,S<:AbstractVector}
    metrics_path = joinpath(outdir, "metrics.csv")
    summary_path = joinpath(outdir, "summary.csv")
    for path in (metrics_path, summary_path)
        isfile(path) && rm(path)
    end
    for row in rows
        append_metrics!(metrics_path, row)
    end
    for row in summary_rows
        append_metrics!(summary_path, row)
    end
    return (; metrics_path, summary_path)
end

"""Run an edge-application NN comparison for a prepared config set."""
function run_edge_grid!(base::C, configs::S = experiment_configs(base)) where {C<:EdgeApplicationConfig,S<:AbstractVector{EdgeApplicationConfig}}
    base.workers > 0 || throw(ArgumentError("ISING_XOR_EDGE_WORKERS must be positive"))
    base.repeats_per_case > 0 || throw(ArgumentError("ISING_XOR_EDGE_REPEATS must be positive"))
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers
    base.side > 1 || throw(ArgumentError("ISING_XOR_EDGE_SIDE must be at least 2"))
    base.output_mode === :two_class && isodd(base.side) && throw(ArgumentError("two_class edge output mode needs an even ISING_XOR_EDGE_SIDE"))

    isempty(configs) && throw(ArgumentError("no edge application configs selected"))
    mkpath(base.outdir)

    all_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    println("Running $(length(configs)) edge-application XOR config(s)")
    for (idx, config) in enumerate(configs)
        println("[$idx/$(length(configs))] $(config.name)")
        result = run_config!(config, base.outdir)
        append!(all_rows, result.rows)
        push!(summary_rows, result.summary)
    end

    robustness_rows = nn_robustness_rows(summary_rows)
    csvs = write_aggregate_csvs!(base.outdir, all_rows, summary_rows)
    robustness_path = joinpath(base.outdir, "robustness_summary.csv")
    isfile(robustness_path) && rm(robustness_path)
    for row in robustness_rows
        append_metrics!(robustness_path, row)
    end
    plot_path = base.plot ? plot_results(base.outdir, all_rows, summary_rows) : nothing
    robustness_plot_path = base.plot ? plot_robustness_results(base.outdir, robustness_rows) : nothing
    println("saved metrics ", csvs.metrics_path)
    println("saved summary ", csvs.summary_path)
    println("saved robustness ", robustness_path)
    !isnothing(plot_path) && println("saved plot ", plot_path)
    !isnothing(robustness_plot_path) && println("saved robustness plot ", robustness_plot_path)
    return (; outdir = base.outdir, rows = all_rows, summary_rows, robustness_rows, plot_path, robustness_plot_path)
end

"""Run the full edge-application NN comparison."""
function main()
    return run_edge_grid!(EdgeApplicationConfig())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
