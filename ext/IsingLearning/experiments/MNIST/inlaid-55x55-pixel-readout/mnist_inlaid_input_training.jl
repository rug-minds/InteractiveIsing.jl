using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CairoMakie
using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using MLDatasets
using Optimisers
using Random
using Serialization
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const INMNIST_FT = Float32
const INMNIST_INPUT_SIDE = 28
const INMNIST_SIDE = 2 * INMNIST_INPUT_SIDE - 1
const INMNIST_NCLASSES = 10

"""
Static active-index set for the inlaid MNIST input layer.

The 28x28 MNIST pixel sites are fixed state entries inside the same 55x55 layer
as the separator spins. Only separator sites and output spins are sampled.
"""
struct InlaidMNISTActiveSet{V<:AbstractVector{Int32}} <: II.UniformIndexPicker
    active::V
end

"""Return the active graph indices for sweep-style algorithms."""
II.sampling_indices(index_set::InlaidMNISTActiveSet) = index_set.active

"""Sample one active graph index uniformly."""
II.pick_idx(rng::Random.AbstractRNG, index_set::InlaidMNISTActiveSet) = rand(rng, index_set.active)

Base.length(index_set::InlaidMNISTActiveSet) = length(index_set.active)

Base.@kwdef struct InlaidMNISTConfig
    name::String = get(ENV, "ISING_MNIST_INLAID_NAME", "inlaid_readout")
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_WORKERS", "32"))
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_EPOCHS", "12"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_BATCHSIZE", "256"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_TRAIN_PER_CLASS", "100"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_TEST_PER_CLASS", "40"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_OUTPUT_REPLICAS", "4"))
    free_reads::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_FREE_READS", "1"))
    nudge_reads::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_NUDGE_READS", "1"))
    eval_reads::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_EVAL_READS", "3"))
    free_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_FREE_SWEEPS", "75"))
    nudge_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_NUDGE_SWEEPS", "75"))
    eval_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_EVAL_SWEEPS", "100"))
    β::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_BETA", "20.0"))
    target_on::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_TARGET_ON", "1.0"))
    target_off::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_TARGET_OFF", "-1.0"))
    lr::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_LR", "0.01"))
    lr_decay::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_LR_DECAY", "0.995"))
    lr_min::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_LR_MIN", "0.001"))
    optimizer::String = lowercase(get(ENV, "ISING_MNIST_INLAID_OPTIMIZER", "adam"))
    weight_decay::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_WEIGHT_DECAY", "0.0"))
    weight_clip::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_WEIGHT_CLIP", "2.0"))
    bias_clip::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_BIAS_CLIP", "2.0"))
    applied_bias_clip::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_APPLIED_BIAS_CLIP", "20.0"))
    train_live_readout::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_INLAID_TRAIN_LIVE_READOUT", "false")))
    readout_gain::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_READOUT_GAIN", "0.03"))
    input_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_INPUT_RADIUS", "1"))
    input_internal_scale::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_INPUT_SCALE", "0.10"))
    output_replica_scale::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_OUTPUT_REPLICA_SCALE", "0.10"))
    output_competition_scale::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_OUTPUT_COMPETITION_SCALE", "0.05"))
    hot_temp::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_HOT_TEMP", "5.0"))
    cold_temp::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_COLD_TEMP", "0.05"))
    reverse_temp::INMNIST_FT = parse(INMNIST_FT, get(ENV, "ISING_MNIST_INLAID_REVERSE_TEMP", "1.0"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_INLAID_SEED", "5317"))
    outdir::String = get(
        ENV,
        "ISING_MNIST_INLAID_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "inlaid_input_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

mutable struct InlaidMNISTModel{C,G,W,B,P,L,A,O,M,R}
    config::C
    graph::G
    weights_io::W
    bias_o::B
    pixel_idxs::P
    live_input_idxs::L
    input_idxs::A
    output_idxs::O
    readout_mask::M
    rng::R
end

struct InlaidMNISTJob{X<:AbstractVector{INMNIST_FT},Y<:AbstractVector{INMNIST_FT}}
    x::X
    y::Y
end

struct InlaidMNISTStep <: Processes.ProcessAlgorithm end

mutable struct InlaidMNISTManagerState{M,G,P,O}
    model::M
    batch_gradient::G
    nsamples::Base.RefValue{Int}
    ncorrect::Base.RefValue{Int}
    nskipped::Base.RefValue{Int}
    total_loss::Base.RefValue{INMNIST_FT}
    params::Base.RefValue{P}
    opt_state::O
    update_idx::Base.RefValue{Int}
end

"""Return a compact rectangular shape for output replicas."""
function output_shape(units::I) where {I<:Integer}
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Return the clamped MNIST pixel graph indices inside the 55x55 input layer."""
function inlaid_pixel_indices(graph::G) where {G}
    layer = graph[1]
    idxs = reshape(collect(II.layerrange(layer)), INMNIST_SIDE, INMNIST_SIDE)
    pixels = Vector{Int32}(undef, INMNIST_INPUT_SIDE^2)
    out_idx = 1
    @inbounds for col in 1:INMNIST_INPUT_SIDE, row in 1:INMNIST_INPUT_SIDE
        pixels[out_idx] = Int32(idxs[2 * row - 1, 2 * col - 1])
        out_idx += 1
    end
    return pixels
end

"""Return the live separator indices in the inlaid input layer."""
function inlaid_live_input_indices(graph::G) where {G}
    pixels = Set(inlaid_pixel_indices(graph))
    live = Int32[]
    for idx in II.layerrange(graph[1])
        idx32 = Int32(idx)
        idx32 in pixels && continue
        push!(live, idx32)
    end
    return live
end

"""Return all sampled graph indices: separator sites and output replicas."""
function inlaid_active_indices(graph::G) where {G}
    active = inlaid_live_input_indices(graph)
    append!(active, Int32.(collect(II.layerrange(graph[2]))))
    return active
end

"""Create the inlaid-input graph with trainable dense readout structure."""
function sampled_graph(config::C, rng::R; shared_adj = nothing) where {C<:InlaidMNISTConfig,R<:Random.AbstractRNG}
    output_rows, output_cols = output_shape(INMNIST_NCLASSES * config.output_replicas)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    input = II.Layer(INMNIST_SIDE, INMNIST_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, INMNIST_SIDE + 3, 0); periodic = false)
    graph = II.IsingGraph(
        input,
        zero_wg,
        out,
        II.Bilinear() + II.MagField(b = g -> II.filltype(Vector, 0f0, II.statelen(g)));
        precision = INMNIST_FT,
        adj = shared_adj,
        index_set = g -> InlaidMNISTActiveSet(inlaid_active_indices(g)),
    )
    if isnothing(shared_adj)
        add_fixed_input_edges!(graph, rng; radius = config.input_internal_radius, scale = config.input_internal_scale)
        add_output_replica_edges!(graph, config)
    end
    II.temp!(graph, config.cold_temp)
    return graph
end

"""Add fixed local couplings inside the inlaid input layer."""
function add_fixed_input_edges!(
    graph::G,
    rng::R;
    radius::I,
    scale::T,
) where {G,R<:Random.AbstractRNG,I<:Integer,T<:AbstractFloat}
    radius <= 0 && return graph
    scale == 0 && return graph
    layer = graph[1]
    rows, cols = size(layer)
    idxs = reshape(collect(II.layerrange(layer)), rows, cols)
    adj = II.adj(graph)
    @inbounds for col in 1:cols, row in 1:rows
        src = idxs[row, col]
        for dcol in -Int(radius):Int(radius), drow in -Int(radius):Int(radius)
            drow == 0 && dcol == 0 && continue
            dst_row = row + drow
            dst_col = col + dcol
            (1 <= dst_row <= rows && 1 <= dst_col <= cols) || continue
            dst = idxs[dst_row, dst_col]
            dst <= src && continue
            w = scale * randn(rng, INMNIST_FT)
            adj[dst, src] = w
            adj[src, dst] = w
        end
    end
    return graph
end

"""Add fixed within-digit alignment and between-digit inhibition for outputs."""
function add_output_replica_edges!(graph::G, config::C) where {G,C<:InlaidMNISTConfig}
    output_idxs = collect(II.layerrange(graph[2]))
    adj = II.adj(graph)
    replicas = config.output_replicas
    @inbounds for local_a in eachindex(output_idxs), local_b in (local_a + 1):length(output_idxs)
        digit_a = (local_a - 1) ÷ replicas
        digit_b = (local_b - 1) ÷ replicas
        w = digit_a == digit_b ? config.output_replica_scale : -config.output_competition_scale
        w == 0 && continue
        src = output_idxs[local_a]
        dst = output_idxs[local_b]
        adj[dst, src] = w
        adj[src, dst] = w
    end
    return graph
end

"""Restore fixed output couplings after dense readout updates touch those slots."""
function sync_output_fixed_edges!(model::M) where {M<:InlaidMNISTModel}
    output_idxs = model.output_idxs
    adj = II.adj(model.graph)
    replicas = model.config.output_replicas
    @inbounds for local_a in eachindex(output_idxs), local_b in (local_a + 1):length(output_idxs)
        digit_a = (local_a - 1) ÷ replicas
        digit_b = (local_b - 1) ÷ replicas
        w = digit_a == digit_b ? model.config.output_replica_scale : -model.config.output_competition_scale
        if w != 0
            src = output_idxs[local_a]
            dst = output_idxs[local_b]
            adj[dst, src] = w
            adj[src, dst] = w
        end
    end
    return model
end

"""Initialize trainable readout parameters and install them into the graph."""
function init_model(config::C, seed::I = config.seed; shared_adj = nothing) where {C<:InlaidMNISTConfig,I<:Integer}
    rng = Random.MersenneTwister(Int(seed))
    graph = sampled_graph(config, rng; shared_adj)
    input_n = INMNIST_SIDE^2
    output_n = INMNIST_NCLASSES * config.output_replicas
    weight_scale = config.readout_gain / sqrt(INMNIST_FT(input_n))
    pixel_idxs = inlaid_pixel_indices(graph)
    input_idxs = Int32.(collect(II.layerrange(graph[1])))
    readout_mask = fill(config.train_live_readout, input_n)
    first_input_idx = first(input_idxs)
    @inbounds for idx in pixel_idxs
        readout_mask[Int(idx - first_input_idx + 1)] = true
    end
    weights_io = weight_scale .* randn(rng, INMNIST_FT, input_n, output_n)
    weights_io .*= reshape(INMNIST_FT.(readout_mask), :, 1)
    model = InlaidMNISTModel(
        config,
        graph,
        weights_io,
        zeros(INMNIST_FT, output_n),
        pixel_idxs,
        inlaid_live_input_indices(graph),
        input_idxs,
        Int32.(collect(II.layerrange(graph[2]))),
        readout_mask,
        rng,
    )
    sync_graph_couplings!(model)
    return model
end

"""Create one worker model with local state/fields and shared adjacency."""
function worker_model(source::M, worker_idx::I) where {M<:InlaidMNISTModel,I<:Integer}
    model = init_model(source.config, source.config.seed + 10_000 + Int(worker_idx); shared_adj = II.adj(source.graph))
    model.weights_io = source.weights_io
    model.bias_o = source.bias_o
    return model
end

"""Install the trainable readout matrix into the shared adjacency."""
function sync_graph_couplings!(model::M) where {M<:InlaidMNISTModel}
    adj = II.adj(model.graph)
    @inbounds for input_pos in axes(model.weights_io, 1)
        model.readout_mask[input_pos] || continue
        input_idx = model.input_idxs[input_pos]
        for output_pos in axes(model.weights_io, 2)
            output_idx = model.output_idxs[output_pos]
            w = -model.weights_io[input_pos, output_pos]
            adj[output_idx, input_idx] = w
            adj[input_idx, output_idx] = w
        end
    end
    sync_output_fixed_edges!(model)
    return model
end

"""Return Optimisers-compatible parameter arrays."""
function parameters(model::M) where {M<:InlaidMNISTModel}
    return (; weights_io = model.weights_io, bias_o = model.bias_o)
end

"""Replace source and worker parameter references after an optimizer update."""
function set_parameters!(model::M, ps::P) where {M<:InlaidMNISTModel,P<:NamedTuple}
    model.weights_io = clamp.(ps.weights_io, -model.config.weight_clip, model.config.weight_clip)
    model.weights_io .*= reshape(INMNIST_FT.(model.readout_mask), :, 1)
    model.bias_o = clamp.(ps.bias_o, -model.config.bias_clip, model.config.bias_clip)
    sync_graph_couplings!(model)
    return model
end

"""Allocate a gradient buffer matching the trainable readout parameters."""
function gradient_buffer(model::M) where {M<:InlaidMNISTModel}
    return (;
        weights_io = zeros(INMNIST_FT, size(model.weights_io)),
        bias_o = zeros(INMNIST_FT, size(model.bias_o)),
    )
end

"""Set every array in a gradient buffer to zero."""
function clear_gradient!(gradient::G) where {G<:NamedTuple}
    fill!(gradient.weights_io, 0f0)
    fill!(gradient.bias_o, 0f0)
    return gradient
end

"""Add one gradient buffer into another."""
function add_gradient!(dest::D, src::S) where {D<:NamedTuple,S<:NamedTuple}
    dest.weights_io .+= src.weights_io
    dest.bias_o .+= src.bias_o
    return dest
end

"""Scale a gradient buffer in place."""
function scale_gradient!(gradient::G, scale::T) where {G<:NamedTuple,T<:Real}
    gradient.weights_io .*= INMNIST_FT(scale)
    gradient.bias_o .*= INMNIST_FT(scale)
    return gradient
end

"""Add L2 decay to readout weights before the optimizer step."""
function add_weight_decay!(gradient::G, ps::P, λ::T) where {G<:NamedTuple,P<:NamedTuple,T<:Real}
    λ > zero(T) || return gradient
    gradient.weights_io .+= INMNIST_FT(λ) .* ps.weights_io
    return gradient
end

"""Return the optimizer learning rate for a one-indexed update."""
function learning_rate_at(config::C, update_idx::I) where {C<:InlaidMNISTConfig,I<:Integer}
    scheduled = config.lr * config.lr_decay^max(Int(update_idx) - 1, 0)
    return max(config.lr_min, scheduled)
end

"""Update the Optimisers.jl learning rate while preserving Adam moments."""
function adjust_learning_rate!(manager::M) where {M<:Processes.ProcessManager}
    η = learning_rate_at(manager.config, manager.state.update_idx[])
    Optimisers.adjust!(manager.state.opt_state, η)
    return η
end

"""Write one bipolar image into the fixed pixel sites."""
function apply_inlaid_pattern!(model::M, x::X) where {M<:InlaidMNISTModel,X<:AbstractVector{INMNIST_FT}}
    @inbounds II.state(model.graph)[model.pixel_idxs] .= x
    return model
end

"""Apply free or nudged output magnetic fields."""
function apply_output_field!(model::M; target = nothing, beta::Real = 0) where {M<:InlaidMNISTModel}
    bias = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    fill!(bias, 0f0)
    output_bias = isnothing(target) ? model.bias_o : model.bias_o .- INMNIST_FT(beta) .* target
    @inbounds bias[model.output_idxs] .= .-clamp.(output_bias, -model.config.applied_bias_clip, model.config.applied_bias_clip)
    return model
end

"""Randomize live separator and output states while leaving pixels fixed."""
function randomize_live_state!(model::M) where {M<:InlaidMNISTModel}
    state = II.state(model.graph)
    @inbounds for idx in model.live_input_idxs
        state[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    @inbounds for idx in model.output_idxs
        state[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    return model
end

"""Evaluate the current sparse Hamiltonian energy."""
function graph_energy(model::M) where {M<:InlaidMNISTModel}
    state = II.state(model.graph)
    bias = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    adj = II.adj(model.graph)
    colptrs = SparseArrays.getcolptr(adj)
    rowvals = SparseArrays.rowvals(adj)
    nzvals = SparseArrays.nonzeros(adj)
    energy = 0f0
    @inbounds for col in 1:size(adj, 2)
        for ptr in colptrs[col]:(colptrs[col + 1] - 1)
            energy -= 0.5f0 * nzvals[ptr] * state[rowvals[ptr]] * state[col]
        end
    end
    @inbounds for idx in eachindex(state)
        energy -= bias[idx] * state[idx]
    end
    return energy
end

"""Run full active-spin Metropolis sweeps with cooling or reverse annealing."""
function anneal!(model::M, context::C, sweeps::I; reverse::Bool = false) where {M<:InlaidMNISTModel,C,I<:Integer}
    total = max(Int(sweeps), 1)
    nactive = length(II.sampling_indices(model.graph))
    algorithm = II.Metropolis()
    for sweep in 1:total
        progress = total == 1 ? 1f0 : INMNIST_FT(sweep - 1) / INMNIST_FT(total - 1)
        temp = if reverse
            progress <= 0.5f0 ?
                model.config.cold_temp + (progress / 0.5f0) * (model.config.reverse_temp - model.config.cold_temp) :
                model.config.reverse_temp + ((progress - 0.5f0) / 0.5f0) * (model.config.cold_temp - model.config.reverse_temp)
        else
            model.config.hot_temp * (model.config.cold_temp / model.config.hot_temp)^progress
        end
        II.temp!(model.graph, temp)
        for _ in 1:nactive
            Processes.step!(algorithm, context)
        end
    end
    II.temp!(model.graph, model.config.cold_temp)
    return model
end

"""Sample a free or nudged phase and return the lowest-energy state."""
function sample_phase!(
    model::M,
    context::C,
    x::X;
    target = nothing,
    beta::Real = 0,
    reads::I,
    sweeps::J,
    initial_state = nothing,
    reverse::Bool = false,
) where {M<:InlaidMNISTModel,C,X<:AbstractVector{INMNIST_FT},I<:Integer,J<:Integer}
    best_energy = Inf32
    best_state = copy(II.state(model.graph))
    for _ in 1:Int(reads)
        isnothing(initial_state) ? randomize_live_state!(model) : (II.state(model.graph) .= initial_state)
        apply_inlaid_pattern!(model, x)
        before_pixels = copy(@view II.state(model.graph)[model.pixel_idxs])
        apply_output_field!(model; target, beta)
        anneal!(model, context, sweeps; reverse)
        all(before_pixels .== @view II.state(model.graph)[model.pixel_idxs]) || error("inlaid pixel states changed during sampling")
        energy = graph_energy(model)
        if energy < best_energy
            best_energy = energy
            best_state .= II.state(model.graph)
        end
    end
    return best_state, best_energy
end

"""Average output replicas into one score per digit."""
function class_scores(output::V, replicas::I) where {V<:AbstractVector,I<:Integer}
    scores = zeros(INMNIST_FT, INMNIST_NCLASSES)
    @inbounds for digit in 1:INMNIST_NCLASSES
        first_idx = (digit - 1) * Int(replicas) + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + Int(replicas) - 1))) / Int(replicas)
    end
    return scores
end

"""Accumulate one free/nudged contrastive gradient into a worker buffer."""
function accumulate_sample_gradient!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    metropolis_context::C,
) where {G<:NamedTuple,M<:InlaidMNISTModel,X<:AbstractVector{INMNIST_FT},Y<:AbstractVector{INMNIST_FT},C}
    config = model.config
    free_state, _ = sample_phase!(model, metropolis_context, x; reads = config.free_reads, sweeps = config.free_sweeps)
    free_input = copy(@view free_state[model.input_idxs])
    free_o = copy(@view free_state[model.output_idxs])
    correct = argmax(class_scores(free_o, config.output_replicas)) == argmax(class_scores(y, config.output_replicas))
    loss = sum(abs2, y .- free_o) / 2
    if all(free_o .== y)
        return (; loss, correct, skipped = true)
    end

    nudged_state, _ = sample_phase!(
        model,
        metropolis_context,
        x;
        target = y,
        beta = config.β,
        reads = config.nudge_reads,
        sweeps = config.nudge_sweeps,
        initial_state = free_state,
        reverse = true,
    )
    nudged_input = @view nudged_state[model.input_idxs]
    nudged_o = @view nudged_state[model.output_idxs]
    invβ = one(INMNIST_FT) / config.β
    @inbounds for output_pos in axes(gradient.weights_io, 2)
        free_out = free_o[output_pos]
        nudged_out = nudged_o[output_pos]
        gradient.bias_o[output_pos] += -(nudged_out - free_out) * invβ
        for input_pos in axes(gradient.weights_io, 1)
            model.readout_mask[input_pos] || continue
            gradient.weights_io[input_pos, output_pos] +=
                -(nudged_input[input_pos] * nudged_out - free_input[input_pos] * free_out) * invβ
        end
    end
    return (; loss, correct, skipped = false)
end

"""Reset one worker's minibatch accounting fields."""
function reset_worker_stats!(context::C) where {C}
    clear_gradient!(context.gradient)
    context.nsamples[] = 0
    context.ncorrect[] = 0
    context.nskipped[] = 0
    context.total_loss[] = 0f0
    return context
end

"""Return the mutable process subcontext used by a manager worker."""
function worker_context(worker::W) where {W}
    return Processes.context(worker)._state
end

"""Initialize one persistent inlaid MNIST worker context."""
function Processes.init(::InlaidMNISTStep, context)
    model = context.model
    x = get(context, :x, zeros(INMNIST_FT, INMNIST_INPUT_SIDE^2))
    y = get(context, :y, zeros(INMNIST_FT, INMNIST_NCLASSES * model.config.output_replicas))
    gradient = get(context, :gradient, gradient_buffer(model))
    metropolis_context = Processes.init(II.Metropolis(), (; model = model.graph))
    return (;
        model,
        x,
        y,
        gradient,
        metropolis_context,
        nsamples = Ref(0),
        ncorrect = Ref(0),
        nskipped = Ref(0),
        total_loss = Ref(0f0),
    )
end

"""Run one contrastive inlaid MNIST sample inside a reusable worker."""
function Processes.step!(::InlaidMNISTStep, context)
    stats = accumulate_sample_gradient!(context.gradient, context.model, context.x, context.y, context.metropolis_context)
    context.nsamples[] += 1
    context.ncorrect[] += stats.correct ? 1 : 0
    context.nskipped[] += stats.skipped ? 1 : 0
    context.total_loss[] += stats.loss
    return nothing
end

"""Keep persistent worker contexts alive across manager jobs."""
function Processes.cleanup(::InlaidMNISTStep, context)
    return nothing
end

"""Create one reusable manager-owned inlaid MNIST worker."""
function inlaid_worker(source::M, worker_idx::I) where {M<:InlaidMNISTModel,I<:Integer}
    model = worker_model(source, worker_idx)
    return Processes.Process(
        :_state => InlaidMNISTStep(),
        Processes.Init(:_state;
            model,
            x = zeros(INMNIST_FT, INMNIST_INPUT_SIDE^2),
            y = zeros(INMNIST_FT, INMNIST_NCLASSES * source.config.output_replicas),
            gradient = gradient_buffer(model),
        );
        repeat = 1,
    )
end

"""Merge worker-local gradients and statistics into the manager state."""
function flush_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_gradient!(manager.state.batch_gradient)
    manager.state.nsamples[] = 0
    manager.state.ncorrect[] = 0
    manager.state.nskipped[] = 0
    manager.state.total_loss[] = 0f0
    for worker in Processes.workers(manager)
        ctx = worker_context(worker)
        add_gradient!(manager.state.batch_gradient, ctx.gradient)
        manager.state.nsamples[] += ctx.nsamples[]
        manager.state.ncorrect[] += ctx.ncorrect[]
        manager.state.nskipped[] += ctx.nskipped[]
        manager.state.total_loss[] += ctx.total_loss[]
        reset_worker_stats!(ctx)
    end
    nsamples = max(manager.state.nsamples[], 1)
    scale_gradient!(manager.state.batch_gradient, -inv(INMNIST_FT(nsamples)))
    return manager.state.batch_gradient
end

"""Clear every manager and worker buffer before a minibatch."""
function clear_manager_buffers!(manager::M) where {M<:Processes.ProcessManager}
    clear_gradient!(manager.state.batch_gradient)
    manager.state.nsamples[] = 0
    manager.state.ncorrect[] = 0
    manager.state.nskipped[] = 0
    manager.state.total_loss[] = 0f0
    for worker in Processes.workers(manager)
        reset_worker_stats!(worker_context(worker))
    end
    return manager
end

"""Update worker parameter references after an optimizer step."""
function sync_worker_params!(manager::M) where {M<:Processes.ProcessManager}
    for worker in Processes.workers(manager)
        ctx = worker_context(worker)
        ctx.model.weights_io = manager.state.model.weights_io
        ctx.model.bias_o = manager.state.model.bias_o
    end
    return manager
end

"""Create the ProcessManager for inlaid MNIST minibatches."""
function inlaid_manager(source::M) where {M<:InlaidMNISTModel}
    optimizer_name = lowercase(source.config.optimizer)
    optimiser = optimizer_name == "adam" ? Optimisers.Adam(source.config.lr) :
        optimizer_name in ("descent", "sgd") ? Optimisers.Descent(source.config.lr) :
        throw(ArgumentError("unknown optimizer `$(source.config.optimizer)`; use `adam` or `descent`"))
    ps = parameters(source)
    state = InlaidMNISTManagerState(
        source,
        gradient_buffer(source),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0f0),
        Ref(ps),
        Optimisers.setup(optimiser, ps),
        Ref(1),
    )
    recipe = (;
        makeworker = (idx, manager) -> inlaid_worker(manager.state.model, idx),
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x .= job.x
            ctx.y .= job.y
            Processes.resetworker!(slot)
            return nothing
        end,
        flush! = manager -> flush_manager_buffers!(manager),
    )
    return Processes.ProcessManager(
        recipe;
        nworkers = source.config.workers,
        config = source.config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = InlaidMNISTJob{Vector{INMNIST_FT},Vector{INMNIST_FT}},
    )
end

"""Run one minibatch and update shared source parameters once."""
function run_minibatch!(manager::M, jobs::J) where {M<:Processes.ProcessManager,J<:AbstractVector}
    clear_manager_buffers!(manager)
    Processes.run!(manager, jobs, Processes.Dynamic())
    add_weight_decay!(manager.state.batch_gradient, manager.state.params[], manager.config.weight_decay)
    η = adjust_learning_rate!(manager)
    manager.state.opt_state, ps_new = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = ps_new
    set_parameters!(manager.state.model, ps_new)
    manager.state.params[] = parameters(manager.state.model)
    sync_worker_params!(manager)
    manager.state.update_idx[] += 1
    return (;
        nsamples = manager.state.nsamples[],
        accuracy = manager.state.nsamples[] == 0 ? 0.0 : manager.state.ncorrect[] / manager.state.nsamples[],
        loss = manager.state.nsamples[] == 0 ? 0f0 : manager.state.total_loss[] / manager.state.nsamples[],
        skipped = manager.state.nskipped[],
        lr = η,
    )
end

"""Load a balanced MNIST subset with `[0, 1]` fixed pixels and replica targets."""
function balanced_mnist(split::Symbol, per_class::I, config::C) where {I<:Integer,C<:InlaidMNISTConfig}
    dataset = split === :train ? MLDatasets.MNIST(split = :train) :
        split === :test ? MLDatasets.MNIST(split = :test) :
        throw(ArgumentError("split must be :train or :test"))
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:INMNIST_NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end
    keep = Int[]
    for digit in 1:INMNIST_NCLASSES
        append!(keep, @view buckets[digit][1:Int(per_class)])
    end
    x = Matrix{INMNIST_FT}(undef, INMNIST_INPUT_SIDE^2, length(keep))
    y = fill(config.target_off, INMNIST_NCLASSES * config.output_replicas, length(keep))
    for (col, idx) in enumerate(keep)
        image = INMNIST_FT.(reshape(images[:, :, idx], :))
        maximum(image) > 1.5f0 && (image ./= 255f0)
        x[:, col] .= image
        label = Int(labels[idx]) + 1
        first_idx = (label - 1) * config.output_replicas + 1
        y[first_idx:(first_idx + config.output_replicas - 1), col] .= config.target_on
    end
    return x, y
end

"""Create concrete manager jobs for selected sample indices."""
function batch_jobs(x::X, y::Y, indices::V) where {X<:AbstractMatrix,Y<:AbstractMatrix,V<:AbstractVector{Int}}
    jobs = InlaidMNISTJob{Vector{INMNIST_FT},Vector{INMNIST_FT}}[]
    for sample_idx in indices
        push!(jobs, InlaidMNISTJob(copy(view(x, :, sample_idx)), copy(view(y, :, sample_idx))))
    end
    return jobs
end

"""Evaluate free-phase balanced accuracy and loss."""
function evaluate(model::M, x::X, y::Y) where {M<:InlaidMNISTModel,X<:AbstractMatrix,Y<:AbstractMatrix}
    config = model.config
    context = Processes.init(II.Metropolis(), (; model = model.graph))
    correct = 0
    loss = 0f0
    pred_counts = zeros(Int, INMNIST_NCLASSES)
    for sample_idx in axes(x, 2)
        target = view(y, :, sample_idx)
        free_state, _ = sample_phase!(
            model,
            context,
            view(x, :, sample_idx);
            reads = config.eval_reads,
            sweeps = config.eval_sweeps,
        )
        output = @view free_state[model.output_idxs]
        pred = argmax(class_scores(output, config.output_replicas))
        truth = argmax(class_scores(target, config.output_replicas))
        pred_counts[pred] += 1
        correct += pred == truth
        loss += sum(abs2, target .- output) / 2
    end
    return (; accuracy = correct / size(x, 2), loss = loss / size(x, 2), pred_counts)
end

"""Append one named tuple row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Serialize trainable inlaid MNIST parameters."""
function save_model(path::P, model::M) where {P<:AbstractString,M<:InlaidMNISTModel}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (;
            weights_io = model.weights_io,
            bias_o = model.bias_o,
            config = model.config,
            pixel_idxs = model.pixel_idxs,
            live_input_idxs = model.live_input_idxs,
            input_idxs = model.input_idxs,
            output_idxs = model.output_idxs,
            readout_mask = model.readout_mask,
        ))
    end
    return path
end

"""Plot accuracy, loss, and minibatch timing for one run."""
function plot_metrics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    fig = Figure(size = (1250, 820))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "Inlaid MNIST accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Loss")
    ax_time = Axis(fig[1, 2], xlabel = "epoch", ylabel = "seconds", title = "Epoch time")
    ax_pred = Axis(fig[2, 2], xlabel = "digit", ylabel = "count", title = "Final test predictions")
    epochs = [row.epoch for row in rows]
    lines!(ax_acc, epochs, [row.train_accuracy for row in rows], label = "train", color = :steelblue)
    lines!(ax_acc, epochs, [row.test_accuracy for row in rows], label = "test", color = :orange)
    lines!(ax_loss, epochs, [row.train_loss for row in rows], label = "train", color = :steelblue)
    lines!(ax_loss, epochs, [row.test_loss for row in rows], label = "test", color = :orange)
    lines!(ax_time, epochs, [row.epoch_time_s for row in rows], color = :black)
    final_counts = split(string(last(rows).test_pred_counts), ';')
    counts = parse.(Int, final_counts)
    barplot!(ax_pred, 0:9, counts, color = :gray60)
    axislegend(ax_acc, position = :rb)
    save(path, fig)
    return path
end

"""Write a concise README with exact run settings and result summary."""
function write_readme!(path::P, config::C, rows::R) where {P<:AbstractString,C<:InlaidMNISTConfig,R<:AbstractVector}
    best = rows[argmax([row.test_accuracy for row in rows])]
    open(path, "w") do io
        println(io, "# Inlaid Input MNIST")
        println(io)
        println(io, "Use of this folder: saved run artifacts for the 55x55 inlaid-input MNIST architecture.")
        println(io)
        println(io, "- architecture: `55x55 partially dynamic input -> $(INMNIST_NCLASSES * config.output_replicas) output replicas`")
        println(io, "- fixed pixels/live separators: `$(INMNIST_INPUT_SIDE^2)` / `$(INMNIST_SIDE^2 - INMNIST_INPUT_SIDE^2)`")
        println(io, "- workers/batchsize: `$(config.workers)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- free/nudged/eval sweeps: `$(config.free_sweeps)` / `$(config.nudge_sweeps)` / `$(config.eval_sweeps)`")
        println(io, "- reads free/nudged/eval: `$(config.free_reads)` / `$(config.nudge_reads)` / `$(config.eval_reads)`")
        println(io, "- optimizer: `$(config.optimizer)`")
        println(io, "- lr/decay/min: `$(config.lr)` / `$(config.lr_decay)` / `$(config.lr_min)`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- parameter/applied bias clip: `$(config.bias_clip)` / `$(config.applied_bias_clip)`")
        println(io, "- output replica/competition couplings: `$(config.output_replica_scale)` / `$(config.output_competition_scale)`")
        println(io, "- train live separator readout: `$(config.train_live_readout)`")
        println(io, "- best test accuracy: `$(best.test_accuracy)` at epoch `$(best.epoch)`")
        println(io, "- final test accuracy: `$(last(rows).test_accuracy)`")
    end
    return path
end

"""Convert prediction counts to a compact CSV-safe string."""
function count_string(counts::V) where {V<:AbstractVector{Int}}
    return join(counts, ";")
end

"""Train one inlaid-input MNIST run and write all artifacts."""
function train(config::C = InlaidMNISTConfig()) where {C<:InlaidMNISTConfig}
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested workers" threads = Threads.nthreads() workers = config.workers
    mkpath(config.outdir)
    metrics_path = joinpath(config.outdir, "metrics.csv")
    isfile(metrics_path) && rm(metrics_path)

    println("loading MNIST train/test slices")
    train_x, train_y = balanced_mnist(:train, config.train_per_class, config)
    test_x, test_y = balanced_mnist(:test, config.test_per_class, config)
    println("initializing graph and manager")
    model = init_model(config)
    manager = inlaid_manager(model)
    rng = Random.MersenneTwister(config.seed + 77)
    rows = NamedTuple[]
    best_accuracy = -Inf
    best_path = joinpath(config.outdir, "best_params.bin")
    final_path = joinpath(config.outdir, "final_params.bin")

    for epoch in 1:config.epochs
        epoch_start = time_ns()
        order = shuffle(rng, collect(axes(train_x, 2)))
        batch_stats = NamedTuple[]
        for first_idx in 1:config.batchsize:length(order)
            idxs = order[first_idx:min(first_idx + config.batchsize - 1, length(order))]
            stats = run_minibatch!(manager, batch_jobs(train_x, train_y, idxs))
            push!(batch_stats, stats)
        end
        epoch_time = (time_ns() - epoch_start) / 1.0e9

        train_eval = evaluate(model, train_x, train_y)
        test_eval = evaluate(model, test_x, test_y)
        row = (;
            epoch,
            train_accuracy = train_eval.accuracy,
            train_loss = train_eval.loss,
            test_accuracy = test_eval.accuracy,
            test_loss = test_eval.loss,
            batch_accuracy = mean(stat.accuracy for stat in batch_stats),
            batch_loss = mean(stat.loss for stat in batch_stats),
            skipped = sum(stat.skipped for stat in batch_stats),
            lr = last(batch_stats).lr,
            epoch_time_s = epoch_time,
            test_pred_counts = count_string(test_eval.pred_counts),
        )
        append_row!(metrics_path, row)
        push!(rows, row)
        if row.test_accuracy > best_accuracy
            best_accuracy = row.test_accuracy
            save_model(best_path, model)
        end
        println("epoch $(epoch): train=$(round(row.train_accuracy; digits=3)) test=$(round(row.test_accuracy; digits=3)) loss=$(round(row.test_loss; digits=3)) time=$(round(epoch_time; digits=2))s lr=$(row.lr)")
    end

    save_model(final_path, model)
    plot_metrics(joinpath(config.outdir, "progress.png"), rows)
    write_readme!(joinpath(config.outdir, "README.md"), config, rows)
    println("saved metrics: $metrics_path")
    println("saved plot: $(joinpath(config.outdir, "progress.png"))")
    println("saved best params: $best_path")
    return (; config, model, rows, metrics_path, best_path, final_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
