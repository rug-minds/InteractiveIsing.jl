using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CairoMakie
using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using MLDatasets
using Random
using Serialization
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const PMNIST_FT = Float32
const PMNIST_INPUT_SIDE = 28
const PMNIST_INPUT_DIM = PMNIST_INPUT_SIDE^2
const PMNIST_NCLASSES = 10

Base.@kwdef struct PaperMNISTManagerConfig
    name::String = "r7_manager"
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_PM_WORKERS", "32"))
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_PM_EPOCHS", "10"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_PM_BATCHSIZE", "64"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_PM_TRAIN_PER_CLASS", "100"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_PM_TEST_PER_CLASS", "20"))
    hidden1_side::Int = parse(Int, get(ENV, "ISING_MNIST_PM_H1_SIDE", "28"))
    hidden2_side::Int = parse(Int, get(ENV, "ISING_MNIST_PM_H2_SIDE", "11"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_PM_OUTPUT_REPLICAS", "4"))
    local_radius::Int = parse(Int, get(ENV, "ISING_MNIST_PM_RADIUS", "7"))
    internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_PM_INTERNAL_RADIUS", "1"))
    output_internal_radius::Int = parse(Int, get(ENV, "ISING_MNIST_PM_OUTPUT_INTERNAL_RADIUS", "1"))
    free_reads::Int = parse(Int, get(ENV, "ISING_MNIST_PM_FREE_READS", "3"))
    nudge_reads::Int = parse(Int, get(ENV, "ISING_MNIST_PM_NUDGE_READS", "3"))
    free_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_PM_FREE_SWEEPS", "50"))
    nudge_sweeps::Int = parse(Int, get(ENV, "ISING_MNIST_PM_NUDGE_SWEEPS", "50"))
    β::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_BETA", "5.0"))
    target_on::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_TARGET_ON", "1.0"))
    target_off::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_TARGET_OFF", "-1.0"))
    lr_w0::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W0", "0.003"))
    lr_w12::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W12", "0.003"))
    lr_w2o::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W2O", "0.003"))
    lr_w11::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W11", "0.001"))
    lr_w22::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W22", "0.001"))
    lr_woo::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_WOO", "0.001"))
    lr_b::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_B", "0.0003"))
    gain_w0::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W0", "0.5"))
    gain_w12::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W12", "0.25"))
    gain_w2o::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W2O", "0.25"))
    gain_w11::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W11", "0.0"))
    gain_w22::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W22", "0.0"))
    gain_woo::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_WOO", "0.0"))
    internal_scale::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_INTERNAL_SCALE", "0.0"))
    output_internal_scale::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_OUTPUT_INTERNAL_SCALE", "0.0"))
    train_internal::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_TRAIN_INTERNAL", "false")))
    weight_clip::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_WEIGHT_CLIP", "1.0"))
    bias_clip::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_BIAS_CLIP", "1.0"))
    applied_bias_clip::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_APPLIED_BIAS_CLIP", "4.0"))
    hot_temp::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_HOT_TEMP", "5.0"))
    cold_temp::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_COLD_TEMP", "0.01"))
    reverse_temp::PMNIST_FT = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_REVERSE_TEMP", "1.0"))
    gradient_normalization::Symbol = Symbol(get(ENV, "ISING_MNIST_PM_GRADIENT_NORMALIZATION", "sum"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_PM_SEED", "2468"))
    outdir::String = get(
        ENV,
        "ISING_MNIST_PM_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "mnist_local_paper_manager_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

mutable struct PaperMNISTModel{C,G,W0,W12,W2O,W11,W22,WOO,M0,M12,M11,M22,MOO,B1,B2,BO,R}
    config::C
    graph::G
    weights_0::W0
    weights_12::W12
    weights_2o::W2O
    weights_11::W11
    weights_22::W22
    weights_oo::WOO
    mask_0::M0
    mask_12::M12
    mask_11::M11
    mask_22::M22
    mask_oo::MOO
    bias_1::B1
    bias_2::B2
    bias_o::BO
    hidden1_idxs::Vector{Int}
    hidden2_idxs::Vector{Int}
    output_idxs::Vector{Int}
    rng::R
end

struct PaperMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

struct PaperMNISTWorkerStep <: Processes.ProcessAlgorithm end

mutable struct PaperMNISTManagerState{M,G}
    model::M
    batch_gradient::G
    nsamples::Base.RefValue{Int}
    ncorrect::Base.RefValue{Int}
    nskipped::Base.RefValue{Int}
    total_loss::Base.RefValue{PMNIST_FT}
end

"""Parse a comma-separated integer list."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Return a shallow config copy with selected fields replaced."""
function copy_config(config::C; kwargs...) where {C<:PaperMNISTManagerConfig}
    fields = Dict{Symbol,Any}(field => getfield(config, field) for field in fieldnames(C))
    for (field, value) in kwargs
        fields[field] = value
    end
    return PaperMNISTManagerConfig(; fields...)
end

"""Append one named tuple to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return a compact 2D display shape for output replicas."""
function factor_shape(units::I) where {I<:Integer}
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Map a lattice coordinate to the common MNIST input coordinate frame."""
function scaled_position(idx::I, side::J) where {I<:Integer,J<:Integer}
    side <= 1 && return one(PMNIST_FT)
    return one(PMNIST_FT) + PMNIST_FT(idx - 1) * PMNIST_FT(PMNIST_INPUT_SIDE - 1) / PMNIST_FT(side - 1)
end

"""Build a Boolean local-connectivity mask between two square image-like layers."""
function local_mask(src_side::I, dst_side::J, radius::K) where {I<:Integer,J<:Integer,K<:Integer}
    src_n = Int(src_side)^2
    dst_n = Int(dst_side)^2
    mask = falses(src_n, dst_n)
    r = PMNIST_FT(radius)
    @inbounds for dst_col in 1:Int(dst_side), dst_row in 1:Int(dst_side)
        dst_idx = (dst_col - 1) * Int(dst_side) + dst_row
        dst_x = scaled_position(dst_row, Int(dst_side))
        dst_y = scaled_position(dst_col, Int(dst_side))
        for src_col in 1:Int(src_side), src_row in 1:Int(src_side)
            src_idx = (src_col - 1) * Int(src_side) + src_row
            abs(scaled_position(src_row, Int(src_side)) - dst_x) <= r || continue
            abs(scaled_position(src_col, Int(src_side)) - dst_y) <= r || continue
            mask[src_idx, dst_idx] = true
        end
    end
    return mask
end

"""Build a local symmetric mask for one rectangular 2D layer."""
function same_layer_mask(rows::I, cols::J, radius::K) where {I<:Integer,J<:Integer,K<:Integer}
    n = Int(rows) * Int(cols)
    mask = falses(n, n)
    r = Int(radius)
    r <= 0 && return mask
    @inbounds for dst_col in 1:Int(cols), dst_row in 1:Int(rows)
        dst_idx = (dst_col - 1) * Int(rows) + dst_row
        for dcol in -r:r, drow in -r:r
            drow == 0 && dcol == 0 && continue
            src_row = dst_row + drow
            src_col = dst_col + dcol
            (1 <= src_row <= rows && 1 <= src_col <= cols) || continue
            src_idx = (src_col - 1) * Int(rows) + src_row
            mask[src_idx, dst_idx] = true
        end
    end
    return mask
end

"""Initialize a symmetric trainable matrix constrained by a same-layer mask."""
function masked_symmetric_weights(rng::R, mask::M, gain::T) where {R<:Random.AbstractRNG,M<:AbstractMatrix{Bool},T<:AbstractFloat}
    n = size(mask, 1)
    weights = zeros(PMNIST_FT, n, n)
    scale = PMNIST_FT(gain) * sqrt(1f0 / max(n, 1))
    @inbounds for col in 1:n, row in 1:(col - 1)
        (mask[row, col] || mask[col, row]) || continue
        w = scale * (2f0 * rand(rng, PMNIST_FT) - 1f0)
        weights[row, col] = w
        weights[col, row] = w
    end
    return weights
end

"""Apply mask, symmetrize, clear self-couplings, and clip same-layer weights."""
function constrain_symmetric_weights!(weights::W, mask::M, clip::T) where {W<:AbstractMatrix,M<:AbstractMatrix{Bool},T<:Real}
    weights .*= mask
    weights .= 0.5f0 .* (weights .+ transpose(weights))
    @inbounds for idx in axes(weights, 1)
        weights[idx, idx] = 0f0
    end
    weights .= clamp.(weights, -PMNIST_FT(clip), PMNIST_FT(clip))
    return weights
end

"""Construct the sampled hidden1-hidden2-output Ising graph."""
function sampled_graph(config::C, rng::R; shared_adj = nothing) where {C<:PaperMNISTManagerConfig,R<:Random.AbstractRNG}
    output_rows, output_cols = factor_shape(PMNIST_NCLASSES * config.output_replicas)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    h1 = II.Layer(config.hidden1_side, config.hidden1_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    h2 = II.Layer(config.hidden2_side, config.hidden2_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, config.hidden1_side + 2, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, config.hidden1_side + config.hidden2_side + 4, 0); periodic = false)
    graph = II.IsingGraph(
        h1,
        zero_wg,
        h2,
        zero_wg,
        out,
        II.Bilinear() + II.MagField(b = g -> II.filltype(Vector, 0f0, II.statelen(g)));
        precision = PMNIST_FT,
        adj = shared_adj,
        index_set = g -> II.ToggledIndexSet(g),
    )
    if isnothing(shared_adj) && !config.train_internal
        add_internal_edges!(graph, 1, rng; scale = config.internal_scale, radius = config.internal_radius)
        add_internal_edges!(graph, 2, rng; scale = config.internal_scale, radius = config.internal_radius)
        add_internal_edges!(graph, 3, rng; scale = config.output_internal_scale, radius = config.output_internal_radius)
    end
    II.temp!(graph, config.cold_temp)
    return graph
end

"""Add fixed symmetric local couplings inside one layer."""
function add_internal_edges!(
    graph::G,
    layer_idx::I,
    rng::R;
    scale::T,
    radius::J,
) where {G,I<:Integer,R<:Random.AbstractRNG,T<:AbstractFloat,J<:Integer}
    radius <= 0 && return graph
    scale == 0 && return graph
    layer = graph[Int(layer_idx)]
    rows, cols = size(layer)
    idxs = reshape(collect(II.layerrange(layer)), rows, cols)
    A = II.adj(graph)
    @inbounds for col in 1:cols, row in 1:rows
        src = idxs[row, col]
        for dcol in -Int(radius):Int(radius), drow in -Int(radius):Int(radius)
            drow == 0 && dcol == 0 && continue
            dst_row = row + drow
            dst_col = col + dcol
            (1 <= dst_row <= rows && 1 <= dst_col <= cols) || continue
            dst = idxs[dst_row, dst_col]
            dst <= src && continue
            w = scale * randn(rng, T)
            A[dst, src] = -w
            A[src, dst] = -w
        end
    end
    return graph
end

"""Initialize trainable paper-style parameters and install graph couplings."""
function init_model(config::C, seed::I = config.seed) where {C<:PaperMNISTManagerConfig,I<:Integer}
    rng = Random.MersenneTwister(Int(seed))
    graph = sampled_graph(config, rng)
    h1n = config.hidden1_side^2
    h2n = config.hidden2_side^2
    outn = PMNIST_NCLASSES * config.output_replicas
    output_rows, output_cols = factor_shape(outn)

    mask_0 = local_mask(PMNIST_INPUT_SIDE, config.hidden1_side, config.local_radius)
    mask_12 = local_mask(config.hidden1_side, config.hidden2_side, config.local_radius)
    mask_11 = same_layer_mask(config.hidden1_side, config.hidden1_side, config.internal_radius)
    mask_22 = same_layer_mask(config.hidden2_side, config.hidden2_side, config.internal_radius)
    mask_oo = same_layer_mask(output_rows, output_cols, config.output_internal_radius)

    weights_0 = config.gain_w0 .* (2f0 .* rand(rng, PMNIST_FT, PMNIST_INPUT_DIM, h1n) .- 1f0) .* sqrt(1f0 / PMNIST_INPUT_DIM)
    weights_12 = config.gain_w12 .* (2f0 .* rand(rng, PMNIST_FT, h1n, h2n) .- 1f0) .* sqrt(1f0 / max(h1n, 1))
    weights_2o = config.gain_w2o .* (2f0 .* rand(rng, PMNIST_FT, h2n, outn) .- 1f0) .* sqrt(1f0 / max(h2n, 1))
    weights_11 = masked_symmetric_weights(rng, mask_11, config.gain_w11)
    weights_22 = masked_symmetric_weights(rng, mask_22, config.gain_w22)
    weights_oo = masked_symmetric_weights(rng, mask_oo, config.gain_woo)
    weights_0 .*= mask_0
    weights_12 .*= mask_12

    model = PaperMNISTModel(
        config,
        graph,
        weights_0,
        weights_12,
        weights_2o,
        weights_11,
        weights_22,
        weights_oo,
        mask_0,
        mask_12,
        mask_11,
        mask_22,
        mask_oo,
        zeros(PMNIST_FT, h1n),
        zeros(PMNIST_FT, h2n),
        zeros(PMNIST_FT, outn),
        collect(II.layerrange(graph[1])),
        collect(II.layerrange(graph[2])),
        collect(II.layerrange(graph[3])),
        rng,
    )
    sync_graph_couplings!(model)
    return model
end

"""Create a worker model with local state/fields and shared parameter arrays."""
function worker_model(source::M, worker_idx::I) where {M<:PaperMNISTModel,I<:Integer}
    rng = Random.MersenneTwister(source.config.seed + 10_000 + Int(worker_idx))
    graph = sampled_graph(source.config, rng; shared_adj = II.adj(source.graph))
    return PaperMNISTModel(
        source.config,
        graph,
        source.weights_0,
        source.weights_12,
        source.weights_2o,
        source.weights_11,
        source.weights_22,
        source.weights_oo,
        source.mask_0,
        source.mask_12,
        source.mask_11,
        source.mask_22,
        source.mask_oo,
        source.bias_1,
        source.bias_2,
        source.bias_o,
        collect(II.layerrange(graph[1])),
        collect(II.layerrange(graph[2])),
        collect(II.layerrange(graph[3])),
        rng,
    )
end

"""Install trainable couplings into the shared graph adjacency."""
function sync_graph_couplings!(model::M) where {M<:PaperMNISTModel}
    A = II.adj(model.graph)
    config = model.config
    if config.train_internal
        @inbounds for srcpos in axes(model.weights_11, 1)
            srcidx = model.hidden1_idxs[srcpos]
            for dstpos in axes(model.weights_11, 2)
                dstpos <= srcpos && continue
                model.mask_11[srcpos, dstpos] || continue
                dstidx = model.hidden1_idxs[dstpos]
                w = -model.weights_11[srcpos, dstpos]
                A[dstidx, srcidx] = w
                A[srcidx, dstidx] = w
            end
        end
        @inbounds for srcpos in axes(model.weights_22, 1)
            srcidx = model.hidden2_idxs[srcpos]
            for dstpos in axes(model.weights_22, 2)
                dstpos <= srcpos && continue
                model.mask_22[srcpos, dstpos] || continue
                dstidx = model.hidden2_idxs[dstpos]
                w = -model.weights_22[srcpos, dstpos]
                A[dstidx, srcidx] = w
                A[srcidx, dstidx] = w
            end
        end
        @inbounds for srcpos in axes(model.weights_oo, 1)
            srcidx = model.output_idxs[srcpos]
            for dstpos in axes(model.weights_oo, 2)
                dstpos <= srcpos && continue
                model.mask_oo[srcpos, dstpos] || continue
                dstidx = model.output_idxs[dstpos]
                w = -model.weights_oo[srcpos, dstpos]
                A[dstidx, srcidx] = w
                A[srcidx, dstidx] = w
            end
        end
    end
    @inbounds for h1pos in axes(model.weights_12, 1)
        h1idx = model.hidden1_idxs[h1pos]
        for h2pos in axes(model.weights_12, 2)
            model.mask_12[h1pos, h2pos] || continue
            h2idx = model.hidden2_idxs[h2pos]
            w = -model.weights_12[h1pos, h2pos]
            A[h2idx, h1idx] = w
            A[h1idx, h2idx] = w
        end
    end
    @inbounds for h2pos in axes(model.weights_2o, 1)
        h2idx = model.hidden2_idxs[h2pos]
        for opos in axes(model.weights_2o, 2)
            oidx = model.output_idxs[opos]
            w = -model.weights_2o[h2pos, opos]
            A[oidx, h2idx] = w
            A[h2idx, oidx] = w
        end
    end
    return model
end

"""Allocate a gradient buffer matching all trainable paper-style arrays."""
function gradient_buffer(model::M) where {M<:PaperMNISTModel}
    return (;
        weights_0 = zeros(PMNIST_FT, size(model.weights_0)),
        weights_12 = zeros(PMNIST_FT, size(model.weights_12)),
        weights_2o = zeros(PMNIST_FT, size(model.weights_2o)),
        weights_11 = zeros(PMNIST_FT, size(model.weights_11)),
        weights_22 = zeros(PMNIST_FT, size(model.weights_22)),
        weights_oo = zeros(PMNIST_FT, size(model.weights_oo)),
        bias_1 = zeros(PMNIST_FT, size(model.bias_1)),
        bias_2 = zeros(PMNIST_FT, size(model.bias_2)),
        bias_o = zeros(PMNIST_FT, size(model.bias_o)),
    )
end

"""Set every array in a gradient buffer to zero."""
function clear_gradient!(gradient::G) where {G<:NamedTuple}
    for field in propertynames(gradient)
        fill!(getproperty(gradient, field), 0f0)
    end
    return gradient
end

"""Add one gradient buffer into another."""
function add_gradient!(dest::D, src::S) where {D<:NamedTuple,S<:NamedTuple}
    for field in propertynames(dest)
        getproperty(dest, field) .+= getproperty(src, field)
    end
    return dest
end

"""Apply a full minibatch gradient to the shared source parameters."""
function apply_gradient!(model::M, gradient::G, nsamples::I) where {M<:PaperMNISTModel,G<:NamedTuple,I<:Integer}
    config = model.config
    scale = config.gradient_normalization === :mean ? inv(PMNIST_FT(max(nsamples, 1))) :
        config.gradient_normalization === :sum ? one(PMNIST_FT) :
        throw(ArgumentError("gradient_normalization must be :sum or :mean, got $(config.gradient_normalization)"))

    model.weights_0 .+= config.lr_w0 .* scale .* gradient.weights_0
    model.weights_12 .+= config.lr_w12 .* scale .* gradient.weights_12
    model.weights_2o .+= config.lr_w2o .* scale .* gradient.weights_2o
    if config.train_internal
        model.weights_11 .+= config.lr_w11 .* scale .* gradient.weights_11
        model.weights_22 .+= config.lr_w22 .* scale .* gradient.weights_22
        model.weights_oo .+= config.lr_woo .* scale .* gradient.weights_oo
        constrain_symmetric_weights!(model.weights_11, model.mask_11, config.weight_clip)
        constrain_symmetric_weights!(model.weights_22, model.mask_22, config.weight_clip)
        constrain_symmetric_weights!(model.weights_oo, model.mask_oo, config.weight_clip)
    end
    model.weights_0 .*= model.mask_0
    model.weights_12 .*= model.mask_12
    model.bias_1 .+= config.lr_b .* scale .* gradient.bias_1
    model.bias_2 .+= config.lr_b .* scale .* gradient.bias_2
    model.bias_o .+= config.lr_b .* scale .* gradient.bias_o

    model.weights_0 .= clamp.(model.weights_0, -config.weight_clip, config.weight_clip)
    model.weights_12 .= clamp.(model.weights_12, -config.weight_clip, config.weight_clip)
    model.weights_2o .= clamp.(model.weights_2o, -config.weight_clip, config.weight_clip)
    model.bias_1 .= clamp.(model.bias_1, -config.bias_clip, config.bias_clip)
    model.bias_2 .= clamp.(model.bias_2, -config.bias_clip, config.bias_clip)
    model.bias_o .= clamp.(model.bias_o, -config.bias_clip, config.bias_clip)
    sync_graph_couplings!(model)
    return model
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

"""Initialize one persistent worker process context."""
function Processes.init(::PaperMNISTWorkerStep, context)
    model = context.model
    x = get(context, :x, zeros(PMNIST_FT, PMNIST_INPUT_DIM))
    y = get(context, :y, zeros(PMNIST_FT, PMNIST_NCLASSES * model.config.output_replicas))
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

"""Accumulate one paper-style EP sample gradient inside a manager worker."""
function Processes.step!(::PaperMNISTWorkerStep, context)
    stats = accumulate_sample_gradient!(context.gradient, context.model, context.x, context.y, context.metropolis_context)
    context.nsamples[] += 1
    context.ncorrect[] += stats.correct ? 1 : 0
    context.nskipped[] += stats.skipped ? 1 : 0
    context.total_loss[] += stats.loss
    return nothing
end

"""Keep the reusable worker context alive across manager jobs."""
function Processes.cleanup(::PaperMNISTWorkerStep, context)
    return nothing
end

"""Create one reusable manager-owned paper-style worker."""
function paper_worker(source::M, worker_idx::I) where {M<:PaperMNISTModel,I<:Integer}
    model = worker_model(source, worker_idx)
    return Processes.Process(
        :_state => PaperMNISTWorkerStep(),
        Processes.Init(:_state;
            model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            y = zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas),
            gradient = gradient_buffer(model),
        );
        repeat = 1,
    )
end

"""Create the ProcessManager for paper-style MNIST minibatches."""
function paper_manager(source::M) where {M<:PaperMNISTModel}
    state = PaperMNISTManagerState(source, gradient_buffer(source), Ref(0), Ref(0), Ref(0), Ref(0f0))
    recipe = (;
        makeworker = (idx, manager) -> paper_worker(manager.state.model, idx),
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
        job_type = PaperMNISTJob{Vector{PMNIST_FT},Vector{PMNIST_FT}},
    )
end

"""Clear every manager and worker gradient/stat buffer before a minibatch."""
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

"""Merge all worker-local paper-style buffers into the manager state."""
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
    return manager.state.batch_gradient
end

"""Run one manager minibatch and update the shared source parameters once."""
function run_minibatch!(manager::M, jobs::J) where {M<:Processes.ProcessManager,J<:AbstractVector}
    clear_manager_buffers!(manager)
    Processes.run!(manager, jobs, Processes.Dynamic())
    apply_gradient!(manager.state.model, manager.state.batch_gradient, manager.state.nsamples[])
    return (;
        nsamples = manager.state.nsamples[],
        accuracy = manager.state.nsamples[] == 0 ? 0.0 : manager.state.ncorrect[] / manager.state.nsamples[],
        loss = manager.state.nsamples[] == 0 ? 0f0 : manager.state.total_loss[] / manager.state.nsamples[],
        skipped = manager.state.nskipped[],
    )
end

"""Apply per-sample input fields and optional output nudge fields."""
function apply_sample_bias!(
    model::M,
    x::X;
    target = nothing,
    beta::Real = 0,
) where {M<:PaperMNISTModel,X<:AbstractVector}
    config = model.config
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    h1_bias = vec(transpose(x) * model.weights_0) .+ model.bias_1
    b[model.hidden1_idxs] .= .-clamp.(h1_bias, -config.applied_bias_clip, config.applied_bias_clip)
    b[model.hidden2_idxs] .= .-clamp.(model.bias_2, -config.applied_bias_clip, config.applied_bias_clip)
    output_bias = isnothing(target) ? model.bias_o : model.bias_o .- PMNIST_FT(beta) .* target
    b[model.output_idxs] .= .-clamp.(output_bias, -config.applied_bias_clip, config.applied_bias_clip)
    return model
end

"""Reset sampled spins to random Ising states."""
function randomize_state!(model::M) where {M<:PaperMNISTModel}
    s = II.state(model.graph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    return model
end

"""Evaluate the package Hamiltonian for choosing the lowest-energy read."""
function graph_energy(model::M) where {M<:PaperMNISTModel}
    s = II.state(model.graph)
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    A = II.adj(model.graph)
    colptrs = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    nzvals = SparseArrays.nonzeros(A)
    energy = 0f0
    @inbounds for col in 1:size(A, 2)
        for ptr in colptrs[col]:(colptrs[col + 1] - 1)
            energy -= 0.5f0 * nzvals[ptr] * s[rowvals[ptr]] * s[col]
        end
    end
    @inbounds for idx in eachindex(s)
        energy -= b[idx] * s[idx]
    end
    return energy
end

"""Run one full active-spin Metropolis sweep."""
function metropolis_sweep!(context::C, nactive::I) where {C,I<:Integer}
    algorithm = II.Metropolis()
    for _ in 1:Int(nactive)
        Processes.step!(algorithm, context)
    end
    return context
end

"""Run a cooling or reverse-annealing schedule."""
function anneal!(model::M, context::C, sweeps::I; reverse::Bool = false) where {M<:PaperMNISTModel,C,I<:Integer}
    config = model.config
    total = max(Int(sweeps), 1)
    nactive = length(II.state(model.graph))
    for sweep in 1:total
        progress = total == 1 ? 1f0 : PMNIST_FT(sweep - 1) / PMNIST_FT(total - 1)
        T = if reverse
            progress <= 0.5f0 ?
                config.cold_temp + (progress / 0.5f0) * (config.reverse_temp - config.cold_temp) :
                config.reverse_temp + ((progress - 0.5f0) / 0.5f0) * (config.cold_temp - config.reverse_temp)
        else
            config.hot_temp * (config.cold_temp / config.hot_temp)^progress
        end
        II.temp!(model.graph, T)
        metropolis_sweep!(context, nactive)
    end
    II.temp!(model.graph, config.cold_temp)
    return model
end

"""Sample one free or nudged phase and return the lowest-energy state."""
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
) where {M<:PaperMNISTModel,C,X<:AbstractVector,I<:Integer,J<:Integer}
    best_energy = Inf32
    best_state = copy(II.state(model.graph))
    for _ in 1:Int(reads)
        isnothing(initial_state) ? randomize_state!(model) : (II.state(model.graph) .= initial_state)
        apply_sample_bias!(model, x; target, beta)
        anneal!(model, context, sweeps; reverse)
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
    scores = zeros(PMNIST_FT, PMNIST_NCLASSES)
    @inbounds for digit in 1:PMNIST_NCLASSES
        first_idx = (digit - 1) * Int(replicas) + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + Int(replicas) - 1))) / Int(replicas)
    end
    return scores
end

"""Accumulate the paper-style one-sided EP gradient for one sample."""
function accumulate_sample_gradient!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    metropolis_context::C,
) where {G<:NamedTuple,M<:PaperMNISTModel,X<:AbstractVector,Y<:AbstractVector,C}
    config = model.config
    free_state, _ = sample_phase!(model, metropolis_context, x; reads = config.free_reads, sweeps = config.free_sweeps)
    free_h1 = copy(@view free_state[model.hidden1_idxs])
    free_h2 = copy(@view free_state[model.hidden2_idxs])
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
    nudged_h1 = @view nudged_state[model.hidden1_idxs]
    nudged_h2 = @view nudged_state[model.hidden2_idxs]
    nudged_o = @view nudged_state[model.output_idxs]

    invβ = one(PMNIST_FT) / config.β
    h1_delta = nudged_h1 .- free_h1
    h2_delta = nudged_h2 .- free_h2
    o_delta = nudged_o .- free_o

    gradient.weights_2o .+= .-(nudged_h2 * transpose(nudged_o) .- free_h2 * transpose(free_o)) .* invβ
    gradient.weights_12 .+= .-(nudged_h1 * transpose(nudged_h2) .- free_h1 * transpose(free_h2)) .* invβ
    gradient.weights_0 .+= .-(x * transpose(h1_delta)) .* invβ
    if config.train_internal
        gradient.weights_11 .+= .-(nudged_h1 * transpose(nudged_h1) .- free_h1 * transpose(free_h1)) .* invβ
        gradient.weights_22 .+= .-(nudged_h2 * transpose(nudged_h2) .- free_h2 * transpose(free_h2)) .* invβ
        gradient.weights_oo .+= .-(nudged_o * transpose(nudged_o) .- free_o * transpose(free_o)) .* invβ
    end
    gradient.bias_1 .+= .-h1_delta .* invβ
    gradient.bias_2 .+= .-h2_delta .* invβ
    gradient.bias_o .+= .-o_delta .* invβ
    return (; loss, correct, skipped = false)
end

"""Load a balanced MNIST subset with `[0, 1]` inputs and repeated output labels."""
function balanced_mnist(split::Symbol, per_class::I, config::C) where {I<:Integer,C<:PaperMNISTManagerConfig}
    dataset = split === :train ? MLDatasets.MNIST(split = :train) :
        split === :test ? MLDatasets.MNIST(split = :test) :
        throw(ArgumentError("split must be :train or :test"))
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:PMNIST_NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end
    keep = Int[]
    for digit in 1:PMNIST_NCLASSES
        append!(keep, @view buckets[digit][1:Int(per_class)])
    end
    x = Matrix{PMNIST_FT}(undef, PMNIST_INPUT_DIM, length(keep))
    y = fill(config.target_off, PMNIST_NCLASSES * config.output_replicas, length(keep))
    for (col, idx) in enumerate(keep)
        x[:, col] .= PMNIST_FT.(reshape(images[:, :, idx], :))
        maximum(@view x[:, col]) > 1.5f0 && (x[:, col] ./= 255f0)
        label = Int(labels[idx]) + 1
        first_idx = (label - 1) * config.output_replicas + 1
        y[first_idx:(first_idx + config.output_replicas - 1), col] .= config.target_on
    end
    return x, y
end

"""Split selected sample indices into concrete manager jobs."""
function batch_jobs(x::X, y::Y, indices::V) where {X<:AbstractMatrix,Y<:AbstractMatrix,V<:AbstractVector{Int}}
    jobs = PaperMNISTJob{Vector{PMNIST_FT},Vector{PMNIST_FT}}[]
    for sample_idx in indices
        push!(jobs, PaperMNISTJob(copy(view(x, :, sample_idx)), copy(view(y, :, sample_idx))))
    end
    return jobs
end

"""Evaluate balanced accuracy and output loss with free-phase sampling."""
function evaluate(model::M, x::X, y::Y) where {M<:PaperMNISTModel,X<:AbstractMatrix,Y<:AbstractMatrix}
    config = model.config
    context = Processes.init(II.Metropolis(), (; model = model.graph))
    correct = 0
    loss = 0f0
    pred_counts = zeros(Int, PMNIST_NCLASSES)
    for sample_idx in axes(x, 2)
        target = view(y, :, sample_idx)
        free_state, _ = sample_phase!(model, context, view(x, :, sample_idx); reads = config.free_reads, sweeps = config.free_sweeps)
        output = @view free_state[model.output_idxs]
        pred = argmax(class_scores(output, config.output_replicas))
        truth = argmax(class_scores(target, config.output_replicas))
        pred_counts[pred] += 1
        correct += pred == truth
        loss += sum(abs2, target .- output) / 2
    end
    return (; accuracy = correct / size(x, 2), loss = loss / size(x, 2), pred_counts)
end

"""Serialize trainable paper-style parameters."""
function save_model(path::P, model::M) where {P<:AbstractString,M<:PaperMNISTModel}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (;
            weights_0 = model.weights_0,
            weights_12 = model.weights_12,
            weights_2o = model.weights_2o,
            weights_11 = model.weights_11,
            weights_22 = model.weights_22,
            weights_oo = model.weights_oo,
            bias_1 = model.bias_1,
            bias_2 = model.bias_2,
            bias_o = model.bias_o,
            config = model.config,
        ))
    end
    return path
end

"""Plot a run's train/test curves."""
function plot_metrics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    fig = Figure(size = (1200, 760))
    ax_acc = Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "Paper MNIST manager accuracy")
    ax_loss = Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Loss")
    train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
    lines!(ax_acc, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], label = "train", color = :steelblue)
    lines!(ax_acc, [row.epoch for row in rows], [row.test_accuracy for row in rows], label = "test", color = :orange)
    lines!(ax_loss, [row.epoch for row in train_rows], [row.train_loss for row in train_rows], label = "train", color = :steelblue)
    lines!(ax_loss, [row.epoch for row in rows], [row.test_loss for row in rows], label = "test", color = :orange)
    axislegend(ax_acc, position = :rb)
    save(path, fig)
    return path
end

"""Write run settings and manager details."""
function write_settings!(path::P, config::C) where {P<:AbstractString,C<:PaperMNISTManagerConfig}
    open(path, "w") do io
        println(io, "# Local Paper MNIST ProcessManager")
        println(io)
        println(io, "- architecture: `784 -> $(config.hidden1_side^2) -> $(config.hidden2_side^2) -> $(PMNIST_NCLASSES * config.output_replicas)`")
        println(io, "- radius: `$(config.local_radius)`")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- batchsize: `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- free/nudge reads: `$(config.free_reads)` / `$(config.nudge_reads)`")
        println(io, "- free/nudge sweeps: `$(config.free_sweeps)` / `$(config.nudge_sweeps)`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- learning rates W0/W12/W2O/B: `$(config.lr_w0)`, `$(config.lr_w12)`, `$(config.lr_w2o)`, `$(config.lr_b)`")
        println(io, "- temperatures hot/cold/reverse: `$(config.hot_temp)`, `$(config.cold_temp)`, `$(config.reverse_temp)`")
        println(io, "- gradient normalization: `$(config.gradient_normalization)`")
        println(io, "- worker graph adjacency: shared with source graph")
        println(io, "- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`")
    end
    return path
end

"""Run one ProcessManager-backed paper-style MNIST experiment."""
function run_config!(config::C) where {C<:PaperMNISTManagerConfig}
    mkpath(config.outdir)
    write_settings!(joinpath(config.outdir, "local_paper_manager_settings.md"), config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    xtest, ytest = balanced_mnist(:test, config.test_per_class, config)
    source = init_model(config)
    manager = paper_manager(source)
    csv_path = joinpath(config.outdir, "mnist_local_paper_like_ep.csv")
    best_path = joinpath(config.outdir, "best_model.bin")
    final_path = joinpath(config.outdir, "final_model.bin")
    best_accuracy = Ref(-Inf)
    rows = NamedTuple[]

    try
        for epoch in 0:config.epochs
            seconds = 0.0
            train_accuracy = missing
            train_loss = missing
            skipped = missing
            if epoch > 0
                order = Random.shuffle(source.rng, collect(axes(xtrain, 2)))
                total_correct = 0
                total_loss = 0f0
                total_skipped = 0
                total_seen = 0
                seconds = @elapsed begin
                    for first_idx in 1:config.batchsize:length(order)
                        last_idx = min(first_idx + config.batchsize - 1, length(order))
                        jobs = batch_jobs(xtrain, ytrain, @view order[first_idx:last_idx])
                        stats = run_minibatch!(manager, jobs)
                        total_seen += stats.nsamples
                        total_correct += round(Int, stats.accuracy * stats.nsamples)
                        total_loss += stats.loss * stats.nsamples
                        total_skipped += stats.skipped
                    end
                end
                train_accuracy = total_seen == 0 ? 0.0 : total_correct / total_seen
                train_loss = total_seen == 0 ? 0f0 : total_loss / total_seen
                skipped = total_skipped
            end

            test = evaluate(source, xtest, ytest)
            if test.accuracy > best_accuracy[]
                best_accuracy[] = test.accuracy
                save_model(best_path, source)
            end
            row = (;
                timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                epoch,
                seconds,
                train_accuracy,
                train_loss,
                skipped,
                test_accuracy = test.accuracy,
                test_loss = test.loss,
                pred_counts = join(test.pred_counts, "-"),
                best_accuracy = best_accuracy[],
                best_path,
                final_path = epoch == config.epochs ? final_path : "",
            )
            append_row!(csv_path, row)
            push!(rows, row)
            println(row)
            flush(stdout)
        end
        save_model(final_path, source)
        plot_metrics(joinpath(config.outdir, "manager_learning_summary.png"), rows)
        println("saved manager paper-style MNIST run in ", config.outdir)
        return (; config, rows, best_accuracy = best_accuracy[])
    finally
        close(manager)
    end
end

"""Run a radius grid using the ProcessManager paper-style recipe."""
function main()
    base = PaperMNISTManagerConfig()
    base.workers > 0 || throw(ArgumentError("ISING_MNIST_PM_WORKERS must be positive"))
    base.batchsize > 0 || throw(ArgumentError("ISING_MNIST_PM_BATCHSIZE must be positive"))
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers
    radii = parse_int_list(get(ENV, "ISING_MNIST_PM_RADII", string(base.local_radius)), [base.local_radius])
    results = NamedTuple[]
    for radius in radii
        name = "r$(radius)_manager"
        outdir = length(radii) == 1 ? base.outdir : joinpath(base.outdir, "r$(radius)")
        config = copy_config(base; name, local_radius = radius, outdir)
        push!(results, run_config!(config))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
