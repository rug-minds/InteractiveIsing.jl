using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using MLDatasets
using Random
using Serialization
using SparseArrays

const II = IsingLearning.InteractiveIsing
const FT = Float32
const INPUT_SIDE = 28
const INPUT_DIM = INPUT_SIDE^2
const NCLASSES = 10

const HIDDEN1_SIDE = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_H1_SIDE", "28"))
const HIDDEN2_SIDE = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_H2_SIDE", "28"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_OUTPUT_REPLICAS", "4"))
const OUTPUT_LAYOUT = Symbol(get(ENV, "ISING_MNIST_LOCAL_PAPER_OUTPUT_LAYOUT", "factor"))
const LOCAL_RADIUS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_RADIUS", "5"))
const INTERNAL_RADIUS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_INTERNAL_RADIUS", "2"))
const OUTPUT_INTERNAL_RADIUS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_OUTPUT_INTERNAL_RADIUS", "1"))

const TRAIN_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_TRAIN_PER_CLASS", "20"))
const TEST_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_TEST_PER_CLASS", "10"))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_EPOCHS", "5"))
const FREE_READS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_FREE_READS", "3"))
const NUDGE_READS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_NUDGE_READS", "3"))
const FREE_SWEEPS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_FREE_SWEEPS", "50"))
const NUDGE_SWEEPS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_PAPER_NUDGE_SWEEPS", "50"))
const BETA = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_BETA", "5.0"))
const TARGET_ON = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_TARGET_ON", "1.0"))
const TARGET_OFF = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_TARGET_OFF", "-1.0"))

const LR_W0 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_W0", "0.01"))
const LR_W12 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_W12", "0.01"))
const LR_W2O = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_W2O", "0.01"))
const LR_W11 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_W11", "0.001"))
const LR_W22 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_W22", "0.001"))
const LR_WOO = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_WOO", "0.001"))
const LR_B = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_LR_B", "0.001"))
const GAIN_W0 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_W0", "0.5"))
const GAIN_W12 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_W12", "0.25"))
const GAIN_W2O = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_W2O", "0.25"))
const GAIN_W11 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_W11", "0.0"))
const GAIN_W22 = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_W22", "0.0"))
const GAIN_WOO = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_GAIN_WOO", "0.0"))
const INTERNAL_SCALE = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_INTERNAL_SCALE", "0.01"))
const OUTPUT_INTERNAL_SCALE = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_OUTPUT_INTERNAL_SCALE", "0.01"))
const TRAIN_INTERNAL = parse(Bool, get(ENV, "ISING_MNIST_LOCAL_PAPER_TRAIN_INTERNAL", "false"))
const WEIGHT_CLIP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_WEIGHT_CLIP", "1.0"))
const BIAS_CLIP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_BIAS_CLIP", "1.0"))
const APPLIED_BIAS_CLIP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_APPLIED_BIAS_CLIP", "4.0"))

const HOT_TEMP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_HOT_TEMP", "5.0"))
const COLD_TEMP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_COLD_TEMP", "0.01"))
const REVERSE_TEMP = parse(FT, get(ENV, "ISING_MNIST_LOCAL_PAPER_REVERSE_TEMP", "1.0"))
const EVAL_MODE = Symbol(get(ENV, "ISING_MNIST_LOCAL_PAPER_EVAL_MODE", "best_energy"))
const OUTDIR = get(ENV, "ISING_MNIST_LOCAL_PAPER_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_local_paper_like")))
const LOAD_PATH = get(ENV, "ISING_MNIST_LOCAL_PAPER_LOAD", "")

mkpath(OUTDIR)

mutable struct LocalPaperMNIST{G,W0,W12,W2O,W11,W22,WOO,M0,M12,M11,M22,MOO,B1,B2,BO,R}
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

"""Build a local symmetric mask for one rectangular 2D layer."""
function same_layer_mask(rows::Integer, cols::Integer, radius::Integer)
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
    weights = zeros(FT, n, n)
    scale = FT(gain) * sqrt(1f0 / max(n, 1))
    @inbounds for col in 1:n, row in 1:(col - 1)
        (mask[row, col] || mask[col, row]) || continue
        w = scale * (2f0 * rand(rng, FT) - 1f0)
        weights[row, col] = w
        weights[col, row] = w
    end
    return weights
end

"""Apply a mask, clear self-couplings, symmetrize, and clip same-layer weights."""
function constrain_symmetric_weights!(weights::W, mask::M) where {W<:AbstractMatrix,M<:AbstractMatrix{Bool}}
    weights .*= mask
    weights .= 0.5f0 .* (weights .+ transpose(weights))
    @inbounds for idx in axes(weights, 1)
        weights[idx, idx] = 0f0
    end
    weights .= clamp.(weights, -WEIGHT_CLIP, WEIGHT_CLIP)
    return weights
end

"""Append one named-tuple row to a CSV file."""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return a compact display shape for the output replica layer."""
function factor_shape(units::Integer)
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Return the output grid layout used for class-replica spins."""
function output_shape()
    OUTPUT_LAYOUT === :replica_digit && return OUTPUT_REPLICAS, NCLASSES
    return factor_shape(NCLASSES * OUTPUT_REPLICAS)
end

"""Map a lattice coordinate to the common MNIST input coordinate frame."""
function scaled_position(idx::Integer, side::Integer)
    side <= 1 && return FT(1)
    return one(FT) + FT(idx - 1) * FT(INPUT_SIDE - 1) / FT(side - 1)
end

"""Build a Boolean local-connectivity mask between two square image-like layers."""
function local_mask(src_side::Integer, dst_side::Integer, radius::Integer)
    src_n = Int(src_side)^2
    dst_n = Int(dst_side)^2
    mask = falses(src_n, dst_n)
    r = FT(radius)
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

"""Add fixed symmetric local couplings inside one layer."""
function add_internal_edges!(
    graph::G,
    layer_idx::Integer,
    rng::R;
    scale::T,
    radius::Integer,
) where {G,R<:Random.AbstractRNG,T<:AbstractFloat}
    radius <= 0 && return graph
    layer = graph[Int(layer_idx)]
    rows, cols = size(layer)
    idxs = reshape(collect(II.layerrange(layer)), rows, cols)
    A = adj(graph)
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

"""Construct the sampled hidden1-hidden2-output Ising graph."""
function build_graph(rng::R) where {R<:Random.AbstractRNG}
    output_rows, output_cols = output_shape()
    zero_wg = AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    h1 = II.Layer(HIDDEN1_SIDE, HIDDEN1_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    h2 = II.Layer(HIDDEN2_SIDE, HIDDEN2_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, HIDDEN1_SIDE + 2, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, HIDDEN1_SIDE + HIDDEN2_SIDE + 4, 0); periodic = false)
    graph = II.IsingGraph(
        h1,
        zero_wg,
        h2,
        zero_wg,
        out,
        II.Bilinear() + II.MagField(b = g -> II.filltype(Vector, 0f0, II.statelen(g)));
        index_set = g -> II.ToggledIndexSet(g),
    )
    if !TRAIN_INTERNAL
        add_internal_edges!(graph, 1, rng; scale = INTERNAL_SCALE, radius = INTERNAL_RADIUS)
        add_internal_edges!(graph, 2, rng; scale = INTERNAL_SCALE, radius = INTERNAL_RADIUS)
        add_internal_edges!(graph, 3, rng; scale = OUTPUT_INTERNAL_SCALE, radius = OUTPUT_INTERNAL_RADIUS)
    end
    II.temp!(graph, COLD_TEMP)
    return graph
end

"""Initialize trainable local-paper MNIST parameters and install graph couplings."""
function init_model(seed::Integer = 1)
    rng = Random.MersenneTwister(seed)
    graph = build_graph(rng)
    h1n = HIDDEN1_SIDE^2
    h2n = HIDDEN2_SIDE^2
    outn = NCLASSES * OUTPUT_REPLICAS
    mask_0 = local_mask(INPUT_SIDE, HIDDEN1_SIDE, LOCAL_RADIUS)
    mask_12 = local_mask(HIDDEN1_SIDE, HIDDEN2_SIDE, LOCAL_RADIUS)
    output_rows, output_cols = output_shape()
    mask_11 = same_layer_mask(HIDDEN1_SIDE, HIDDEN1_SIDE, INTERNAL_RADIUS)
    mask_22 = same_layer_mask(HIDDEN2_SIDE, HIDDEN2_SIDE, INTERNAL_RADIUS)
    mask_oo = same_layer_mask(output_rows, output_cols, OUTPUT_INTERNAL_RADIUS)
    weights_0 = GAIN_W0 .* (2f0 .* rand(rng, FT, INPUT_DIM, h1n) .- 1f0) .* sqrt(1f0 / INPUT_DIM)
    weights_12 = GAIN_W12 .* (2f0 .* rand(rng, FT, h1n, h2n) .- 1f0) .* sqrt(1f0 / max(h1n, 1))
    weights_2o = GAIN_W2O .* (2f0 .* rand(rng, FT, h2n, outn) .- 1f0) .* sqrt(1f0 / max(h2n, 1))
    weights_11 = masked_symmetric_weights(rng, mask_11, GAIN_W11)
    weights_22 = masked_symmetric_weights(rng, mask_22, GAIN_W22)
    weights_oo = masked_symmetric_weights(rng, mask_oo, GAIN_WOO)
    weights_0 .*= mask_0
    weights_12 .*= mask_12
    model = LocalPaperMNIST(
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
        zeros(FT, h1n),
        zeros(FT, h2n),
        zeros(FT, outn),
        collect(II.layerrange(graph[1])),
        collect(II.layerrange(graph[2])),
        collect(II.layerrange(graph[3])),
        rng,
    )
    sync_graph_couplings!(model)
    return model
end

"""Install trainable couplings and output biases with the package sign convention."""
function sync_graph_couplings!(model::M) where {M<:LocalPaperMNIST}
    A = adj(model.graph)
    if TRAIN_INTERNAL
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

"""Apply per-sample input fields and optional output nudge fields."""
function apply_sample_bias!(
    model::M,
    x::X;
    target = nothing,
    beta::Real = 0,
) where {M<:LocalPaperMNIST,X<:AbstractVector}
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    h1_bias = vec(transpose(x) * model.weights_0) .+ model.bias_1
    b[model.hidden1_idxs] .= .-clamp.(h1_bias, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    b[model.hidden2_idxs] .= .-clamp.(model.bias_2, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    output_bias = isnothing(target) ? model.bias_o : model.bias_o .- FT(beta) .* target
    b[model.output_idxs] .= .-clamp.(output_bias, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    return model
end

"""Reset all sampled spins to random Ising states."""
function randomize_state!(model::M) where {M<:LocalPaperMNIST}
    s = II.state(model.graph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    return model
end

"""Evaluate the package Hamiltonian for choosing the lowest-energy read."""
function graph_energy(model::M) where {M<:LocalPaperMNIST}
    s = II.state(model.graph)
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    A = adj(model.graph)
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
function metropolis_sweep!(algorithm::A, context::C, nactive::Integer) where {A,C}
    for _ in 1:Int(nactive)
        StatefulAlgorithms.step!(algorithm, context)
    end
    return context
end

"""Run a cooling or reverse-annealing schedule."""
function anneal!(model::M, context::C, sweeps::Integer; reverse::Bool = false) where {M<:LocalPaperMNIST,C}
    total = max(Int(sweeps), 1)
    nactive = length(II.state(model.graph))
    for sweep in 1:total
        progress = total == 1 ? 1f0 : FT(sweep - 1) / FT(total - 1)
        T = if reverse
            progress <= 0.5f0 ?
                COLD_TEMP + (progress / 0.5f0) * (REVERSE_TEMP - COLD_TEMP) :
                REVERSE_TEMP + ((progress - 0.5f0) / 0.5f0) * (COLD_TEMP - REVERSE_TEMP)
        else
            HOT_TEMP * (COLD_TEMP / HOT_TEMP)^progress
        end
        II.temp!(model.graph, T)
        metropolis_sweep!(II.Metropolis(), context, nactive)
    end
    II.temp!(model.graph, COLD_TEMP)
    return model
end

"""Sample one free or nudged phase and return the lowest-energy state."""
function sample_phase!(
    model::M,
    x::X;
    target = nothing,
    beta::Real = 0,
    reads::Integer,
    sweeps::Integer,
    initial_state = nothing,
    reverse::Bool = false,
) where {M<:LocalPaperMNIST,X<:AbstractVector}
    context = StatefulAlgorithms.init(II.Metropolis(), (; model = model.graph))
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
function class_scores(output::V) where {V<:AbstractVector}
    scores = zeros(FT, NCLASSES)
    @inbounds for digit in 1:NCLASSES
        first_idx = (digit - 1) * OUTPUT_REPLICAS + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + OUTPUT_REPLICAS - 1))) / OUTPUT_REPLICAS
    end
    return scores
end

"""Load a balanced MNIST subset with `[0, 1]` inputs and repeated output labels."""
function balanced_mnist(split::Symbol, per_class::Integer)
    dataset = split === :train ? MNIST(split = :train) : MNIST(split = :test)
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end
    keep = Int[]
    for digit in 1:NCLASSES
        append!(keep, @view buckets[digit][1:Int(per_class)])
    end
    x = Matrix{FT}(undef, INPUT_DIM, length(keep))
    y = fill(TARGET_OFF, NCLASSES * OUTPUT_REPLICAS, length(keep))
    for (col, idx) in enumerate(keep)
        x[:, col] .= FT.(reshape(images[:, :, idx], :))
        maximum(@view x[:, col]) > 1.5f0 && (x[:, col] ./= 255f0)
        label = Int(labels[idx]) + 1
        first_idx = (label - 1) * OUTPUT_REPLICAS + 1
        y[first_idx:(first_idx + OUTPUT_REPLICAS - 1), col] .= TARGET_ON
    end
    return x, y
end

"""Train one sample with one-sided paper-style EP on the local architecture."""
function train_one!(model::M, x::X, y::Y) where {M<:LocalPaperMNIST,X<:AbstractVector,Y<:AbstractVector}
    free_state, _ = sample_phase!(model, x; reads = FREE_READS, sweeps = FREE_SWEEPS)
    free_h1 = copy(@view free_state[model.hidden1_idxs])
    free_h2 = copy(@view free_state[model.hidden2_idxs])
    free_o = copy(@view free_state[model.output_idxs])
    if all(free_o .== y)
        return (loss = sum(abs2, y .- free_o) / 2, correct = argmax(class_scores(free_o)) == argmax(class_scores(y)), skipped = true)
    end
    nudged_state, _ = sample_phase!(model, x; target = y, beta = BETA, reads = NUDGE_READS, sweeps = NUDGE_SWEEPS, initial_state = free_state, reverse = true)
    nudged_h1 = @view nudged_state[model.hidden1_idxs]
    nudged_h2 = @view nudged_state[model.hidden2_idxs]
    nudged_o = @view nudged_state[model.output_idxs]

    invβ = one(FT) / BETA
    h1_delta = nudged_h1 .- free_h1
    h2_delta = nudged_h2 .- free_h2
    o_delta = nudged_o .- free_o

    model.weights_2o .+= LR_W2O .* (.-(nudged_h2 * transpose(nudged_o) .- free_h2 * transpose(free_o)) .* invβ)
    model.weights_12 .+= LR_W12 .* (.-(nudged_h1 * transpose(nudged_h2) .- free_h1 * transpose(free_h2)) .* invβ)
    model.weights_0 .+= LR_W0 .* (.-(x * transpose(h1_delta)) .* invβ)
    if TRAIN_INTERNAL
        model.weights_11 .+= LR_W11 .* (.-(nudged_h1 * transpose(nudged_h1) .- free_h1 * transpose(free_h1)) .* invβ)
        model.weights_22 .+= LR_W22 .* (.-(nudged_h2 * transpose(nudged_h2) .- free_h2 * transpose(free_h2)) .* invβ)
        model.weights_oo .+= LR_WOO .* (.-(nudged_o * transpose(nudged_o) .- free_o * transpose(free_o)) .* invβ)
        constrain_symmetric_weights!(model.weights_11, model.mask_11)
        constrain_symmetric_weights!(model.weights_22, model.mask_22)
        constrain_symmetric_weights!(model.weights_oo, model.mask_oo)
    end
    model.weights_12 .*= model.mask_12
    model.weights_0 .*= model.mask_0
    model.bias_1 .+= LR_B .* (.-h1_delta .* invβ)
    model.bias_2 .+= LR_B .* (.-h2_delta .* invβ)
    model.bias_o .+= LR_B .* (.-o_delta .* invβ)

    model.weights_0 .= clamp.(model.weights_0, -WEIGHT_CLIP, WEIGHT_CLIP)
    model.weights_12 .= clamp.(model.weights_12, -WEIGHT_CLIP, WEIGHT_CLIP)
    model.weights_2o .= clamp.(model.weights_2o, -WEIGHT_CLIP, WEIGHT_CLIP)
    model.bias_1 .= clamp.(model.bias_1, -BIAS_CLIP, BIAS_CLIP)
    model.bias_2 .= clamp.(model.bias_2, -BIAS_CLIP, BIAS_CLIP)
    model.bias_o .= clamp.(model.bias_o, -BIAS_CLIP, BIAS_CLIP)
    sync_graph_couplings!(model)

    return (loss = sum(abs2, y .- free_o) / 2, correct = argmax(class_scores(free_o)) == argmax(class_scores(y)), skipped = false)
end

"""Average class scores and output states over independent free-phase reads."""
function mean_read_output!(model::M, x::X, reads::Integer) where {M<:LocalPaperMNIST,X<:AbstractVector}
    score_sum = zeros(FT, NCLASSES)
    output_sum = zeros(FT, length(model.output_idxs))
    for _ in 1:Int(reads)
        free_state, _ = sample_phase!(model, x; reads = 1, sweeps = FREE_SWEEPS)
        output = @view free_state[model.output_idxs]
        score_sum .+= class_scores(output)
        output_sum .+= output
    end
    inv_reads = one(FT) / FT(reads)
    return score_sum .* inv_reads, output_sum .* inv_reads
end

"""Evaluate balanced accuracy and output loss with free-phase sampling."""
function evaluate(model::M, x::X, y::Y) where {M<:LocalPaperMNIST,X<:AbstractMatrix,Y<:AbstractMatrix}
    correct = 0
    loss = 0f0
    pred_counts = zeros(Int, NCLASSES)
    for sample_idx in axes(x, 2)
        target = view(y, :, sample_idx)
        if EVAL_MODE === :mean_reads
            scores, output = mean_read_output!(model, view(x, :, sample_idx), FREE_READS)
            pred = argmax(scores)
        else
            free_state, _ = sample_phase!(model, view(x, :, sample_idx); reads = FREE_READS, sweeps = FREE_SWEEPS)
            output = @view free_state[model.output_idxs]
            pred = argmax(class_scores(output))
        end
        truth = argmax(class_scores(target))
        pred_counts[pred] += 1
        correct += pred == truth
        loss += sum(abs2, target .- output) / 2
    end
    return (; accuracy = correct / size(x, 2), loss = loss / size(x, 2), pred_counts)
end

"""Serialize trainable local-paper parameters."""
function save_model(path::P, model::M) where {P<:AbstractString,M<:LocalPaperMNIST}
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
        ))
    end
    return path
end

"""Load trainable local-paper parameters from a checkpoint."""
function load_model!(model::M, path::P) where {M<:LocalPaperMNIST,P<:AbstractString}
    params = open(deserialize, path)
    model.weights_0 .= params.weights_0
    model.weights_12 .= params.weights_12
    model.weights_2o .= params.weights_2o
    hasproperty(params, :weights_11) && (model.weights_11 .= params.weights_11)
    hasproperty(params, :weights_22) && (model.weights_22 .= params.weights_22)
    hasproperty(params, :weights_oo) && (model.weights_oo .= params.weights_oo)
    model.bias_1 .= params.bias_1
    model.bias_2 .= params.bias_2
    model.bias_o .= params.bias_o
    sync_graph_couplings!(model)
    return model
end

"""Run the local paper-style MNIST experiment."""
function main()
    xtrain, ytrain = EPOCHS > 0 ? balanced_mnist(:train, TRAIN_PER_CLASS) : (Matrix{FT}(undef, INPUT_DIM, 0), Matrix{FT}(undef, NCLASSES * OUTPUT_REPLICAS, 0))
    xtest, ytest = balanced_mnist(:test, TEST_PER_CLASS)
    model = init_model(2468)
    isempty(LOAD_PATH) || load_model!(model, LOAD_PATH)
    csv_path = joinpath(OUTDIR, "mnist_local_paper_like_ep.csv")
    best_path = joinpath(OUTDIR, "best_model.bin")
    final_path = joinpath(OUTDIR, "final_model.bin")
    best_accuracy = Ref(-Inf)

    open(joinpath(OUTDIR, "local_paper_like_settings.md"), "w") do io
        println(io, "# Local Paper-Like MNIST EP")
        println(io, "- architecture: `784 -> $(HIDDEN1_SIDE^2) -> $(HIDDEN2_SIDE^2) -> $(NCLASSES * OUTPUT_REPLICAS)`")
        println(io, "- output layout: `$(OUTPUT_LAYOUT)`")
        println(io, "- local radius: `$(LOCAL_RADIUS)`, internal radius: `$(INTERNAL_RADIUS)`, output internal radius: `$(OUTPUT_INTERNAL_RADIUS)`")
        println(io, "- train/test per class: `$(TRAIN_PER_CLASS)` / `$(TEST_PER_CLASS)`")
        println(io, "- free/nudge reads: `$(FREE_READS)` / `$(NUDGE_READS)`")
        println(io, "- free/nudge sweeps: `$(FREE_SWEEPS)` / `$(NUDGE_SWEEPS)`")
        println(io, "- beta: `$(BETA)`")
        println(io, "- target on/off: `$(TARGET_ON)`, `$(TARGET_OFF)`")
        println(io, "- eval mode: `$(EVAL_MODE)`")
        println(io, "- train internal couplings: `$(TRAIN_INTERNAL)`")
        println(io, "- learning rates W0/W12/W2O/W11/W22/WOO/B: `$(LR_W0)`, `$(LR_W12)`, `$(LR_W2O)`, `$(LR_W11)`, `$(LR_W22)`, `$(LR_WOO)`, `$(LR_B)`")
        println(io, "- init gains W0/W12/W2O/W11/W22/WOO: `$(GAIN_W0)`, `$(GAIN_W12)`, `$(GAIN_W2O)`, `$(GAIN_W11)`, `$(GAIN_W22)`, `$(GAIN_WOO)`")
        println(io, "- loaded checkpoint: `$(isempty(LOAD_PATH) ? "none" : LOAD_PATH)`")
    end

    for epoch in 0:EPOCHS
        seconds = 0.0
        train_accuracy = missing
        train_loss = missing
        skipped = missing
        if epoch > 0
            order = Random.shuffle(model.rng, collect(axes(xtrain, 2)))
            ncorrect = 0
            total_loss = 0f0
            nskipped = 0
            seconds = @elapsed begin
                for sample_idx in order
                    stats = train_one!(model, view(xtrain, :, sample_idx), view(ytrain, :, sample_idx))
                    ncorrect += stats.correct
                    total_loss += stats.loss
                    nskipped += stats.skipped
                end
            end
            train_accuracy = ncorrect / length(order)
            train_loss = total_loss / length(order)
            skipped = nskipped
        end
        test = evaluate(model, xtest, ytest)
        if test.accuracy > best_accuracy[]
            best_accuracy[] = test.accuracy
            save_model(best_path, model)
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
            final_path = epoch == EPOCHS ? final_path : "",
        )
        append_row!(csv_path, row)
        println(row)
        flush(stdout)
    end
    save_model(final_path, model)
    println("Saved local paper-like MNIST EP run in ", OUTDIR)
end

main()
