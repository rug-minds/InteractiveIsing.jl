using Dates

const PMNIST_BOOT_T0 = time()
const PMNIST_BOOT_PROGRESS = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_PROGRESS", "true")))

"""Print a startup checkpoint before the experiment config has been constructed."""
function boot_progress(message::S; t0::T = PMNIST_BOOT_T0) where {S<:AbstractString,T<:Real}
    PMNIST_BOOT_PROGRESS || return nothing
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] bootstrap ", message, " elapsed_s=", round(time() - Float64(t0); digits = 3))
    flush(stdout)
    return nothing
end

boot_progress("script started")
t_pkg_import = time()
using Pkg
boot_progress("Pkg loaded"; t0 = t_pkg_import)
t_activate = time()
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
boot_progress("project activated"; t0 = t_activate)

t_imports = time()
using IsingLearning
using IsingLearning.InteractiveIsing
using MLDatasets
using Optimisers
using Random
using Serialization
using SparseArrays
using Statistics
boot_progress("dependencies loaded"; t0 = t_imports)

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const PMNIST_FT = Float32
const PMNIST_INPUT_SIDE = 28
const PMNIST_INPUT_DIM = PMNIST_INPUT_SIDE^2
const PMNIST_NCLASSES = 10

Base.@kwdef struct LocalMNISTManagerConfig{T<:AbstractFloat,S<:AbstractString}
    name::S = get(ENV, "ISING_MNIST_PM_NAME", "local_manager")
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_PM_WORKERS", "32"))
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_PM_EPOCHS", "200"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_PM_BATCHSIZE", "32"))
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
    β::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_BETA", "5.0"))
    target_on::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_TARGET_ON", "1.0"))
    target_off::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_TARGET_OFF", "-1.0"))
    lr_w0::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W0", "0.004"))
    lr_w12::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W12", "0.004"))
    lr_w2o::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W2O", "0.004"))
    lr_w11::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W11", "0.001"))
    lr_w22::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_W22", "0.001"))
    lr_woo::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_WOO", "0.001"))
    lr_b::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LR_B", "0.0004"))
    optimizer::S = lowercase(get(ENV, "ISING_MNIST_PM_OPTIMIZER", "adam"))
    gain_w0::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W0", "0.5"))
    gain_w12::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W12", "0.25"))
    gain_w2o::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W2O", "0.25"))
    gain_w11::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W11", "0.0"))
    gain_w22::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_W22", "0.0"))
    gain_woo::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_GAIN_WOO", "0.0"))
    internal_scale::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_INTERNAL_SCALE", "0.0"))
    output_internal_scale::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_OUTPUT_INTERNAL_SCALE", "0.0"))
    train_internal::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_TRAIN_INTERNAL", "false")))
    weight_clip::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_WEIGHT_CLIP", "1.0"))
    bias_clip::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_BIAS_CLIP", "1.0"))
    applied_bias_clip::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_APPLIED_BIAS_CLIP", "4.0"))
    hot_temp::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_HOT_TEMP", "5.0"))
    cold_temp::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_COLD_TEMP", "0.01"))
    reverse_temp::T = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_REVERSE_TEMP", "1.0"))
    gradient_normalization::Symbol = Symbol(get(ENV, "ISING_MNIST_PM_GRADIENT_NORMALIZATION", "mean"))
    progress::Bool = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_PROGRESS", "true")))
    progress_every::Int = parse(Int, get(ENV, "ISING_MNIST_PM_PROGRESS_EVERY", "10"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_PM_SEED", "2468"))
    outdir::S = get(
        ENV,
        "ISING_MNIST_PM_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", "mnist_local_manager_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
end

mutable struct LocalMNISTModel{C,G,E,R}
    config::C
    graph::G
    edge_groups::E
    input_idxs::Vector{Int}
    hidden1_idxs::Vector{Int}
    hidden2_idxs::Vector{Int}
    output_idxs::Vector{Int}
    rng::R
end

struct LocalMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

mutable struct LocalMNISTManagerState{M,G,P,O}
    model::M
    batch_gradient::G
    optimizer_gradient::G
    params::Base.RefValue{P}
    opt_state::O
    update_idx::Base.RefValue{Int}
    nsamples::Base.RefValue{Int}
    ncorrect::Base.RefValue{Int}
    nskipped::Base.RefValue{Int}
    total_loss::Base.RefValue{PMNIST_FT}
end

"""Return the optional checkpoint used to initialize model parameters."""
function resume_checkpoint_path()
    return get(ENV, "ISING_MNIST_PM_RESUME_CHECKPOINT", "")
end

"""Construct the configured single-step dynamics algorithm for phase routines."""
function mnist_dynamics_algorithm()
    name = lowercase(strip(get(ENV, "ISING_MNIST_PM_DYNAMICS", "metropolis")))
    if name == "metropolis"
        return II.Metropolis()
    elseif name == "local_langevin"
        stepsize = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LANGEVIN_STEPSIZE", "0.1"))
        adjusted = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_LANGEVIN_ADJUSTED", "false")))
        order = Symbol(get(ENV, "ISING_MNIST_PM_LANGEVIN_ORDER", "random"))
        return II.LocalLangevin(; stepsize, adjusted, order)
    elseif name == "global_langevin"
        stepsize = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LANGEVIN_STEPSIZE", "0.1"))
        adjusted = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_LANGEVIN_ADJUSTED", "false")))
        return II.GlobalLangevin(; stepsize, adjusted)
    elseif name == "block_langevin"
        stepsize = parse(PMNIST_FT, get(ENV, "ISING_MNIST_PM_LANGEVIN_STEPSIZE", "0.1"))
        adjusted = parse(Bool, lowercase(get(ENV, "ISING_MNIST_PM_LANGEVIN_ADJUSTED", "false")))
        block_size = parse(Int, get(ENV, "ISING_MNIST_PM_LANGEVIN_BLOCK_SIZE", "256"))
        return II.BlockLangevin(; stepsize, adjusted, block_size)
    end
    throw(ArgumentError("unknown ISING_MNIST_PM_DYNAMICS=`$name`"))
end

"""Parse a comma-separated integer list."""
function parse_int_list(value::S, default::V) where {S<:AbstractString,V<:AbstractVector{Int}}
    isempty(strip(value)) && return default
    return [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

"""Return a shallow config copy with selected fields replaced."""
function copy_config(config::C; kwargs...) where {C<:LocalMNISTManagerConfig}
    fields = Dict{Symbol,Any}(field => getfield(config, field) for field in fieldnames(C))
    for (field, value) in kwargs
        fields[field] = value
    end
    return LocalMNISTManagerConfig(; fields...)
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

"""Print a timestamped progress checkpoint and flush stdout immediately."""
function progress_log(config::C, message::S; t0 = nothing, kwargs...) where {C<:LocalMNISTManagerConfig,S<:AbstractString}
    config.progress || return nothing
    details = String[]
    isnothing(t0) || push!(details, string("elapsed_s=", round(time() - Float64(t0); digits = 3)))
    for (key, value) in pairs(kwargs)
        push!(details, string(key, "=", value))
    end
    suffix = isempty(details) ? "" : " " * join(details, " ")
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message, suffix)
    flush(stdout)
    return nothing
end

"""Return true when an indexed progress checkpoint should be printed."""
function should_log_progress(config::C, idx::I, total::J) where {C<:LocalMNISTManagerConfig,I<:Integer,J<:Integer}
    idx == 1 && return true
    idx == total && return true
    return config.progress_every > 0 && idx % config.progress_every == 0
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

"""Return the first magnetic field, which owns trainable base biases."""
function base_magfield(isinggraph::G) where {G}
    for hterm in II.hamiltonians(isinggraph.hamiltonian)
        hterm isa II.MagField && return hterm
    end
    error("local MNIST graph has no base MagField")
end

"""Return the second magnetic field, used for worker-local input/nudge fields."""
function sample_magfield(isinggraph::G) where {G}
    seen = 0
    for hterm in II.hamiltonians(isinggraph.hamiltonian)
        if hterm isa II.MagField
            seen += 1
            seen == 2 && return hterm
        end
    end
    error("local MNIST graph has no worker-local sample MagField")
end

"""Return a lookup from stored directed sparse-adjacency coordinates to nzval pointers."""
function adjacency_pointer_lookup(A::AType) where {AType}
    lookup = Dict{Tuple{Int,Int},Int}()
    rows = SparseArrays.rowvals(A)
    colptr = SparseArrays.getcolptr(A)
    @inbounds for col in 1:(length(colptr) - 1)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            lookup[(rows[ptr], col)] = ptr
        end
    end
    return lookup
end

"""Return the stored sparse-adjacency pointer for one existing directed coupling."""
function stored_ptr(lookup::D, row::I, col::J) where {D<:AbstractDict,I<:Integer,J<:Integer}
    return get(lookup, (Int(row), Int(col))) do
        throw(ArgumentError("missing stored adjacency entry at row $(row), col $(col)"))
    end
end

"""Build pointer metadata for a trainable bipartite edge block already stored in `J`."""
function edge_group(lookup::D, src_idxs::S, dst_idxs::V, mask::M) where {D<:AbstractDict,S<:AbstractVector,V<:AbstractVector,M<:AbstractMatrix{Bool}}
    srcpos = Int[]
    dstpos = Int[]
    forward = Int[]
    reverse = Int[]
    @inbounds for src in eachindex(src_idxs), dst in eachindex(dst_idxs)
        mask[src, dst] || continue
        push!(srcpos, src)
        push!(dstpos, dst)
        push!(forward, stored_ptr(lookup, dst_idxs[dst], src_idxs[src]))
        push!(reverse, stored_ptr(lookup, src_idxs[src], dst_idxs[dst]))
    end
    return (; srcpos, dstpos, forward, reverse)
end

"""Build pointer metadata for a symmetric same-layer edge block already stored in `J`."""
function same_layer_edge_group(lookup::D, idxs::V, mask::M) where {D<:AbstractDict,V<:AbstractVector,M<:AbstractMatrix{Bool}}
    srcpos = Int[]
    dstpos = Int[]
    forward = Int[]
    reverse = Int[]
    @inbounds for src in eachindex(idxs), dst in (src + 1):lastindex(idxs)
        (mask[src, dst] || mask[dst, src]) || continue
        push!(srcpos, src)
        push!(dstpos, dst)
        push!(forward, stored_ptr(lookup, idxs[dst], idxs[src]))
        push!(reverse, stored_ptr(lookup, idxs[src], idxs[dst]))
    end
    return (; srcpos, dstpos, forward, reverse)
end

"""Return empty edge metadata for disabled optional connectivity groups."""
function empty_edge_group()
    return (; srcpos = Int[], dstpos = Int[], forward = Int[], reverse = Int[])
end

"""Install random local couplings directly into `J` between two layers."""
function add_bipartite_edges!(
    graph::G,
    src_idxs::S,
    dst_idxs::D,
    mask::M,
    rng::R,
    gain::T,
) where {G,S<:AbstractVector,D<:AbstractVector,M<:AbstractMatrix{Bool},R<:Random.AbstractRNG,T<:Real}
    A = II.adj(graph)
    scale = PMNIST_FT(gain) * sqrt(1f0 / PMNIST_FT(max(length(src_idxs), 1)))
    @inbounds for src in eachindex(src_idxs), dst in eachindex(dst_idxs)
        mask[src, dst] || continue
        w = scale * (2f0 * rand(rng, PMNIST_FT) - 1f0)
        A[dst_idxs[dst], src_idxs[src]] = w
        A[src_idxs[src], dst_idxs[dst]] = w
    end
    return graph
end

"""Install random local couplings directly into `J` within one layer."""
function add_same_layer_edges!(
    graph::G,
    idxs::V,
    mask::M,
    rng::R,
    gain::T,
) where {G,V<:AbstractVector,M<:AbstractMatrix{Bool},R<:Random.AbstractRNG,T<:Real}
    A = II.adj(graph)
    scale = PMNIST_FT(gain) * sqrt(1f0 / PMNIST_FT(max(length(idxs), 1)))
    @inbounds for src in eachindex(idxs), dst in (src + 1):lastindex(idxs)
        (mask[src, dst] || mask[dst, src]) || continue
        w = scale * (2f0 * rand(rng, PMNIST_FT) - 1f0)
        A[idxs[dst], idxs[src]] = w
        A[idxs[src], idxs[dst]] = w
    end
    return graph
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

"""Construct the sampled input-hidden-readout Ising graph."""
function sampled_graph(
    config::C,
    rng::R;
    shared_adj = nothing,
    shared_base_bias = nothing,
) where {C<:LocalMNISTManagerConfig,R<:Random.AbstractRNG}
    output_rows, output_cols = factor_shape(PMNIST_NCLASSES * config.output_replicas)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    input = II.Layer(PMNIST_INPUT_SIDE, PMNIST_INPUT_SIDE, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    h1 = II.Layer(config.hidden1_side, config.hidden1_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + 2, 0); periodic = false)
    h2 = II.Layer(config.hidden2_side, config.hidden2_side, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + config.hidden1_side + 4, 0); periodic = false)
    out = II.Layer(output_rows, output_cols, II.StateSet(-1f0, 1f0), II.Discrete(), II.Coords(0, PMNIST_INPUT_SIDE + config.hidden1_side + config.hidden2_side + 6, 0); periodic = false)
    base_bias = isnothing(shared_base_bias) ? (g -> II.filltype(Vector, 0f0, II.statelen(g))) : shared_base_bias
    graph = II.IsingGraph(
        input,
        zero_wg,
        h1,
        zero_wg,
        h2,
        zero_wg,
        out,
        II.Bilinear() +
            II.MagField(b = base_bias) +
            II.MagField(b = g -> II.filltype(Vector, 0f0, II.statelen(g)));
        precision = PMNIST_FT,
        adj = shared_adj,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.off!(graph.index_set, 1)
    II.temp!(graph, config.cold_temp)
    return graph
end

"""Initialize trainable local-MNIST parameters directly in `J` and `MagField.b`."""
function init_model(config::C, seed::I = config.seed) where {C<:LocalMNISTManagerConfig,I<:Integer}
    t_model = time()
    rng = Random.MersenneTwister(Int(seed))
    t_graph = time()
    graph = sampled_graph(config, rng)
    progress_log(config, "model graph shell initialized"; t0 = t_graph, states = II.statelen(graph))
    outn = PMNIST_NCLASSES * config.output_replicas
    output_rows, output_cols = factor_shape(outn)
    input_idxs = collect(II.layerrange(graph[1]))
    hidden1_idxs = collect(II.layerrange(graph[2]))
    hidden2_idxs = collect(II.layerrange(graph[3]))
    output_idxs = collect(II.layerrange(graph[4]))

    t_masks = time()
    input_hidden_connectivity = local_mask(PMNIST_INPUT_SIDE, config.hidden1_side, config.local_radius)
    hidden_hidden_connectivity = local_mask(config.hidden1_side, config.hidden2_side, config.local_radius)
    hidden1_internal_connectivity = same_layer_mask(config.hidden1_side, config.hidden1_side, config.internal_radius)
    hidden2_internal_connectivity = same_layer_mask(config.hidden2_side, config.hidden2_side, config.internal_radius)
    output_internal_connectivity = same_layer_mask(output_rows, output_cols, config.output_internal_radius)
    progress_log(
        config,
        "model connectivity masks built";
        t0 = t_masks,
        input_hidden = count(input_hidden_connectivity),
        hidden_hidden = count(hidden_hidden_connectivity),
        hidden_output = length(hidden2_idxs) * length(output_idxs),
    )

    t_edges = time()
    add_bipartite_edges!(graph, input_idxs, hidden1_idxs, input_hidden_connectivity, rng, config.gain_w0)
    add_bipartite_edges!(graph, hidden1_idxs, hidden2_idxs, hidden_hidden_connectivity, rng, config.gain_w12)
    add_bipartite_edges!(graph, hidden2_idxs, output_idxs, trues(length(hidden2_idxs), length(output_idxs)), rng, config.gain_w2o)
    config.train_internal && add_same_layer_edges!(graph, hidden1_idxs, hidden1_internal_connectivity, rng, config.gain_w11)
    config.train_internal && add_same_layer_edges!(graph, hidden2_idxs, hidden2_internal_connectivity, rng, config.gain_w22)
    config.train_internal && add_same_layer_edges!(graph, output_idxs, output_internal_connectivity, rng, config.gain_woo)
    progress_log(config, "model J entries initialized"; t0 = t_edges, stored_entries = length(SparseArrays.nonzeros(II.adj(graph))))

    A = II.adj(graph)
    t_lookup = time()
    lookup = adjacency_pointer_lookup(A)
    progress_log(config, "model J pointer lookup built"; t0 = t_lookup, pointers = length(lookup))
    t_groups = time()
    edge_groups = (;
        input_hidden = edge_group(lookup, input_idxs, hidden1_idxs, input_hidden_connectivity),
        hidden_hidden = edge_group(lookup, hidden1_idxs, hidden2_idxs, hidden_hidden_connectivity),
        hidden_output = edge_group(lookup, hidden2_idxs, output_idxs, trues(length(hidden2_idxs), length(output_idxs))),
        hidden1_internal = config.train_internal ? same_layer_edge_group(lookup, hidden1_idxs, hidden1_internal_connectivity) : empty_edge_group(),
        hidden2_internal = config.train_internal ? same_layer_edge_group(lookup, hidden2_idxs, hidden2_internal_connectivity) : empty_edge_group(),
        output_internal = config.train_internal ? same_layer_edge_group(lookup, output_idxs, output_internal_connectivity) : empty_edge_group(),
    )
    progress_log(config, "model edge groups built"; t0 = t_groups)
    progress_log(config, "model initialized"; t0 = t_model, stored_entries = length(SparseArrays.nonzeros(A)))

    return LocalMNISTModel(config, graph, edge_groups, input_idxs, hidden1_idxs, hidden2_idxs, output_idxs, rng)
end

"""Load a serialized manager checkpoint and install compatible graph parameters."""
function resume_model!(model::M, path::P) where {M<:LocalMNISTModel,P<:AbstractString}
    isempty(path) && return model
    isfile(path) || throw(ArgumentError("resume checkpoint does not exist: `$path`"))
    saved = open(path, "r") do io
        deserialize(io)
    end

    hasproperty(saved, :w) && hasproperty(saved, :b) ||
        throw(ArgumentError("resume checkpoint must contain graph parameters `w` and `b`"))
    length(saved.w) == length(SparseArrays.nonzeros(II.adj(model.graph))) ||
        throw(ArgumentError("resume checkpoint has $(length(saved.w)) J entries, model has $(length(SparseArrays.nonzeros(II.adj(model.graph))))"))
    length(saved.b) == length(base_magfield(model.graph).b) ||
        throw(ArgumentError("resume checkpoint has $(length(saved.b)) b entries, model has $(length(base_magfield(model.graph).b))"))
    SparseArrays.nonzeros(II.adj(model.graph)) .= saved.w
    base_magfield(model.graph).b .= saved.b
    return model
end

"""Create a worker model with local state/fields and shared `J`/base-field pointers."""
function worker_model(source::M, worker_idx::I) where {M<:LocalMNISTModel,I<:Integer}
    rng = Random.MersenneTwister(source.config.seed + 10_000 + Int(worker_idx))
    graph = sampled_graph(source.config, rng; shared_adj = II.adj(source.graph), shared_base_bias = base_magfield(source.graph).b)
    return LocalMNISTModel(
        source.config,
        graph,
        source.edge_groups,
        collect(II.layerrange(graph[1])),
        collect(II.layerrange(graph[2])),
        collect(II.layerrange(graph[3])),
        collect(II.layerrange(graph[4])),
        rng,
    )
end

"""Allocate a gradient buffer matching all trainable parameter arrays."""
function gradient_buffer(model::M) where {M<:LocalMNISTModel}
    return (;
        w = zeros(PMNIST_FT, length(SparseArrays.nonzeros(II.adj(model.graph)))),
        b = zeros(PMNIST_FT, length(base_magfield(model.graph).b)),
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

"""Return the trainable array tuple owned by a local-MNIST model."""
function trainable_params(model::M) where {M<:LocalMNISTModel}
    return (;
        w = copy(SparseArrays.nonzeros(II.adj(model.graph))),
        b = copy(base_magfield(model.graph).b),
    )
end

"""Construct the configured Optimisers.jl rule for one parameter group."""
function optimizer_rule(config::C, lr::T) where {C<:LocalMNISTManagerConfig,T<:Real}
    optimizer = lowercase(config.optimizer)
    if optimizer == "adam"
        return Optimisers.Adam(PMNIST_FT(lr))
    elseif optimizer in ("descent", "sgd")
        return Optimisers.Descent(PMNIST_FT(lr))
    end
    throw(ArgumentError("unknown ISING_MNIST_PM_OPTIMIZER=`$(config.optimizer)`; use `adam` or `descent`"))
end

"""Create one optimizer state per trainable parameter array."""
function optimizer_states(config::C, params::P) where {C<:LocalMNISTManagerConfig,P<:NamedTuple}
    return (;
        w = Optimisers.setup(optimizer_rule(config, config.lr_w12), params.w),
        b = Optimisers.setup(optimizer_rule(config, config.lr_b), params.b),
    )
end

"""Write the scaled optimizer gradient while preserving the contrastive update direction."""
function write_optimizer_gradient!(
    dest::D,
    src::S,
    config::C,
    nsamples::I,
) where {D<:NamedTuple,S<:NamedTuple,C<:LocalMNISTManagerConfig,I<:Integer}
    scale = config.gradient_normalization === :mean ? inv(PMNIST_FT(max(nsamples, 1))) :
        config.gradient_normalization === :sum ? one(PMNIST_FT) :
        throw(ArgumentError("gradient_normalization must be :sum or :mean, got $(config.gradient_normalization)"))

    for field in propertynames(dest)
        getproperty(dest, field) .= .-scale .* getproperty(src, field)
    end
    return dest
end

"""Install updated parameter arrays and enforce graph-level clipping."""
function install_params!(model::M, params::P) where {M<:LocalMNISTModel,P<:NamedTuple}
    config = model.config
    SparseArrays.nonzeros(II.adj(model.graph)) .= clamp.(params.w, -config.weight_clip, config.weight_clip)
    base_magfield(model.graph).b .= clamp.(params.b, -config.bias_clip, config.bias_clip)
    return model
end

"""Apply one optimizer update to all trainable parameter arrays."""
function apply_optimizer_update!(manager::M) where {M<:Processes.ProcessManager}
    params = manager.state.params[]
    gradient = manager.state.optimizer_gradient
    opt_state = manager.state.opt_state

    state_w, w = Optimisers.update(opt_state.w, params.w, gradient.w)
    state_b, b = Optimisers.update(opt_state.b, params.b, gradient.b)

    manager.state.opt_state = (; w = state_w, b = state_b)
    manager.state.params[] = (; w, b)
    manager.state.update_idx[] += 1
    install_params!(manager.state.model, manager.state.params[])
    manager.state.params[] = trainable_params(manager.state.model)
    return manager.state.model
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

"""Update worker-local counters after one contrastive sample."""
function update_worker_stats!(
    nsamples::Base.RefValue{I},
    ncorrect::Base.RefValue{J},
    nskipped::Base.RefValue{K},
    total_loss::Base.RefValue{T},
    stats::S,
) where {I<:Integer,J<:Integer,K<:Integer,T<:Real,S}
    nsamples[] += 1
    ncorrect[] += stats.correct ? 1 : 0
    nskipped[] += stats.skipped ? 1 : 0
    total_loss[] += stats.loss
    return nothing
end

"""Return the number of single dynamics steps used for a configured sweep count."""
function phase_steps(model::M, sweeps::I) where {M<:LocalMNISTModel,I<:Integer}
    return max(1, Int(sweeps) * length(II.state(model.graph)))
end

"""Build one reusable free-phase temperature-scheduled dynamics step."""
function free_phase_step_algorithm(dynamics_algorithm::D, temperature_algorithm::T) where {D,T}
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias free_temperature = temperature_algorithm

        free_temperature(dynamics.model)
        dynamics()
    end
end

"""Build one reusable nudged-phase temperature-scheduled dynamics step."""
function nudged_phase_step_algorithm(dynamics_algorithm::D, temperature_algorithm::T) where {D,T}
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias nudge_temperature = temperature_algorithm

        nudge_temperature(dynamics.model)
        dynamics()
    end
end

"""Build the reusable free-phase routine."""
function free_phase_algorithm(
    dynamics_algorithm::D,
    temperature_algorithm::T,
    steps::I,
) where {D,T,I<:Integer}
    phase_step = free_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias phase_step = phase_step
        @state mnist_model
        @state x
        @state free_state
        @state free_best_energy
        @state rng

        RandomizeGraphState!(dynamics.model, rng)
        install_sample_bias!(mnist_model, x)
        @repeat steps phase_step()
        CaptureBestEnergyState!(dynamics.model, free_best_energy, free_state)
    end
end

"""Build the reusable nudged-phase routine."""
function nudged_phase_algorithm(
    dynamics_algorithm::D,
    temperature_algorithm::T,
    steps::I,
) where {D,T,I<:Integer}
    phase_step = nudged_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    return Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias phase_step = phase_step
        @state mnist_model
        @state x
        @state y
        @state free_state
        @state nudged_state
        @state nudged_best_energy

        SetGraphState!(dynamics.model, free_state)
        install_nudged_sample_bias!(mnist_model, x, y)
        @repeat steps phase_step()
        CaptureBestEnergyState!(dynamics.model, nudged_best_energy, nudged_state)
    end
end

"""Build the reusable LoopAlgorithm that runs free/nudged phases and fills buffers."""
function local_worker_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    free_steps = max(1, config.free_sweeps * Int(nstates))
    nudge_steps = max(1, config.nudge_sweeps * Int(nstates))
    free_reads = max(1, config.free_reads)
    nudge_reads = max(1, config.nudge_reads)
    free_temperature = GeometricTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    nudge_temperature = ReverseAnnealTemperatureSchedule(; cold_T = config.cold_temp, peak_T = config.reverse_temp, n_steps = nudge_steps)
    free_phase = free_phase_algorithm(dynamics_algorithm, free_temperature, free_steps)
    nudged_phase = nudged_phase_algorithm(dynamics_algorithm, nudge_temperature, nudge_steps)
    return Processes.@Routine begin
        @alias free_phase = free_phase
        @alias nudged_phase = nudged_phase
        @state mnist_model
        @state x
        @state y
        @state gradient
        @state free_state
        @state nudged_state
        @state free_best_energy
        @state nudged_best_energy
        @state rng
        @state nsamples
        @state ncorrect
        @state nskipped
        @state total_loss

        ResetBestEnergyCapture!(free_best_energy, free_state)
        @repeat free_reads free_phase()

        ResetBestEnergyCapture!(nudged_best_energy, nudged_state)
        @repeat nudge_reads nudged_phase()

        stats = finish_contrastive_sample!(gradient, mnist_model, x, y, free_state, nudged_state)
        update_worker_stats!(nsamples, ncorrect, nskipped, total_loss, stats)
    end
end

"""Create one reusable manager-owned local-MNIST worker."""
function local_worker(source::M, worker_idx::I, algorithm::A) where {M<:LocalMNISTModel,I<:Integer,A}
    log_worker = should_log_progress(source.config, worker_idx, source.config.workers)
    t_worker = time()
    t_model = time()
    model = worker_model(source, worker_idx)
    log_worker && progress_log(source.config, "worker model initialized"; t0 = t_model, worker = worker_idx)
    graph_state = II.state(model.graph)
    t_process = time()
    proc = Processes.Process(
        algorithm,
        Processes.Init(:_state;
            mnist_model = model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            y = zeros(PMNIST_FT, PMNIST_NCLASSES * source.config.output_replicas),
            gradient = gradient_buffer(model),
            free_state = similar(graph_state),
            nudged_state = similar(graph_state),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            nudged_best_energy = Ref(PMNIST_FT(Inf)),
            rng = model.rng,
            nsamples = Ref(0),
            ncorrect = Ref(0),
            nskipped = Ref(0),
            total_loss = Ref(0f0),
        ),
        Processes.Init(:dynamics; model = model.graph);
        repeat = 1,
    )
    log_worker && progress_log(source.config, "worker process initialized"; t0 = t_process, worker = worker_idx)
    log_worker && progress_log(source.config, "worker initialized"; t0 = t_worker, worker = worker_idx)
    return proc
end

"""Create the ProcessManager for local-MNIST minibatches."""
function local_manager(source::M) where {M<:LocalMNISTModel}
    t_manager = time()
    progress_log(source.config, "manager construction started"; workers = source.config.workers)
    params = trainable_params(source)
    state = LocalMNISTManagerState(
        source,
        gradient_buffer(source),
        gradient_buffer(source),
        Ref(params),
        optimizer_states(source.config, params),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0f0),
    )
    dynamics_algorithm = mnist_dynamics_algorithm()
    t_resolve = time()
    worker_algorithm = Processes.resolve(local_worker_algorithm(deepcopy(dynamics_algorithm), source.config, length(II.state(source.graph))))
    progress_log(source.config, "worker algorithm resolved"; t0 = t_resolve)
    recipe = (;
        makeworker = (idx, manager) -> begin
            should_log_progress(manager.config, idx, manager.config.workers) &&
                progress_log(manager.config, "manager constructing worker"; worker = idx, workers = manager.config.workers)
            worker = local_worker(manager.state.model, idx, worker_algorithm)
            should_log_progress(manager.config, idx, manager.config.workers) &&
                progress_log(manager.config, "manager constructed worker"; worker = idx, workers = manager.config.workers)
            return worker
        end,
        prepare! = (slot, job, manager) -> begin
            ctx = worker_context(slot.worker)
            ctx.x .= job.x
            ctx.y .= job.y
            Processes.resetworker!(slot)
            return nothing
        end,
        flush! = manager -> flush_manager_buffers!(manager),
    )
    manager = Processes.ProcessManager(
        recipe;
        nworkers = source.config.workers,
        config = source.config,
        state,
        flush_policy = Processes.FlushAtEnd(),
        worker_init = Processes.MakeEachWorker(),
        poll_interval = 0.0,
        job_type = LocalMNISTJob{Vector{PMNIST_FT},Vector{PMNIST_FT}},
    )
    progress_log(source.config, "manager constructed"; t0 = t_manager, workers = source.config.workers)
    return manager
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

"""Merge all worker-local buffers into the manager state."""
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
function run_minibatch!(manager::M, jobs::J; log_progress::Bool = true) where {M<:Processes.ProcessManager,J<:AbstractVector}
    t_clear = time()
    clear_manager_buffers!(manager)
    log_progress && progress_log(manager.config, "minibatch buffers cleared"; t0 = t_clear, jobs = length(jobs))
    t_run = time()
    log_progress && progress_log(manager.config, "minibatch manager run started"; jobs = length(jobs), workers = manager.config.workers)
    Processes.run!(manager, jobs, Processes.Dynamic())
    log_progress && progress_log(manager.config, "minibatch manager run finished"; t0 = t_run, jobs = length(jobs))
    t_update = time()
    write_optimizer_gradient!(manager.state.optimizer_gradient, manager.state.batch_gradient, manager.config, manager.state.nsamples[])
    apply_optimizer_update!(manager)
    log_progress && progress_log(manager.config, "minibatch optimizer update finished"; t0 = t_update, nsamples = manager.state.nsamples[], skipped = manager.state.nskipped[])
    return (;
        nsamples = manager.state.nsamples[],
        accuracy = manager.state.nsamples[] == 0 ? 0.0 : manager.state.ncorrect[] / manager.state.nsamples[],
        loss = manager.state.nsamples[] == 0 ? 0f0 : manager.state.total_loss[] / manager.state.nsamples[],
        skipped = manager.state.nskipped[],
    )
end

"""Install per-sample input fields and optional output nudge fields."""
function install_sample_bias!(
    model::M,
    x::X,
    target = nothing,
    beta::Real = 0,
) where {M<:LocalMNISTModel,X<:AbstractVector}
    length(x) == length(model.input_idxs) ||
        throw(DimensionMismatch("input length $(length(x)) does not match input layer length $(length(model.input_idxs))"))
    II.off!(model.graph.index_set, 1)
    fill!(II.state(model.graph[1]), 0f0)

    sample_b = sample_magfield(model.graph).b
    fill!(sample_b, 0f0)
    A = II.adj(model.graph)
    rows = SparseArrays.rowvals(A)
    nz = SparseArrays.nonzeros(A)

    # The input layer is inactive; its couplings in J are converted to a local field.
    @inbounds for (xpos, input_idx) in enumerate(model.input_idxs)
        xval = x[xpos]
        for ptr in SparseArrays.getcolptr(A)[input_idx]:(SparseArrays.getcolptr(A)[input_idx + 1] - 1)
            sample_b[rows[ptr]] += nz[ptr] * xval
        end
    end
    if !isnothing(target)
        @inbounds sample_b[model.output_idxs] .+= PMNIST_FT(beta) .* target
    end
    sample_b .= clamp.(sample_b, -model.config.applied_bias_clip, model.config.applied_bias_clip)
    return model
end

"""Install per-sample input fields and the configured output nudge."""
function install_nudged_sample_bias!(
    model::M,
    x::X,
    target::Y,
) where {M<:LocalMNISTModel,X<:AbstractVector,Y<:AbstractVector}
    return install_sample_bias!(model, x, target, model.config.β)
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

"""Accumulate local-MNIST contrastive gradients from captured free/nudged states."""
function accumulate_local_contrastive_gradient!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    free_state::F,
    nudged_state::N,
) where {G<:NamedTuple,M<:LocalMNISTModel,X<:AbstractVector,Y<:AbstractVector,F<:AbstractVector,N<:AbstractVector}
    config = model.config
    free_h1 = @view free_state[model.hidden1_idxs]
    free_h2 = @view free_state[model.hidden2_idxs]
    free_o = @view free_state[model.output_idxs]
    nudged_h1 = @view nudged_state[model.hidden1_idxs]
    nudged_h2 = @view nudged_state[model.hidden2_idxs]
    nudged_o = @view nudged_state[model.output_idxs]

    invβ = one(PMNIST_FT) / config.β
    h1_delta = nudged_h1 .- free_h1
    h2_delta = nudged_h2 .- free_h2
    o_delta = nudged_o .- free_o

    # Store update directions for the graph parameters. The manager negates once
    # before passing them to Optimisers.jl, whose convention is descent.
    input_hidden = model.edge_groups.input_hidden
    @inbounds for idx in eachindex(input_hidden.forward)
        update = x[input_hidden.srcpos[idx]] * h1_delta[input_hidden.dstpos[idx]] * invβ
        gradient.w[input_hidden.forward[idx]] += update
        gradient.w[input_hidden.reverse[idx]] += update
    end

    hidden_hidden = model.edge_groups.hidden_hidden
    @inbounds for idx in eachindex(hidden_hidden.forward)
        src = hidden_hidden.srcpos[idx]
        dst = hidden_hidden.dstpos[idx]
        update = (nudged_h1[src] * nudged_h2[dst] - free_h1[src] * free_h2[dst]) * invβ
        gradient.w[hidden_hidden.forward[idx]] += update
        gradient.w[hidden_hidden.reverse[idx]] += update
    end

    hidden_output = model.edge_groups.hidden_output
    @inbounds for idx in eachindex(hidden_output.forward)
        src = hidden_output.srcpos[idx]
        dst = hidden_output.dstpos[idx]
        update = (nudged_h2[src] * nudged_o[dst] - free_h2[src] * free_o[dst]) * invβ
        gradient.w[hidden_output.forward[idx]] += update
        gradient.w[hidden_output.reverse[idx]] += update
    end

    if config.train_internal
        h1_internal = model.edge_groups.hidden1_internal
        @inbounds for idx in eachindex(h1_internal.forward)
            src = h1_internal.srcpos[idx]
            dst = h1_internal.dstpos[idx]
            update = (nudged_h1[src] * nudged_h1[dst] - free_h1[src] * free_h1[dst]) * invβ
            gradient.w[h1_internal.forward[idx]] += update
            gradient.w[h1_internal.reverse[idx]] += update
        end
        h2_internal = model.edge_groups.hidden2_internal
        @inbounds for idx in eachindex(h2_internal.forward)
            src = h2_internal.srcpos[idx]
            dst = h2_internal.dstpos[idx]
            update = (nudged_h2[src] * nudged_h2[dst] - free_h2[src] * free_h2[dst]) * invβ
            gradient.w[h2_internal.forward[idx]] += update
            gradient.w[h2_internal.reverse[idx]] += update
        end
        output_internal = model.edge_groups.output_internal
        @inbounds for idx in eachindex(output_internal.forward)
            src = output_internal.srcpos[idx]
            dst = output_internal.dstpos[idx]
            update = (nudged_o[src] * nudged_o[dst] - free_o[src] * free_o[dst]) * invβ
            gradient.w[output_internal.forward[idx]] += update
            gradient.w[output_internal.reverse[idx]] += update
        end
    end
    gradient.b[model.hidden1_idxs] .+= h1_delta .* invβ
    gradient.b[model.hidden2_idxs] .+= h2_delta .* invβ
    gradient.b[model.output_idxs] .+= o_delta .* invβ
    return gradient
end

"""Compute stats and fill the worker gradient buffer for one completed sample."""
function finish_contrastive_sample!(
    gradient::G,
    model::M,
    x::X,
    y::Y,
    free_state::F,
    nudged_state::N,
) where {G<:NamedTuple,M<:LocalMNISTModel,X<:AbstractVector,Y<:AbstractVector,F<:AbstractVector,N<:AbstractVector}
    config = model.config
    free_o = @view free_state[model.output_idxs]
    correct = argmax(class_scores(free_o, config.output_replicas)) == argmax(class_scores(y, config.output_replicas))
    loss = sum(abs2, y .- free_o) / 2
    if all(free_o .== y)
        return (; loss, correct, skipped = true)
    end
    accumulate_local_contrastive_gradient!(gradient, model, x, y, free_state, nudged_state)
    return (; loss, correct, skipped = false)
end

"""Load a balanced MNIST subset with `[0, 1]` inputs and repeated output labels."""
function balanced_mnist(split::Symbol, per_class::I, config::C) where {I<:Integer,C<:LocalMNISTManagerConfig}
    dataset = split === :train ? MLDatasets.MNIST(split = :train) :
        split === :test ? MLDatasets.MNIST(split = :test) :
        throw(ArgumentError("split must be :train or :test"))
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:PMNIST_NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end
    keep = Int[]
    rng = Random.MersenneTwister(hash((config.seed, split, Int(per_class))))
    for digit in 1:PMNIST_NCLASSES
        length(buckets[digit]) >= Int(per_class) ||
            throw(ArgumentError("split $(split) has only $(length(buckets[digit])) samples for digit $(digit - 1)"))

        # Shuffle each class bucket once so reduced runs stay representative.
        digit_indices = copy(buckets[digit])
        Random.shuffle!(rng, digit_indices)
        append!(keep, @view digit_indices[1:Int(per_class)])
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
    jobs = LocalMNISTJob{Vector{PMNIST_FT},Vector{PMNIST_FT}}[]
    for sample_idx in indices
        push!(jobs, LocalMNISTJob(copy(view(x, :, sample_idx)), copy(view(y, :, sample_idx))))
    end
    return jobs
end

"""Build a free-phase LoopAlgorithm for validation and checkpoint evaluation."""
function validation_free_phase_algorithm(
    dynamics_algorithm::D,
    config::C,
    nstates::I,
) where {D,C<:LocalMNISTManagerConfig,I<:Integer}
    free_steps = max(1, config.free_sweeps * Int(nstates))
    free_reads = max(1, config.free_reads)
    free_temperature = GeometricTemperatureSchedule(; start_T = config.hot_temp, stop_T = config.cold_temp, n_steps = free_steps)
    free_phase = free_phase_algorithm(dynamics_algorithm, free_temperature, free_steps)
    return Processes.@Routine begin
        @alias free_phase = free_phase
        @state mnist_model
        @state x
        @state free_state
        @state free_best_energy
        @state rng

        ResetBestEnergyCapture!(free_best_energy, free_state)
        @repeat free_reads free_phase()
    end
end

"""Create one reusable process for free-phase validation sampling."""
function free_phase_process(model::M, dynamics_algorithm::D) where {M<:LocalMNISTModel,D}
    graph_state = II.state(model.graph)
    algorithm = Processes.resolve(validation_free_phase_algorithm(deepcopy(dynamics_algorithm), model.config, length(graph_state)))
    return Processes.Process(
        algorithm,
        Processes.Init(:_state;
            mnist_model = model,
            x = zeros(PMNIST_FT, PMNIST_INPUT_DIM),
            free_state = similar(graph_state),
            free_best_energy = Ref(PMNIST_FT(Inf)),
            rng = model.rng,
        ),
        Processes.Init(:dynamics; model = model.graph);
        repeat = 1,
    )
end

"""Evaluate balanced accuracy and output loss with free-phase sampling."""
function evaluate(model::M, x::X, y::Y) where {M<:LocalMNISTModel,X<:AbstractMatrix,Y<:AbstractMatrix}
    config = model.config
    process = free_phase_process(model, mnist_dynamics_algorithm())
    context = Processes.context(process)._state
    correct = 0
    loss = 0f0
    pred_counts = zeros(Int, PMNIST_NCLASSES)
    for sample_idx in axes(x, 2)
        target = view(y, :, sample_idx)
        context.x .= view(x, :, sample_idx)
        Processes.reset!(process)
        run(process)
        wait(process)
        output = @view context.free_state[model.output_idxs]
        pred = argmax(class_scores(output, config.output_replicas))
        truth = argmax(class_scores(target, config.output_replicas))
        pred_counts[pred] += 1
        correct += pred == truth
        loss += sum(abs2, target .- output) / 2
    end
    return (; accuracy = correct / size(x, 2), loss = loss / size(x, 2), pred_counts)
end

"""Serialize trainable parameters."""
function save_model(path::P, model::M) where {P<:AbstractString,M<:LocalMNISTModel}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (;
            w = copy(SparseArrays.nonzeros(II.adj(model.graph))),
            b = copy(base_magfield(model.graph).b),
            config = model.config,
        ))
    end
    return path
end

"""Load CairoMakie only when plots are written."""
function ensure_cairomakie()
    mod = @__MODULE__
    isdefined(mod, :CairoMakie) || Base.invokelatest(Core.eval, mod, :(using CairoMakie))
    return Base.invokelatest(getfield, mod, :CairoMakie)
end

"""Plot a run's train/test accuracy and loss curves without timing diagnostics."""
function plot_metrics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    CM = ensure_cairomakie()
    fig = CM.Figure(size = (1200, 760))
    ax_acc = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "Single-hidden local MNIST accuracy")
    ax_loss = CM.Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Mean squared output error")
    train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
    if !isempty(train_rows)
        CM.lines!(ax_acc, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], label = "train", color = :steelblue)
        CM.lines!(ax_loss, [row.epoch for row in train_rows], [row.train_loss for row in train_rows], label = "train", color = :steelblue)
    end
    CM.lines!(ax_acc, [row.epoch for row in rows], [row.test_accuracy for row in rows], label = "test", color = :orange)
    CM.lines!(ax_loss, [row.epoch for row in rows], [row.test_loss for row in rows], label = "test", color = :orange)
    CM.axislegend(ax_acc, position = :rb)
    CM.save(path, fig)
    return path
end

"""Plot timing and skipped-sample diagnostics separately from learning curves."""
function plot_diagnostics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    CM = ensure_cairomakie()
    mkpath(dirname(path))
    timed_rows = [row for row in rows if row.epoch > 0]
    isempty(timed_rows) && return nothing

    fig = CM.Figure(size = (1000, 640))
    ax_time = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "seconds", title = "Epoch Time")
    ax_skip = CM.Axis(fig[2, 1], xlabel = "epoch", ylabel = "skipped samples", title = "Skipped Samples")
    CM.lines!(ax_time, [row.epoch for row in timed_rows], [row.seconds for row in timed_rows], color = :steelblue)
    CM.scatter!(ax_time, [row.epoch for row in timed_rows], [row.seconds for row in timed_rows], color = :steelblue)

    skipped_rows = [row for row in timed_rows if !ismissing(row.skipped)]
    if !isempty(skipped_rows)
        CM.lines!(ax_skip, [row.epoch for row in skipped_rows], [row.skipped for row in skipped_rows], color = :orange)
        CM.scatter!(ax_skip, [row.epoch for row in skipped_rows], [row.skipped for row in skipped_rows], color = :orange)
    end

    CM.save(path, fig)
    return path
end

"""Write run settings and manager details."""
function write_settings!(path::P, config::C) where {P<:AbstractString,C<:LocalMNISTManagerConfig}
    open(path, "w") do io
        println(io, "# Single-Hidden Local MNIST")
        println(io)
        println(io, "- architecture: inactive input layer `784`, sampled layers `$(config.hidden1_side^2) -> $(config.hidden2_side^2) -> $(PMNIST_NCLASSES * config.output_replicas)`")
        println(io, "- radius: `$(config.local_radius)`")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- batchsize: `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- free/nudge reads: `$(config.free_reads)` / `$(config.nudge_reads)`")
        println(io, "- free/nudge sweeps: `$(config.free_sweeps)` / `$(config.nudge_sweeps)`")
        println(io, "- beta: `$(config.β)`")
        println(io, "- optimizer: `$(config.optimizer)`")
        println(io, "- learning rates W0/W12/W2O/B: `$(config.lr_w0)`, `$(config.lr_w12)`, `$(config.lr_w2o)`, `$(config.lr_b)`")
        println(io, "- temperatures hot/cold/reverse: `$(config.hot_temp)`, `$(config.cold_temp)`, `$(config.reverse_temp)`")
        println(io, "- gradient normalization: `$(config.gradient_normalization)`")
        println(io, "- progress logging: `$(config.progress)`, every `$(config.progress_every)` indexed steps")
        checkpoint = resume_checkpoint_path()
        !isempty(checkpoint) && println(io, "- resumed from: `$(checkpoint)`")
        println(io, "- worker graph adjacency: shared with source graph")
        println(io, "- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`")
    end
    return path
end

"""Run one ProcessManager-backed local MNIST experiment."""
function run_config!(config::C) where {C<:LocalMNISTManagerConfig}
    t_run = time()
    progress_log(
        config,
        "run started";
        name = config.name,
        radius = config.local_radius,
        epochs = config.epochs,
        workers = config.workers,
        threads = Threads.nthreads(),
        outdir = config.outdir,
    )
    t_setup = time()
    mkpath(config.outdir)
    write_settings!(joinpath(config.outdir, "settings.md"), config)
    progress_log(config, "run output initialized"; t0 = t_setup)
    t_train = time()
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    progress_log(config, "training data loaded"; t0 = t_train, samples = size(xtrain, 2), per_class = config.train_per_class)
    t_test = time()
    xtest, ytest = balanced_mnist(:test, config.test_per_class, config)
    progress_log(config, "test data loaded"; t0 = t_test, samples = size(xtest, 2), per_class = config.test_per_class)
    t_source = time()
    source = init_model(config)
    progress_log(config, "source model ready"; t0 = t_source)
    checkpoint = resume_checkpoint_path()
    if !isempty(checkpoint)
        t_resume = time()
        progress_log(config, "checkpoint resume started"; path = checkpoint)
        resume_model!(source, checkpoint)
        progress_log(config, "checkpoint resume finished"; t0 = t_resume, path = checkpoint)
    end
    t_manager = time()
    manager = local_manager(source)
    progress_log(config, "manager ready"; t0 = t_manager, workers = config.workers)
    csv_path = joinpath(config.outdir, "metrics.csv")
    best_path = joinpath(config.outdir, "best_params.bin")
    final_path = joinpath(config.outdir, "final_params.bin")
    best_accuracy = Ref(-Inf)
    rows = NamedTuple[]

    try
        for epoch in 0:config.epochs
            t_epoch = time()
            progress_log(config, "epoch started"; epoch)
            seconds = 0.0
            train_accuracy = missing
            train_loss = missing
            skipped = missing
            if epoch > 0
                order = Random.shuffle(source.rng, collect(axes(xtrain, 2)))
                nbatches = cld(length(order), config.batchsize)
                progress_log(config, "epoch training started"; epoch, batches = nbatches, samples = length(order))
                total_correct = 0
                total_loss = 0f0
                total_skipped = 0
                total_seen = 0
                seconds = @elapsed begin
                    for (batch_idx, first_idx) in enumerate(1:config.batchsize:length(order))
                        last_idx = min(first_idx + config.batchsize - 1, length(order))
                        log_batch = should_log_progress(config, batch_idx, nbatches)
                        if log_batch
                            progress_log(
                                config,
                                "batch build started";
                                epoch,
                                batch = batch_idx,
                                batches = nbatches,
                                first = first_idx,
                                last = last_idx,
                            )
                        end
                        t_jobs = time()
                        jobs = batch_jobs(xtrain, ytrain, @view order[first_idx:last_idx])
                        log_batch && progress_log(config, "batch jobs built"; t0 = t_jobs, epoch, batch = batch_idx, jobs = length(jobs))
                        t_batch = time()
                        log_batch && progress_log(config, "batch run started"; epoch, batch = batch_idx, jobs = length(jobs))
                        stats = run_minibatch!(manager, jobs; log_progress = log_batch)
                        log_batch && progress_log(
                            config,
                            "batch run finished";
                            t0 = t_batch,
                            epoch,
                            batch = batch_idx,
                            nsamples = stats.nsamples,
                            accuracy = round(stats.accuracy; digits = 4),
                            loss = round(stats.loss; digits = 4),
                            skipped = stats.skipped,
                        )
                        total_seen += stats.nsamples
                        total_correct += round(Int, stats.accuracy * stats.nsamples)
                        total_loss += stats.loss * stats.nsamples
                        total_skipped += stats.skipped
                    end
                end
                train_accuracy = total_seen == 0 ? 0.0 : total_correct / total_seen
                train_loss = total_seen == 0 ? 0f0 : total_loss / total_seen
                skipped = total_skipped
                progress_log(
                    config,
                    "epoch training finished";
                    epoch,
                    elapsed_s = round(seconds; digits = 3),
                    train_accuracy = round(train_accuracy; digits = 4),
                    train_loss = round(train_loss; digits = 4),
                    skipped,
                )
            end

            t_eval = time()
            progress_log(config, "epoch evaluation started"; epoch, samples = size(xtest, 2))
            test = evaluate(source, xtest, ytest)
            progress_log(
                config,
                "epoch evaluation finished";
                t0 = t_eval,
                epoch,
                test_accuracy = round(test.accuracy; digits = 4),
                test_loss = round(test.loss; digits = 4),
            )
            if test.accuracy > best_accuracy[]
                best_accuracy[] = test.accuracy
                t_save_best = time()
                save_model(best_path, source)
                progress_log(config, "best checkpoint saved"; t0 = t_save_best, epoch, best_accuracy = round(best_accuracy[]; digits = 4), path = best_path)
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
            progress_log(config, "epoch finished"; t0 = t_epoch, epoch)
        end
        t_final = time()
        save_model(final_path, source)
        progress_log(config, "final checkpoint saved"; t0 = t_final, path = final_path)
        t_plots = time()
        plot_metrics(joinpath(config.outdir, "progress.png"), rows)
        plot_diagnostics(joinpath(config.outdir, "diagnostics", "epoch_time.png"), rows)
        progress_log(config, "plots saved"; t0 = t_plots)
        progress_log(config, "run finished"; t0 = t_run, best_accuracy = round(best_accuracy[]; digits = 4), outdir = config.outdir)
        println("saved local MNIST manager run in ", config.outdir)
        return (; config, rows, best_accuracy = best_accuracy[])
    finally
        progress_log(config, "manager closing")
        close(manager)
        progress_log(config, "manager closed")
    end
end

"""Run a radius grid using the ProcessManager local-MNIST recipe."""
function main()
    base = LocalMNISTManagerConfig()
    base.workers > 0 || throw(ArgumentError("ISING_MNIST_PM_WORKERS must be positive"))
    base.batchsize > 0 || throw(ArgumentError("ISING_MNIST_PM_BATCHSIZE must be positive"))
    Threads.nthreads() < base.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = base.workers
    radii = parse_int_list(get(ENV, "ISING_MNIST_PM_RADII", "1,2,3,4,5,6,7,8,9,10"), collect(1:10))
    root_outdir = haskey(ENV, "ISING_MNIST_PM_OUTDIR") ? base.outdir :
        joinpath(@__DIR__, "experiments", "current", "radius_1_to_10_e$(base.epochs)_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    results = NamedTuple[]
    for radius in radii
        name = "r$(radius)"
        outdir = joinpath(root_outdir, name)
        config = copy_config(base; name, local_radius = radius, outdir)
        push!(results, run_config!(config))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
