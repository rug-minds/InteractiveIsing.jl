using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

println("[bootstrap] activated project")
flush(stdout)

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using LinearAlgebra: mul!
using MLDatasets
using Optimisers
using ProgressMeter
using Random
using Serialization
using SparseArrays
using Statistics

println("[bootstrap] loaded packages")
flush(stdout)

const II = IsingLearning.InteractiveIsing
const StatefulAlgorithms = II.StatefulAlgorithms
const FT = Float32
const INPUT_DIM = IsingLearning.MNIST_INPUT_DIM
const NCLASSES = IsingLearning.MNIST_NCLASSES

"""Build a date-first run directory name so `experiments/current` stays sortable."""
function default_run_dirname(tag::S) where {S<:AbstractString}
    return Dates.format(now(), "yyyymmdd_HHMMSS") * "_" * String(tag)
end

Base.@kwdef struct InputFieldMNISTConfig{T<:AbstractFloat,S<:AbstractString}
    workers::Int = parse(Int, get(ENV, "ISING_MNIST_IF_WORKERS", "32"))
    epochs::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EPOCHS", "200"))
    batchsize::Int = parse(Int, get(ENV, "ISING_MNIST_IF_BATCHSIZE", "128"))
    scheduler::S = get(ENV, "ISING_MNIST_IF_SCHEDULER", "spawn")
    nudge_mode::S = get(ENV, "ISING_MNIST_IF_NUDGE_MODE", "one_sided")
    nudge_temp_schedule::S = get(ENV, "ISING_MNIST_IF_NUDGE_TEMP_SCHEDULE", "fixed")
    chunk_size::Int = parse(Int, get(ENV, "ISING_MNIST_IF_CHUNK_SIZE", "0"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_PER_CLASS", "5421"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TEST_PER_CLASS", "892"))
    train_eval_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS", "100"))
    eval_every::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EVAL_EVERY", "1"))
    progress_every_batches::Int = parse(Int, get(ENV, "ISING_MNIST_IF_PROGRESS_EVERY_BATCHES", "25"))
    early_stop_decline_epochs::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS", "2"))
    early_stop_decline_batches::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EARLY_STOP_DECLINE_BATCHES", "2"))
    early_stop_min_batches::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EARLY_STOP_MIN_BATCHES", "25"))
    hidden::Int = parse(Int, get(ENV, "ISING_MNIST_IF_HIDDEN", "120"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS", "4"))
    sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_SWEEPS", "500"))
    β::T = parse(FT, get(ENV, "ISING_MNIST_IF_BETA", "5.0"))
    nudge_tau::T = parse(FT, get(ENV, "ISING_MNIST_IF_NUDGE_TAU", "0.25"))
    lr::T = parse(FT, get(ENV, "ISING_MNIST_IF_LR", "0.003"))
    w_normalization::S = get(ENV, "ISING_MNIST_IF_W_NORMALIZATION", "none")
    w_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_W_NORM", "1.0"))
    project_output_bias_prior::Bool = parse(Bool, get(ENV, "ISING_MNIST_IF_PROJECT_OUTPUT_BIAS_PRIOR", "false"))
    positive_target_weight::T = parse(FT, get(ENV, "ISING_MNIST_IF_POSITIVE_TARGET_WEIGHT", string(NCLASSES - 1)))
    negative_target_weight::T = parse(FT, get(ENV, "ISING_MNIST_IF_NEGATIVE_TARGET_WEIGHT", "1.0"))
    w_input_normalization::S = get(ENV, "ISING_MNIST_IF_W_INPUT_NORMALIZATION", "none")
    w_input_row_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_W_INPUT_ROW_NORM", "0.14"))
    temp::T = parse(FT, get(ENV, "ISING_MNIST_IF_TEMP", "0.0001"))
    nudge_temp_peak::T = parse(FT, get(ENV, "ISING_MNIST_IF_NUDGE_TEMP_PEAK", string(temp)))
    stepsize::T = parse(FT, get(ENV, "ISING_MNIST_IF_STEPSIZE", "0.5"))
    weight_scale::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_SCALE", "0.005"))
    weight_decay::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_DECAY", "0.0001"))
    adaptive_weight_decay::Bool = parse(Bool, get(ENV, "ISING_MNIST_IF_ADAPTIVE_WEIGHT_DECAY", "false"))
    adaptive_w_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_ADAPTIVE_W_NORM", "2.0"))
    adaptive_w_input_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_ADAPTIVE_W_INPUT_NORM", "5.0"))
    adaptive_decay_gain::T = parse(FT, get(ENV, "ISING_MNIST_IF_ADAPTIVE_DECAY_GAIN", "0.0005"))
    adaptive_max_decay::T = parse(FT, get(ENV, "ISING_MNIST_IF_ADAPTIVE_MAX_DECAY", "0.003"))
    skip_worker_close::Bool = parse(Bool, get(ENV, "ISING_MNIST_IF_SKIP_WORKER_CLOSE", "false"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_IF_SEED", "20260526"))
    resume_from::S = get(ENV, "ISING_MNIST_IF_RESUME_FROM", "")
    resume_epoch::Int = parse(Int, get(ENV, "ISING_MNIST_IF_RESUME_EPOCH", "-1"))
    reset_opt_state_on_resume::Bool = parse(Bool, get(ENV, "ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME", "false"))
    gradient_quality_min_cosine::T = parse(FT, get(ENV, "ISING_MNIST_IF_GRADIENT_QUALITY_MIN_COSINE", "0.5"))
    outdir::S = get(
        ENV,
        "ISING_MNIST_IF_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("mnist_784_120_40_softplus_margin_gradient_quality_adam")),
    )
end

struct InputFieldMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

struct InputFieldMNISTJobBuffer{J<:AbstractVector}
    jobs::J
end

mutable struct InputFieldMNISTManagerState{L,G,P,B,O,W,R}
    layer::L
    source_graph::G
    params::Base.RefValue{P}
    batch_gradient::B
    nsamples::Base.RefValue{Int}
    gradient_quality_total::Base.RefValue{Int}
    gradient_quality_accepted::Base.RefValue{Int}
    gradient_quality_rejected::Base.RefValue{Int}
    gradient_quality_sum_cosine::Base.RefValue{Float64}
    gradient_quality_sum_angle_degrees::Base.RefValue{Float64}
    opt_state::O
    input_hidden_w::Base.RefValue{W}
    reverse_sparse_ptrs::R
end

mutable struct InputFieldMNISTEvalManagerState{L,G,P,W}
    layer::L
    source_graph::G
    nsamples::Base.RefValue{Int}
    ncorrect::Base.RefValue{Int}
    total_loss::Base.RefValue{FT}
    pred_counts::P
    input_hidden_w::Base.RefValue{W}
end

mutable struct InputFieldMNISTSession{C,XT,YT,XE,YE,XS,YS,M,V,TJ,EJ,R}
    config::C
    xtrain::XT
    ytrain::YT
    xtrain_eval::XE
    ytrain_eval::YE
    xtest::XS
    ytest::YS
    manager::M
    validator::V
    train_jobs::TJ
    eval_jobs::EJ
    rows::R
    best_accuracy::Base.RefValue{Float64}
    csv_path::String
    best_path::String
    latest_path::String
    final_path::String
end

"""Return `true` when the run should restore optimizer and graph state from disk."""
function has_resume_checkpoint(config::C) where {C<:InputFieldMNISTConfig}
    return !isempty(config.resume_from)
end

"""Materialize one config as a named tuple so selected fields can be overridden cleanly."""
function config_namedtuple(config::C) where {C<:InputFieldMNISTConfig}
    names = fieldnames(typeof(config))
    return NamedTuple{names}(Tuple(getfield(config, name) for name in names))
end

"""Copy one config while replacing selected keyword fields."""
function updated_config(config::C; kwargs...) where {C<:InputFieldMNISTConfig}
    return InputFieldMNISTConfig(; config_namedtuple(config)..., kwargs...)
end

"""Return `true` when the run uses the symmetric `+β/-β` contrastive estimator."""
function uses_symmetric_nudging(config::C) where {C<:InputFieldMNISTConfig}
    return String(config.nudge_mode) == "symmetric"
end

"""Return `true` when per-sample forward/symmetric gradient agreement gates updates."""
function uses_gradient_quality_nudging(config::C) where {C<:InputFieldMNISTConfig}
    return String(config.nudge_mode) == "gradient_quality"
end

"""Return the per-sample contrastive scale denominator for the selected nudge mode."""
function contrastive_beta_scale(config::C) where {C<:InputFieldMNISTConfig}
    beta = FT(config.β)
    return (uses_symmetric_nudging(config) || uses_gradient_quality_nudging(config)) ? FT(2) * beta : beta
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

"""Rewrite one CSV file from a retained vector of named-tuple rows."""
function write_rows_csv!(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    rm(path; force = true)
    for row in rows
        append_row!(path, row)
    end
    return path
end

"""Return the number of sampled spins stepped by the MNIST dynamics."""
function active_units(graph::G) where {G}
    length(graph) == 2 && return II.nstates(graph)
    total = 0
    for layer_idx in 2:length(graph)
        total += length(II.layerrange(graph[layer_idx]))
    end
    return total
end

"""Set every parameter-gradient array to zero."""
function clear_buffer!(buffer::B) where {B}
    fill!(buffer.w, zero(eltype(buffer.w)))
    fill!(buffer.b, zero(eltype(buffer.b)))
    hasproperty(buffer, :w_input) && fill!(buffer.w_input, zero(eltype(buffer.w_input)))
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""Add one gradient buffer into another buffer with matching fields."""
function add_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.w
    dest.b .+= src.b
    hasproperty(dest, :w_input) && (dest.w_input .+= src.w_input)
    hasproperty(dest, :α) && (dest.α .+= src.α)
    return dest
end

"""Scale a gradient buffer in place."""
function scale_buffer!(buffer::B, scale::T) where {B,T<:Real}
    buffer.w .*= scale
    buffer.b .*= scale
    hasproperty(buffer, :w_input) && (buffer.w_input .*= scale)
    hasproperty(buffer, :α) && (buffer.α .*= scale)
    return buffer
end

"""Return worker-local counters for accepted and rejected gradient estimates."""
function gradient_quality_stats()
    return (;
        total = Ref(0),
        accepted = Ref(0),
        rejected = Ref(0),
        sum_cosine = Ref(0.0),
        sum_angle_degrees = Ref(0.0),
    )
end

"""Reset a gradient-quality statistics object in-place."""
function reset_gradient_quality_stats!(stats::S) where {S}
    stats.total[] = 0
    stats.accepted[] = 0
    stats.rejected[] = 0
    stats.sum_cosine[] = 0.0
    stats.sum_angle_degrees[] = 0.0
    return stats
end

"""Return the dot product between two full MNIST gradient buffers."""
function gradient_buffer_dot(a::A, b::B) where {A,B}
    value = sum(a.w .* b.w) + sum(a.b .* b.b)
    hasproperty(a, :w_input) && (value += sum(a.w_input .* b.w_input))
    hasproperty(a, :α) && (value += sum(a.α .* b.α))
    return Float64(value)
end

"""Return the squared norm of one full MNIST gradient buffer."""
function gradient_buffer_norm2(buffer::B) where {B}
    value = sum(abs2, buffer.w) + sum(abs2, buffer.b)
    hasproperty(buffer, :w_input) && (value += sum(abs2, buffer.w_input))
    hasproperty(buffer, :α) && (value += sum(abs2, buffer.α))
    return Float64(value)
end

"""Return the cosine overlap and angle disagreement between two gradient buffers."""
function gradient_buffer_cosine_angle(a::A, b::B) where {A,B}
    anorm2 = gradient_buffer_norm2(a)
    bnorm2 = gradient_buffer_norm2(b)
    if anorm2 <= 0.0 || bnorm2 <= 0.0
        return (; cosine = NaN, angle_degrees = NaN)
    end
    cosine = clamp(gradient_buffer_dot(a, b) / sqrt(anorm2 * bnorm2), -1.0, 1.0)
    return (; cosine, angle_degrees = acos(cosine) * 180.0 / pi)
end

"""Accumulate one gradient quality decision into worker-local counters."""
function record_gradient_quality!(stats::S, cosine::Real, angle_degrees::Real, accepted::Bool) where {S}
    stats.total[] += 1
    if accepted
        stats.accepted[] += 1
    else
        stats.rejected[] += 1
    end
    stats.sum_cosine[] += isfinite(cosine) ? Float64(cosine) : -1.0
    stats.sum_angle_degrees[] += isfinite(angle_degrees) ? Float64(angle_degrees) : 180.0
    return stats
end

"""Return an `Float64` Euclidean norm for a dense or sparse parameter container."""
function parameter_norm(x::X) where {X}
    return sqrt(sum(abs2, x))
end

"""Return parameter and last-minibatch gradient norms for collapse diagnostics."""
function training_norms(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    params = manager.state.params[]
    grad = manager.state.batch_gradient
    w_input_norm = hasproperty(params, :w_input) ? parameter_norm(params.w_input) : 0.0
    grad_w_input_norm = hasproperty(grad, :w_input) ? parameter_norm(grad.w_input) : 0.0
    effective_w_decay = effective_weight_decay(manager.config, parameter_norm(params.w), FT(manager.config.adaptive_w_norm))
    effective_w_input_decay = hasproperty(params, :w_input) ?
        effective_weight_decay(manager.config, parameter_norm(params.w_input), FT(manager.config.adaptive_w_input_norm)) :
        FT(manager.config.weight_decay)
    return (;
        w_norm = parameter_norm(params.w),
        b_norm = parameter_norm(params.b),
        w_input_norm,
        grad_w_norm = parameter_norm(grad.w),
        grad_b_norm = parameter_norm(grad.b),
        grad_w_input_norm,
        effective_w_decay,
        effective_w_input_decay,
        symmetry_error = sparse_parameter_symmetry_error(params.w, II.adj(manager.state.source_graph), manager.state.reverse_sparse_ptrs),
        grad_symmetry_error = sparse_parameter_symmetry_error(grad.w, II.adj(manager.state.source_graph), manager.state.reverse_sparse_ptrs),
    )
end

"""Return last-minibatch gradient quality acceptance and disagreement metrics."""
function training_quality(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    total = manager.state.gradient_quality_total[]
    accepted = manager.state.gradient_quality_accepted[]
    rejected = manager.state.gradient_quality_rejected[]
    return (;
        gradient_quality_total = total,
        gradient_quality_accepted = accepted,
        gradient_quality_rejected = rejected,
        gradient_quality_reject_fraction = total == 0 ? 0.0 : rejected / total,
        gradient_quality_avg_cosine = total == 0 ? NaN : manager.state.gradient_quality_sum_cosine[] / total,
        gradient_quality_avg_angle_degrees = total == 0 ? NaN : manager.state.gradient_quality_sum_angle_degrees[] / total,
    )
end

"""Normalize raw MNIST images to `[0, 1]` input-field intensities."""
function normalize_images(images::A, ::Type{T}) where {A,T<:AbstractFloat}
    x = T.(images)
    maximum(x) > one(T) && (x ./= T(255))
    return reshape(x, :, size(images, ndims(images)))
end

"""Build four-replica output targets with `-1` off spins and `+1` on spins."""
function repeated_targets(labels::L, replicas::I, ::Type{T}) where {L,I<:Integer,T<:AbstractFloat}
    y = fill(-one(T), NCLASSES * Int(replicas), length(labels))
    @inbounds for (col, label) in enumerate(labels)
        first_idx = Int(label) * Int(replicas) + 1
        y[first_idx:(first_idx + Int(replicas) - 1), col] .= one(T)
    end
    return y
end

"""Load a balanced MNIST split with the same number of examples per class."""
function balanced_mnist(split::Symbol, per_class::I, config::C) where {I<:Integer,C<:InputFieldMNISTConfig}
    dataset = split === :train ? MLDatasets.MNIST(split = :train) :
        split === :test ? MLDatasets.MNIST(split = :test) :
        throw(ArgumentError("split must be :train or :test"))
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end

    keep = Int[]
    rng = Random.MersenneTwister(hash((config.seed, split, Int(per_class))))
    for digit in 1:NCLASSES
        length(buckets[digit]) >= Int(per_class) ||
            throw(ArgumentError("split $(split) has only $(length(buckets[digit])) samples for digit $(digit - 1)"))

        # Shuffle class buckets once so reduced runs are still representative.
        digit_indices = copy(buckets[digit])
        Random.shuffle!(rng, digit_indices)
        append!(keep, @view digit_indices[1:Int(per_class)])
    end

    xall = normalize_images(images, FT)
    yall = repeated_targets(labels, config.output_replicas, FT)
    return Matrix{FT}(xall[:, keep]), Matrix{FT}(yall[:, keep])
end

"""Average output replicas into one score per digit."""
function class_scores(output::V, replicas::I) where {V<:AbstractVector,I<:Integer}
    scores = zeros(eltype(output), NCLASSES)
    @inbounds for digit in 1:NCLASSES
        first_idx = (digit - 1) * Int(replicas) + 1
        scores[digit] = mean(@view output[first_idx:(first_idx + Int(replicas) - 1)])
    end
    return scores
end

"""Create a reusable buffer of concrete manager jobs for one fixed capacity."""
function InputFieldMNISTJobBuffer(capacity::I, input_dim::J, output_dim::K) where {I<:Integer,J<:Integer,K<:Integer}
    jobs = InputFieldMNISTJob{Vector{FT},Vector{FT}}[
        InputFieldMNISTJob(zeros(FT, Int(input_dim)), zeros(FT, Int(output_dim))) for _ in 1:Int(capacity)
    ]
    return InputFieldMNISTJobBuffer(jobs)
end

"""Fill preallocated manager jobs from selected sample columns and return the active view."""
function fill_jobs!(buffer::B, x::X, y::Y, indices::V) where {B<:InputFieldMNISTJobBuffer,X<:AbstractMatrix,Y<:AbstractMatrix,V<:AbstractVector{Int}}
    n = length(indices)
    n <= length(buffer.jobs) || throw(ArgumentError("job buffer capacity $(length(buffer.jobs)) is smaller than requested jobs $(n)"))
    @inbounds for slot_idx in 1:n
        sample_idx = indices[slot_idx]
        job = buffer.jobs[slot_idx]
        job.x .= view(x, :, sample_idx)
        job.y .= view(y, :, sample_idx)
    end
    return @view buffer.jobs[1:n]
end

"""Create one gradient buffer matching the reduced field-input baseline parameters."""
function input_field_gradient_buffer(graph::G, input_hidden_w::W) where {G,W<:AbstractMatrix}
    return (;
        w = zeros(eltype(graph), length(SparseArrays.getnzval(II.adj(graph)))),
        b = zeros(eltype(graph), II.nstates(graph)),
        w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
    )
end

"""Return the optimizer-facing parameter arrays for the reduced field-input baseline."""
function input_field_params(graph::G, input_hidden_w::W) where {G,W<:AbstractMatrix}
    return (;
        w = copy(SparseArrays.getnzval(II.adj(graph))),
        b = copy(IsingLearning._mnist_base_magfield(graph).b),
        w_input = copy(input_hidden_w),
    )
end

"""Build a CSC nonzero-pointer map from each directed edge to its reverse edge."""
function sparse_reverse_ptrs(adjacency::A) where {A<:SparseArrays.AbstractSparseMatrix}
    rowvals = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    reverse_ptrs = Vector{Int}(undef, length(SparseArrays.nonzeros(adjacency)))
    ptrs_by_edge = Dict{Tuple{Int,Int},Int}()
    sizehint!(ptrs_by_edge, length(reverse_ptrs))

    # Register every stored directed edge once in CSC pointer order.
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            ptrs_by_edge[(Int(rowvals[ptr]), Int(col))] = Int(ptr)
        end
    end

    # Resolve reverse pointers once so later symmetry passes only touch vectors.
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = Int(rowvals[ptr])
            reverse_ptrs[ptr] = get(
                () -> throw(ArgumentError("missing reverse sparse edge for ($(row), $(col))")),
                ptrs_by_edge,
                (Int(col), row),
            )
        end
    end
    return reverse_ptrs
end

"""Symmetrize paired directed sparse entries in-place for an undirected Ising coupling."""
function symmetrize_sparse_parameter_values!(
    values::V,
    adjacency::A,
    reverse_ptrs::R,
) where {V<:AbstractVector,A<:SparseArrays.AbstractSparseMatrix,R<:AbstractVector{<:Integer}}
    rowvals = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            row <= col && continue
            rev_ptr = reverse_ptrs[ptr]
            avg = (values[ptr] + values[rev_ptr]) / eltype(values)(2)
            values[ptr] = avg
            values[rev_ptr] = avg
        end
    end
    return values
end

"""Return the maximum absolute mismatch between paired symmetric sparse entries."""
function sparse_parameter_symmetry_error(
    values::V,
    adjacency::A,
    reverse_ptrs::R,
) where {V<:AbstractVector,A<:SparseArrays.AbstractSparseMatrix,R<:AbstractVector{<:Integer}}
    rowvals = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    err = zero(eltype(values))
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            row <= col && continue
            rev_ptr = reverse_ptrs[ptr]
            err = max(err, abs(values[ptr] - values[rev_ptr]))
        end
    end
    return err
end

"""Constrain the recurrent sparse Ising couplings to a selected global norm."""
function normalize_sparse_couplings!(values::V, config::C) where {V<:AbstractVector,C<:InputFieldMNISTConfig}
    mode = String(config.w_normalization)
    mode == "none" && return values
    target_norm = eltype(values)(config.w_norm)
    target_norm > zero(target_norm) || throw(ArgumentError("ISING_MNIST_IF_W_NORM must be positive"))
    if mode == "global"
        norm2 = sum(abs2, values)
        norm = sqrt(norm2)
        norm == zero(norm) && return values
        values .*= target_norm / norm
    else
        throw(ArgumentError("unknown recurrent w normalization mode: $(mode)"))
    end
    return values
end

"""Return a near-square display shape for one flattened MNIST hidden/output layer."""
function mnist_layer_shape(units::I) where {I<:Integer}
    side = floor(Int, sqrt(Int(units)))
    while side > 1 && mod(Int(units), side) != 0
        side -= 1
    end
    return side, Int(units) ÷ side
end

"""Create the reduced hidden/output graph and external input projection for the baseline."""
function build_layer(config::C) where {C<:InputFieldMNISTConfig}
    hidden_units = Int(config.hidden)
    output_units = NCLASSES * Int(config.output_replicas)
    hidden_rows, hidden_cols = mnist_layer_shape(hidden_units)
    output_rows, output_cols = mnist_layer_shape(output_units)
    side = IsingLearning.D_MNIST
    rng = Random.MersenneTwister(config.seed)
    scale = FT(config.weight_scale)

    hidden_layer = II.Layer(
        hidden_rows,
        hidden_cols,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        II.Coords(0, side + 2, 0);
        periodic = false,
    )
    output_layer = II.Layer(
        output_rows,
        output_cols,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        II.Coords(0, side + hidden_cols + 4, 0);
        periodic = false,
    )

    # Keep the dense image-to-hidden coupling outside the sampled graph.
    input_hidden_w = scale .* randn(rng, FT, hidden_units, INPUT_DIM)

    # The sampled graph contains only hidden/output couplings.
    nedges = 2 * hidden_units * output_units
    rows = Vector{Int}(undef, nedges)
    cols = Vector{Int}(undef, nedges)
    vals = Vector{FT}(undef, nedges)
    ptr = 1
    @inbounds for output_pos in 1:output_units
        graph_output_idx = hidden_units + output_pos
        for hidden_idx in 1:hidden_units
            weight = scale * randn(rng, FT)
            rows[ptr] = hidden_idx
            cols[ptr] = graph_output_idx
            vals[ptr] = weight
            ptr += 1
            rows[ptr] = graph_output_idx
            cols[ptr] = hidden_idx
            vals[ptr] = weight
            ptr += 1
        end
    end

    base_bias = zeros(FT, hidden_units + output_units)
    image_field = zeros(FT, hidden_units + output_units)
    hamiltonian = II.Bilinear() +
        II.MagField(b = II.Force(base_bias)) +
        II.MagField(b = II.Force(image_field)) +
        II.SoftplusMarginNudging(
            β = II.UniformArray(zero(FT)),
            y = g -> II.filltype(Vector, zero(FT), II.statelen(g)),
            τ = II.UniformArray(FT(config.nudge_tau)),
        )
    graph = II.IsingGraph(
        hidden_layer,
        output_layer,
        hamiltonian;
        precision = FT,
        adj = II.UndirectedAdjacency(sparse(rows, cols, vals, hidden_units + output_units, hidden_units + output_units)),
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)

    relaxation_steps = max(1, round(Int, config.sweeps * active_units(graph)))
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = FT(0.15),
        adjusted = false,
        order = :cyclic,
    )
    layer = IsingLearning.LayeredIsingGraphLayer(
        graph;
        input_idxs = Base.OneTo(INPUT_DIM),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        free_relaxation_steps = relaxation_steps,
        nudged_relaxation_steps = relaxation_steps,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return (; graph, layer, relaxation_steps, input_hidden_w)
end

"""Write one projected MNIST image field into the hidden slice of a worker-local buffer."""
function project_input_field_pattern!(pattern::P, input_hidden_w::W, x::X) where {
    P<:AbstractVector,
    W<:AbstractMatrix,
    X<:AbstractVector,
}
    fill!(pattern, zero(eltype(pattern)))
    hidden_view = @view pattern[1:size(input_hidden_w, 1)]
    mul!(hidden_view, input_hidden_w, x)
    return pattern
end

"""Install one precomputed worker-local input field into the reduced baseline graph."""
function install_input_field_pattern!(isinggraph::G, pattern::P) where {G,P<:AbstractVector}
    input_field = IsingLearning._mnist_input_magfield(isinggraph)
    isnothing(input_field) && error("reduced baseline graph is missing its worker-local input field")
    copyto!(input_field.b, pattern)
    return isinggraph
end

"""Return the nudged-phase temperature algorithm selected by the experiment config."""
function nudged_temperature_algorithm(config::C, n_steps::I) where {C<:InputFieldMNISTConfig,I<:Integer}
    mode = String(config.nudge_temp_schedule)
    if mode == "fixed"
        return IsingLearning.GeometricDynamicsTemperatureSchedule(;
            start_T = FT(config.temp),
            stop_T = FT(config.temp),
            n_steps = Int(n_steps),
        )
    elseif mode == "reverse_anneal"
        return IsingLearning.ReverseAnnealDynamicsTemperatureSchedule(;
            cold_T = FT(config.temp),
            peak_T = FT(config.nudge_temp_peak),
            n_steps = Int(n_steps),
        )
    end
    throw(ArgumentError("ISING_MNIST_IF_NUDGE_TEMP_SCHEDULE must be fixed or reverse_anneal; got $(mode)"))
end

"""Accumulate one MNIST field-input contrastive gradient into one worker buffer."""
function accumulate_input_field_gradient!(
    isinggraph::G,
    nudged_state,
    equilibrium_state,
    x,
    buffers,
    β::Real;
    direct_contrast::Bool = false,
) where G
    if direct_contrast
        IsingLearning.contrastive_gradient_new(isinggraph, nudged_state, equilibrium_state, β; buffers = buffers)
    else
        IsingLearning.contrastive_gradient(isinggraph, nudged_state, equilibrium_state, β; buffers = buffers)
    end
    hidden_count = size(buffers.w_input, 1)
    @inbounds for input_idx in eachindex(x)
        xval = x[input_idx]
        for hidden_idx in 1:hidden_count
            buffers.w_input[hidden_idx, input_idx] += -xval * (nudged_state[hidden_idx] - equilibrium_state[hidden_idx])
        end
    end
    return
end

"""Return the object inside a `Ref`, or the object itself for direct values."""
@inline ref_value(x::Base.RefValue) = x[]
@inline ref_value(x) = x

"""Balance softplus-margin target masks so sparse positive replicas are not drowned out."""
function reweight_softplus_target_mask!(
    isinggraph::G,
    y::Y,
    positive_target_weight::P,
    negative_target_weight::N,
) where {G,Y<:AbstractVector,P<:Real,N<:Real}
    nudge = IsingLearning.hamiltonian_or_nothing(isinggraph.hamiltonian, II.SoftplusMarginNudging)
    isnothing(nudge) && return isinggraph

    output_idxs = II.layerrange(isinggraph[end])
    length(y) == length(output_idxs) ||
        throw(DimensionMismatch("target length $(length(y)) does not match output layer length $(length(output_idxs))"))
    @inbounds for local_idx in eachindex(y)
        graph_idx = output_idxs[local_idx]
        nudge.mask[graph_idx] = y[local_idx] > zero(eltype(y)) ?
            eltype(nudge.mask)(positive_target_weight) :
            eltype(nudge.mask)(negative_target_weight)
    end
    return isinggraph
end

# Apply a possibly reference-backed MNIST output target to one worker graph.
StatefulAlgorithms.@ProcessAlgorithm function ApplyTargetsRef!(
    isinggraph::G,
    y,
    positive_target_weight::Float32,
    negative_target_weight::Float32,
) where G
    target = ref_value(y)
    IsingLearning.apply_targets(isinggraph, target)
    reweight_softplus_target_mask!(isinggraph, target, positive_target_weight, negative_target_weight)
    return nothing
end

# Project one reference-backed MNIST sample into the worker-local field buffer and install it.
StatefulAlgorithms.@ProcessAlgorithm function ApplyProjectedInputFieldRef!(
    isinggraph::G,
    input_hidden_w,
    x,
    input_pattern::AbstractVector,
) where G
    project_input_field_pattern!(input_pattern, ref_value(input_hidden_w), ref_value(x))
    install_input_field_pattern!(isinggraph, input_pattern)
    return nothing
end

# Accumulate the input-field contrastive gradient from a reference-backed input.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateInputFieldGradientRef!(
    isinggraph::G,
    nudged_state,
    equilibrium_state,
    x,
    buffers,
    β::Real,
) where G
    accumulate_input_field_gradient!(
        isinggraph,
        nudged_state,
        equilibrium_state,
        ref_value(x),
        buffers,
        β,
    )
    return nothing
end

# Accept one symmetric sample gradient only when it agrees with the forward estimate.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateInputFieldGradientQualityRef!(
    isinggraph::G,
    plus_state,
    minus_state,
    equilibrium_state,
    x,
    buffers,
    forward_buffers,
    symmetric_buffers,
    quality_stats,
    β::Real,
    min_cosine::Real,
) where G
    clear_buffer!(forward_buffers)
    clear_buffer!(symmetric_buffers)

    sample_x = ref_value(x)
    accumulate_input_field_gradient!(
        isinggraph,
        plus_state,
        equilibrium_state,
        sample_x,
        forward_buffers,
        β,
    )
    accumulate_input_field_gradient!(
        isinggraph,
        plus_state,
        minus_state,
        sample_x,
        symmetric_buffers,
        β;
        direct_contrast = true,
    )

    agreement = gradient_buffer_cosine_angle(forward_buffers, symmetric_buffers)
    accepted = isfinite(agreement.cosine) && agreement.cosine >= Float64(min_cosine)
    record_gradient_quality!(quality_stats, agreement.cosine, agreement.angle_degrees, accepted)
    accepted && add_buffer!(buffers, symmetric_buffers)
    return nothing
end

"""Build the free-phase input-field routine for one MNIST sample."""
function input_field_free_phase_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    free_steps = layer.free_relaxation_steps
    n_units = layer.nunits

    return StatefulAlgorithms.@Routine begin
        @state x
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state equilibrium_state = zeros(FT, n_units)
        @alias dynamics = dynamics_algorithm

        # Fold the image into the worker-local input field, then relax once.
        IsingLearning.initstate!(dynamics.model)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        model = @repeat free_steps dynamics()
        IsingLearning.CopyGraphState!(equilibrium_state, model)
    end
end

"""Build one nudged dynamics step that refreshes temperature before stepping."""
function input_field_nudged_phase_step_algorithm(dynamics_algorithm::D, temperature_algorithm::T) where {D,T}
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        @alias temperature = temperature_algorithm

        temperature(dynamics.T)
        dynamics()
    end
end

"""Build the positive-nudged input-field routine for one MNIST sample."""
function input_field_plus_nudged_phase_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    nudged_steps = layer.nudged_relaxation_steps
    temperature_algorithm = nudged_temperature_algorithm(config, nudged_steps)
    phase_step = input_field_nudged_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    n_units = layer.nunits
    default_β = layer.β
    default_positive_target_weight = FT(config.positive_target_weight)
    default_negative_target_weight = FT(config.negative_target_weight)

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern
        @state phase_beta = default_β
        @state positive_target_weight = default_positive_target_weight
        @state negative_target_weight = default_negative_target_weight
        @state equilibrium_state
        @state plus_state = zeros(FT, n_units)
        @alias dynamics = dynamics_algorithm
        @alias step = phase_step

        # Restart from the free equilibrium, add the target nudge, then relax.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        ApplyTargetsRef!(dynamics.model, y, positive_target_weight, negative_target_weight)
        SetInputFieldClampingBeta!(isinggraph = dynamics.model, phase_beta = phase_beta)
        @repeat nudged_steps step()
        IsingLearning.CopyGraphState!(plus_state, dynamics.model)

        # Reset clamping so the worker can be rerun without rebuilding state.
        SetInputFieldClampingBeta!(isinggraph = dynamics.model, phase_beta = @transform(_ -> 0f0, phase_beta))
    end
end

"""Build the negative-nudged input-field routine for one MNIST sample."""
function input_field_minus_nudged_phase_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    nudged_steps = layer.nudged_relaxation_steps
    temperature_algorithm = nudged_temperature_algorithm(config, nudged_steps)
    phase_step = input_field_nudged_phase_step_algorithm(dynamics_algorithm, temperature_algorithm)
    n_units = layer.nunits

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern
        @state phase_beta
        @state positive_target_weight
        @state negative_target_weight
        @state equilibrium_state
        @state minus_state = zeros(FT, n_units)
        @alias dynamics = dynamics_algorithm
        @alias step = phase_step

        # Restart from the free equilibrium, add the negative target nudge, then relax.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        ApplyTargetsRef!(dynamics.model, y, positive_target_weight, negative_target_weight)
        SetInputFieldNegativeClampingBeta!(isinggraph = dynamics.model, phase_beta = phase_beta)
        @repeat nudged_steps step()
        IsingLearning.CopyGraphState!(minus_state, dynamics.model)

        # Reset clamping so the worker can be rerun without rebuilding state.
        SetInputFieldClampingBeta!(isinggraph = dynamics.model, phase_beta = @transform(_ -> 0f0, phase_beta))
    end
end

"""Build the one-sided single-sample input-field equilibrium-propagation step."""
function input_field_one_sided_contrastive_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    default_β = layer.β
    free_phase = input_field_free_phase_algorithm(layer)
    nudged_phase = input_field_plus_nudged_phase_algorithm(layer, config)

    return StatefulAlgorithms.@CompositeAlgorithm begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @input phase_beta::FT = default_β

        @context free_context = free_phase()
        @context nudged_context = nudged_phase()
        @bind x => free_context.x
        @bind x => nudged_context.x
        @bind y => nudged_context.y
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => nudged_context.input_hidden_w
        @merge free_context.input_pattern, nudged_context.input_pattern
        @merge free_context.equilibrium_state, nudged_context.equilibrium_state
        @bind phase_beta => nudged_context.phase_beta

        AccumulateInputFieldGradientRef!(
            nudged_context.dynamics.model,
            nudged_context.plus_state,
            free_context.equilibrium_state,
            x,
            buffers,
            phase_beta,
        )
    end
end

"""Build the symmetric `+β/-β` single-sample input-field contrastive step."""
function input_field_symmetric_contrastive_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    default_β = layer.β
    free_phase = input_field_free_phase_algorithm(layer)
    plus_phase = input_field_plus_nudged_phase_algorithm(layer, config)
    minus_phase = input_field_minus_nudged_phase_algorithm(layer, config)

    return StatefulAlgorithms.@CompositeAlgorithm begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @input phase_beta::FT = default_β

        @context free_context = free_phase()
        @context plus_context = plus_phase()
        @context minus_context = minus_phase()
        @bind x => free_context.x
        @bind x => plus_context.x
        @bind x => minus_context.x
        @bind y => plus_context.y
        @bind y => minus_context.y
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => plus_context.input_hidden_w
        @bind input_hidden_w => minus_context.input_hidden_w
        @merge free_context.input_pattern, plus_context.input_pattern
        @merge free_context.input_pattern, minus_context.input_pattern
        @merge free_context.equilibrium_state, plus_context.equilibrium_state
        @merge free_context.equilibrium_state, minus_context.equilibrium_state
        @bind phase_beta => plus_context.phase_beta
        @merge plus_context.x, minus_context.x
        @merge plus_context.y, minus_context.y
        @merge plus_context.input_hidden_w, minus_context.input_hidden_w
        @merge plus_context.input_pattern, minus_context.input_pattern
        @merge plus_context.equilibrium_state, minus_context.equilibrium_state
        @merge plus_context.phase_beta, minus_context.phase_beta
        @merge plus_context.positive_target_weight, minus_context.positive_target_weight
        @merge plus_context.negative_target_weight, minus_context.negative_target_weight

        AccumulateInputFieldGradientRef!(
            plus_context.dynamics.model,
            plus_context.plus_state,
            minus_context.minus_state,
            x,
            buffers,
            phase_beta,
        )
    end
end

"""Build the symmetric step with a forward/symmetric cosine quality gate."""
function input_field_gradient_quality_contrastive_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    default_β = layer.β
    default_min_cosine = FT(config.gradient_quality_min_cosine)
    free_phase = input_field_free_phase_algorithm(layer)
    plus_phase = input_field_plus_nudged_phase_algorithm(layer, config)
    minus_phase = input_field_minus_nudged_phase_algorithm(layer, config)

    return StatefulAlgorithms.@CompositeAlgorithm begin
        @state x
        @state y
        @state buffers
        @state forward_buffers
        @state symmetric_buffers
        @state quality_stats
        @state quality_min_cosine = default_min_cosine
        @state input_hidden_w
        @input phase_beta::FT = default_β

        @context free_context = free_phase()
        @context plus_context = plus_phase()
        @context minus_context = minus_phase()
        @bind x => free_context.x
        @bind x => plus_context.x
        @bind x => minus_context.x
        @bind y => plus_context.y
        @bind y => minus_context.y
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => plus_context.input_hidden_w
        @bind input_hidden_w => minus_context.input_hidden_w
        @merge free_context.input_pattern, plus_context.input_pattern
        @merge free_context.input_pattern, minus_context.input_pattern
        @merge free_context.equilibrium_state, plus_context.equilibrium_state
        @merge free_context.equilibrium_state, minus_context.equilibrium_state
        @bind phase_beta => plus_context.phase_beta
        @merge plus_context.x, minus_context.x
        @merge plus_context.y, minus_context.y
        @merge plus_context.input_hidden_w, minus_context.input_hidden_w
        @merge plus_context.input_pattern, minus_context.input_pattern
        @merge plus_context.equilibrium_state, minus_context.equilibrium_state
        @merge plus_context.phase_beta, minus_context.phase_beta
        @merge plus_context.positive_target_weight, minus_context.positive_target_weight
        @merge plus_context.negative_target_weight, minus_context.negative_target_weight

        AccumulateInputFieldGradientQualityRef!(
            plus_context.dynamics.model,
            plus_context.plus_state,
            minus_context.minus_state,
            free_context.equilibrium_state,
            x,
            buffers,
            forward_buffers,
            symmetric_buffers,
            quality_stats,
            phase_beta,
            quality_min_cosine,
        )
    end
end

"""Build the selected single-sample input-field contrastive step."""
function input_field_contrastive_algorithm(layer::L, config::C) where {
    L<:IsingLearning.LayeredIsingGraphLayer,
    C<:InputFieldMNISTConfig,
}
    if uses_gradient_quality_nudging(config)
        return input_field_gradient_quality_contrastive_algorithm(layer, config)
    end
    if uses_symmetric_nudging(config)
        return input_field_symmetric_contrastive_algorithm(layer, config)
    end
    return input_field_one_sided_contrastive_algorithm(layer, config)
end

"""Return the mutable sample/buffer context stored in one worker."""
function worker_context(worker::W) where {W}
    return StatefulAlgorithms.context(worker)._state
end

"""Return the graph owned by a worker dynamics context."""
function worker_graph(worker::W) where {W}
    return StatefulAlgorithms.context(worker).dynamics.model
end

"""Build a worker graph whose adjacency is the source graph adjacency object."""
function shared_worker_graph(source::G) where {G}
    graph = IsingLearning._shared_mnist_worker_graph(source; input_mode = :field)
    II.adj(graph) === II.adj(source) || error("worker graph does not share source adjacency")
    SparseArrays.nonzeros(II.adj(graph)) === SparseArrays.nonzeros(II.adj(source)) ||
        error("worker graph J storage is not pointer-shared with source graph")
    IsingLearning._mnist_base_magfield(graph).b === IsingLearning._mnist_base_magfield(source).b ||
        error("worker graph base bias storage is not pointer-shared with source graph")
    return graph
end

# Set the graph clamping beta from process state without capturing an external Ref.
StatefulAlgorithms.@ProcessAlgorithm function SetInputFieldClampingBeta!(isinggraph::G, phase_beta::Float32) where G
    IsingLearning.set_clamping_beta!(isinggraph, phase_beta)
    return nothing
end

# Set the negative graph clamping beta for the symmetric `+β/-β` contrastive phase.
StatefulAlgorithms.@ProcessAlgorithm function SetInputFieldNegativeClampingBeta!(isinggraph::G, phase_beta::Float32) where G
    IsingLearning.set_clamping_beta!(isinggraph, -phase_beta)
    return nothing
end

"""Accumulate one validation sample into worker-local counters without score allocations."""
function accumulate_input_field_validation_stats!(
    y::AbstractVector,
    equilibrium_state::AbstractVector,
    output_idxs::AbstractVector{Int},
    pred_counts::AbstractVector{Int},
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    total_loss::Base.RefValue,
    output_replicas::Integer,
)
    best_pred = 1
    best_truth = 1
    best_pred_score = typemin(Float32)
    best_truth_score = typemin(Float32)
    loss = 0f0
    replicas = Int(output_replicas)

    @inbounds for digit in 1:NCLASSES
        pred_score = 0f0
        truth_score = 0f0
        first_idx = (digit - 1) * replicas + 1
        for replica_idx in first_idx:(first_idx + replicas - 1)
            out = Float32(equilibrium_state[output_idxs[replica_idx]])
            target = Float32(y[replica_idx])
            pred_score += out
            truth_score += target
            loss += (out - target)^2
        end
        if pred_score > best_pred_score
            best_pred_score = pred_score
            best_pred = digit
        end
        if truth_score > best_truth_score
            best_truth_score = truth_score
            best_truth = digit
        end
    end

    correct = best_pred == best_truth
    pred_counts[best_pred] += 1
    nsamples[] += 1
    ncorrect[] += correct ? 1 : 0
    total_loss[] += loss
    return (; loss, correct)
end

# Accumulate one validation sample into worker-local counters without score allocations.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateInputFieldValidationStats!(
    y::AbstractVector,
    equilibrium_state::AbstractVector,
    output_idxs::AbstractVector{Int},
    pred_counts::AbstractVector{Int},
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    total_loss::Base.RefValue,
    output_replicas::Integer,
)
    return accumulate_input_field_validation_stats!(
        y,
        equilibrium_state,
        output_idxs,
        pred_counts,
        nsamples,
        ncorrect,
        total_loss,
        output_replicas,
    )
end

# Accumulate validation statistics from a reference-backed output target.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateInputFieldValidationStatsRef!(
    y,
    equilibrium_state::AbstractVector,
    output_idxs::AbstractVector{Int},
    pred_counts::AbstractVector{Int},
    nsamples::Base.RefValue,
    ncorrect::Base.RefValue,
    total_loss::Base.RefValue,
    output_replicas::Integer,
)
    accumulate_input_field_validation_stats!(
        ref_value(y),
        equilibrium_state,
        output_idxs,
        pred_counts,
        nsamples,
        ncorrect,
        total_loss,
        output_replicas,
    )
    return nothing
end

"""Build one ProcessManager worker from the already resolved baseline LoopAlgorithm."""
function input_field_worker(
    algorithm::A,
    layer::L,
    graph::G,
    input_hidden_w::R,
) where {A,L<:IsingLearning.LayeredIsingGraphLayer,G,R<:Base.RefValue}
    state = II.state(graph)
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            x = Ref(zeros(eltype(graph), INPUT_DIM)),
            y = Ref(zeros(eltype(graph), length(layer.output_layer))),
            input_hidden_w = input_hidden_w,
            buffers = input_field_gradient_buffer(graph, input_hidden_w[]),
            forward_buffers = input_field_gradient_buffer(graph, input_hidden_w[]),
            symmetric_buffers = input_field_gradient_buffer(graph, input_hidden_w[]),
            quality_stats = gradient_quality_stats(),
            equilibrium_state = copy(state),
            nudged_state = similar(state),
        ),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Build the reusable validation algorithm used by manager-owned validation workers."""
function input_field_validation_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = deepcopy(layer.validation_algorithm)
    relaxation_steps = layer.free_relaxation_steps
    n_units = layer.nunits
    replica_count = length(layer.output_layer) ÷ NCLASSES

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state equilibrium_state = zeros(FT, n_units)
        @state output_idxs
        @state output_replicas = replica_count
        @state nsamples
        @state ncorrect
        @state total_loss
        @state pred_counts
        @alias dynamics = dynamics_algorithm

        # Validation is a free phase plus worker-local statistics; no source graph state is mutated.
        IsingLearning.initstate!(dynamics.model)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        model = @repeat relaxation_steps dynamics()
        IsingLearning.CopyGraphState!(equilibrium_state, model)
        AccumulateInputFieldValidationStatsRef!(
            y,
            equilibrium_state,
            output_idxs,
            pred_counts,
            nsamples,
            ncorrect,
            total_loss,
            output_replicas,
        )
    end
end

"""Build one manager-owned validation worker with all counters preallocated."""
function input_field_validation_worker(
    algorithm::A,
    layer::L,
    graph::G,
    input_hidden_w::R,
) where {A,L<:IsingLearning.LayeredIsingGraphLayer,G,R<:Base.RefValue}
    state = II.state(graph)
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            x = Ref(zeros(eltype(graph), INPUT_DIM)),
            y = Ref(zeros(eltype(graph), length(layer.output_layer))),
            input_hidden_w = input_hidden_w,
            equilibrium_state = copy(state),
            output_idxs = collect(Int, II.layerrange(graph[end])),
            nsamples = Ref(0),
            ncorrect = Ref(0),
            total_loss = Ref(0f0),
            pred_counts = zeros(Int, NCLASSES),
        ),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Load one prebuilt training job into a persistent field-input worker."""
function load_input_field_training_job!(slot::S, job::J, manager::M) where {
    S,
    J<:InputFieldMNISTJob,
    M<:StatefulAlgorithms.ProcessManager,
}
    ctx = worker_context(slot.worker)
    copyto!(ctx.x[], job.x)
    copyto!(ctx.y[], job.y)
    StatefulAlgorithms.resetworker!(slot)
    return nothing
end

"""Load one prebuilt validation job into a persistent field-input worker."""
function load_input_field_validation_job!(slot::S, job::J, manager::M) where {
    S,
    J<:InputFieldMNISTJob,
    M<:StatefulAlgorithms.ProcessManager,
}
    ctx = worker_context(slot.worker)
    copyto!(ctx.x[], job.x)
    copyto!(ctx.y[], job.y)
    StatefulAlgorithms.resetworker!(slot)
    return nothing
end

"""Close one persistent field-input Process worker."""
function close_process_worker!(slot::S, manager::M) where {S,M<:StatefulAlgorithms.ProcessManager}
    close(slot.worker)
    return slot.worker
end

"""Clear the manager batch buffer and all worker-local gradient buffers."""
function clear_manager_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    manager.state.nsamples[] = 0
    manager.state.gradient_quality_total[] = 0
    manager.state.gradient_quality_accepted[] = 0
    manager.state.gradient_quality_rejected[] = 0
    manager.state.gradient_quality_sum_cosine[] = 0.0
    manager.state.gradient_quality_sum_angle_degrees[] = 0.0
    for worker in StatefulAlgorithms.workers(manager)
        ctx = worker_context(worker)
        clear_buffer!(ctx.buffers)
        if uses_gradient_quality_nudging(manager.config)
            clear_buffer!(ctx.forward_buffers)
            clear_buffer!(ctx.symmetric_buffers)
            reset_gradient_quality_stats!(ctx.quality_stats)
        end
    end
    return manager
end

"""Reset one validation worker's accounting fields."""
function reset_validation_worker_stats!(ctx::C) where {C}
    ctx.nsamples[] = 0
    ctx.ncorrect[] = 0
    ctx.total_loss[] = 0f0
    fill!(ctx.pred_counts, 0)
    return ctx
end

"""Clear validation manager and worker-local statistic buffers."""
function clear_validation_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    manager.state.nsamples[] = 0
    manager.state.ncorrect[] = 0
    manager.state.total_loss[] = 0f0
    fill!(manager.state.pred_counts, 0)
    for worker in StatefulAlgorithms.workers(manager)
        reset_validation_worker_stats!(worker_context(worker))
    end
    return manager
end

"""Remove output-bias common class-prior drift from one gradient buffer."""
function project_output_bias_prior!(gradient::G, layer::L, output_replicas::I) where {G,L,I<:Integer}
    replicas = Int(output_replicas)
    output_idxs = collect(Int, layer.output_layer)
    length(output_idxs) == NCLASSES * replicas ||
        throw(ArgumentError("output replica count does not match layer output range"))
    @inbounds for digit in 1:NCLASSES
        first_idx = (digit - 1) * replicas + 1
        class_mean = zero(eltype(gradient.b))
        for replica_idx in first_idx:(first_idx + replicas - 1)
            class_mean += gradient.b[output_idxs[replica_idx]]
        end
        class_mean /= eltype(gradient.b)(replicas)
        for replica_idx in first_idx:(first_idx + replicas - 1)
            gradient.b[output_idxs[replica_idx]] -= class_mean
        end
    end
    return gradient
end

"""Constrain the external pixel-to-hidden projection after optimizer updates."""
function normalize_input_projection!(w_input::W, config::C) where {W<:AbstractMatrix,C<:InputFieldMNISTConfig}
    mode = String(config.w_input_normalization)
    mode == "none" && return w_input
    target_norm = eltype(w_input)(config.w_input_row_norm)
    target_norm > zero(target_norm) || throw(ArgumentError("ISING_MNIST_IF_W_INPUT_ROW_NORM must be positive"))
    if mode == "row"
        @inbounds for hidden_idx in axes(w_input, 1)
            norm2 = zero(eltype(w_input))
            for input_idx in axes(w_input, 2)
                val = w_input[hidden_idx, input_idx]
                norm2 += val * val
            end
            norm = sqrt(norm2)
            norm == zero(norm) && continue
            scale = target_norm / norm
            for input_idx in axes(w_input, 2)
                w_input[hidden_idx, input_idx] *= scale
            end
        end
    elseif mode == "global"
        norm2 = sum(abs2, w_input)
        norm = sqrt(norm2)
        norm == zero(norm) && return w_input
        w_input .*= target_norm / norm
    else
        throw(ArgumentError("unknown w_input normalization mode: $(mode)"))
    end
    return w_input
end

"""Return the decay coefficient for one parameter group under optional norm control."""
function effective_weight_decay(config::C, norm::T, target_norm::T) where {C<:InputFieldMNISTConfig,T<:Real}
    base_decay = T(config.weight_decay)
    config.adaptive_weight_decay || return base_decay
    target_norm > zero(T) || return base_decay

    overshoot = max(zero(T), norm / target_norm - one(T))
    adaptive_decay = base_decay + T(config.adaptive_decay_gain) * overshoot
    return min(T(config.adaptive_max_decay), adaptive_decay)
end

"""Add recurrent and input-projection weight decay to one minibatch gradient."""
function apply_weight_decay!(gradient::G, params::P, config::C) where {G,P,C<:InputFieldMNISTConfig}
    w_decay = effective_weight_decay(config, parameter_norm(params.w), FT(config.adaptive_w_norm))
    input_decay = effective_weight_decay(config, parameter_norm(params.w_input), FT(config.adaptive_w_input_norm))
    if w_decay > zero(FT)
        gradient.w .+= w_decay .* params.w
    end
    if input_decay > zero(FT)
        gradient.w_input .+= input_decay .* params.w_input
    end
    return (; w_decay, input_decay)
end

"""Flush worker gradients into one Adam-ready minibatch gradient."""
function flush_manager_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in StatefulAlgorithms.workers(manager)
        ctx = worker_context(worker)
        add_buffer!(manager.state.batch_gradient, ctx.buffers)
        if uses_gradient_quality_nudging(manager.config)
            manager.state.gradient_quality_total[] += ctx.quality_stats.total[]
            manager.state.gradient_quality_accepted[] += ctx.quality_stats.accepted[]
            manager.state.gradient_quality_rejected[] += ctx.quality_stats.rejected[]
            manager.state.gradient_quality_sum_cosine[] += ctx.quality_stats.sum_cosine[]
            manager.state.gradient_quality_sum_angle_degrees[] += ctx.quality_stats.sum_angle_degrees[]
        end
        clear_buffer!(ctx.buffers)
    end

    nsamples = manager.state.nsamples[]
    nsamples > 0 || throw(ArgumentError("cannot flush an empty MNIST minibatch"))
    denominator = uses_gradient_quality_nudging(manager.config) ? manager.state.gradient_quality_accepted[] : nsamples
    if denominator <= 0
        clear_buffer!(manager.state.batch_gradient)
        return manager.state.batch_gradient
    end
    scale_buffer!(manager.state.batch_gradient, inv(contrastive_beta_scale(manager.config) * FT(denominator)))
    symmetrize_sparse_parameter_values!(
        manager.state.batch_gradient.w,
        II.adj(manager.state.source_graph),
        manager.state.reverse_sparse_ptrs,
    )
    manager.config.project_output_bias_prior &&
        project_output_bias_prior!(manager.state.batch_gradient, manager.state.layer, manager.config.output_replicas)
    apply_weight_decay!(manager.state.batch_gradient, manager.state.params[], manager.config)
    return manager.state.batch_gradient
end

"""Flush validation worker counters into the validation manager state."""
function flush_validation_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    manager.state.nsamples[] = 0
    manager.state.ncorrect[] = 0
    manager.state.total_loss[] = 0f0
    fill!(manager.state.pred_counts, 0)
    for worker in StatefulAlgorithms.workers(manager)
        ctx = worker_context(worker)
        manager.state.nsamples[] += ctx.nsamples[]
        manager.state.ncorrect[] += ctx.ncorrect[]
        manager.state.total_loss[] += ctx.total_loss[]
        manager.state.pred_counts .+= ctx.pred_counts
        reset_validation_worker_stats!(ctx)
    end
    return manager.state
end

"""Synchronize the source graph and shared worker fields after an Adam update."""
function sync_after_update!(manager::M, params::P) where {M<:StatefulAlgorithms.ProcessManager,P}
    symmetrize_sparse_parameter_values!(params.w, II.adj(manager.state.source_graph), manager.state.reverse_sparse_ptrs)
    normalize_sparse_couplings!(params.w, manager.config)
    normalize_input_projection!(params.w_input, manager.config)
    IsingLearning.sync_graph_params!(manager.state.source_graph, (; w = params.w, b = params.b))
    manager.state.input_hidden_w[] = params.w_input
    for worker in StatefulAlgorithms.workers(manager)
        IsingLearning._sync_worker_graph_params!(worker_graph(worker), manager.state.source_graph, (; w = params.w, b = params.b))
    end
    return manager
end

"""Construct the ProcessManager that owns persistent input-field workers."""
function input_field_manager(
    layer::L,
    source::G,
    config::C,
    input_hidden_w::R,
) where {L<:IsingLearning.LayeredIsingGraphLayer,G,C<:InputFieldMNISTConfig,R<:Base.RefValue}
    reverse_ptrs = sparse_reverse_ptrs(II.adj(source))
    params = input_field_params(source, input_hidden_w[])
    symmetrize_sparse_parameter_values!(params.w, II.adj(source), reverse_ptrs)
    normalize_sparse_couplings!(params.w, config)
    normalize_input_projection!(params.w_input, config)
    input_hidden_w[] = params.w_input
    optimiser = Optimisers.Adam(config.lr)
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(layer, config))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        input_field_gradient_buffer(source, input_hidden_w[]),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0.0),
        Ref(0.0),
        Optimisers.setup(optimiser, params),
        input_hidden_w,
        reverse_ptrs,
    )
    recipe = (;
        makeworker = (idx, manager) -> input_field_worker(
            algorithm,
            manager.state.layer,
            shared_worker_graph(manager.state.source_graph),
            manager.state.input_hidden_w,
        ),
        loadjob! = load_input_field_training_job!,
        providearguments = (slot, job, manager) -> (; phase_beta = manager.config.β),
        close! = close_process_worker!,
        sync_to_state! = flush_manager_buffers!,
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        sync_policy = StatefulAlgorithms.SyncAtEnd(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        execution = StatefulAlgorithms.ChannelWorkers(),
        poll_interval = 0.0,
        job_type = InputFieldMNISTJob{Vector{FT},Vector{FT}},
    )
end

"""Construct the ProcessManager that owns persistent validation workers."""
function input_field_validation_manager(
    layer::L,
    source::G,
    config::C,
    input_hidden_w::R,
) where {L<:IsingLearning.LayeredIsingGraphLayer,G,C<:InputFieldMNISTConfig,R<:Base.RefValue}
    algorithm = StatefulAlgorithms.resolve(input_field_validation_algorithm(layer))
    state = InputFieldMNISTEvalManagerState(
        layer,
        source,
        Ref(0),
        Ref(0),
        Ref(0f0),
        zeros(Int, NCLASSES),
        input_hidden_w,
    )
    recipe = (;
        makeworker = (idx, manager) -> input_field_validation_worker(
            algorithm,
            manager.state.layer,
            shared_worker_graph(manager.state.source_graph),
            manager.state.input_hidden_w,
        ),
        loadjob! = load_input_field_validation_job!,
        close! = close_process_worker!,
        sync_to_state! = flush_validation_buffers!,
    )
    return StatefulAlgorithms.ProcessManager(
        recipe;
        nworkers = config.workers,
        config,
        state,
        sync_policy = StatefulAlgorithms.SyncAtEnd(),
        worker_init = StatefulAlgorithms.MakeEachWorker(),
        execution = StatefulAlgorithms.ChannelWorkers(),
        poll_interval = 0.0,
        job_type = InputFieldMNISTJob{Vector{FT},Vector{FT}},
    )
end

"""Run one minibatch through the manager and apply one Adam update."""
function run_minibatch!(manager::M, jobs::J) where {M<:StatefulAlgorithms.ProcessManager,J<:AbstractVector}
    clear_manager_buffers!(manager)
    manager.state.nsamples[] = length(jobs)
    StatefulAlgorithms.run!(manager, jobs)
    if uses_gradient_quality_nudging(manager.config) && manager.state.gradient_quality_accepted[] == 0
        return manager.state.params[]
    end
    manager.state.opt_state, params = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    manager.state.params[] = params
    sync_after_update!(manager, params)
    return params
end

"""Evaluate accuracy and squared error for a prepared set of validation jobs."""
function evaluate_jobs(manager::M, jobs::J) where {
    M<:StatefulAlgorithms.ProcessManager,
    J<:AbstractVector,
}
    clear_validation_buffers!(manager)
    StatefulAlgorithms.run!(manager, jobs)
    nsamples = manager.state.nsamples[]
    return (;
        accuracy = nsamples == 0 ? 0.0 : manager.state.ncorrect[] / nsamples,
        loss = nsamples == 0 ? zero(FT) : manager.state.total_loss[] / nsamples,
        pred_counts = copy(manager.state.pred_counts),
    )
end

"""Evaluate accuracy and squared error using preallocated validation-manager jobs."""
function evaluate(
    manager::M,
    x::X,
    y::Y,
    config::C,
    jobs::B,
) where {M<:StatefulAlgorithms.ProcessManager,X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig,B<:InputFieldMNISTJobBuffer}
    active_jobs = fill_jobs!(jobs, x, y, collect(axes(x, 2)))
    return evaluate_jobs(manager, active_jobs)
end

"""Evaluate accuracy with a lightweight progress message for larger validation sets."""
function evaluate_with_progress(
    manager::M,
    x::X,
    y::Y,
    config::C,
    jobs::B,
    label::S,
) where {
    M<:StatefulAlgorithms.ProcessManager,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:InputFieldMNISTConfig,
    B<:InputFieldMNISTJobBuffer,
    S<:AbstractString,
}
    nsamples = size(x, 2)
    println(label, ": evaluating ", nsamples, " samples")
    flush(stdout)
    return evaluate(manager, x, y, config, jobs)
end

"""Serialize optimizer-facing parameters and run metadata."""
function save_checkpoint(path::P, manager::M, config::C, rows::R) where {P<:AbstractString,M<:StatefulAlgorithms.ProcessManager,C<:InputFieldMNISTConfig,R<:AbstractVector}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (;
            architecture = "784-$(config.hidden)-$(NCLASSES * config.output_replicas)",
            params = manager.state.params[],
            opt_state = manager.state.opt_state,
            rows,
            config = config_namedtuple(config),
        ))
    end
    return path
end

"""Load `CairoMakie` only when a plot needs to be written."""
function ensure_cairomakie()
    mod = @__MODULE__
    isdefined(mod, :CairoMakie) || Base.invokelatest(Core.eval, mod, :(using CairoMakie))
    CM = Base.invokelatest(getfield, mod, :CairoMakie)
    Base.invokelatest(CM.activate!)
    return CM
end

"""Return the sidecar path used for checkpoint schema recovery."""
function recovered_checkpoint_path(path::P) where {P<:AbstractString}
    stem, ext = splitext(path)
    return stem * "_recovered" * ext
end

"""Recover a checkpoint saved with the pre-reset-option config layout."""
function recover_legacy_checkpoint!(recovered_path::P, path::Q) where {P<:AbstractString,Q<:AbstractString}
    project_path = abspath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
    julia_path = joinpath(Sys.BINDIR, Base.julia_exename())
    source = """
    using Optimisers
    using Serialization

    struct InputFieldMNISTConfig{T<:AbstractFloat,S<:AbstractString}
        workers::Int
        epochs::Int
        batchsize::Int
        scheduler::S
        nudge_mode::S
        nudge_temp_schedule::S
        chunk_size::Int
        train_per_class::Int
        test_per_class::Int
        train_eval_per_class::Int
        eval_every::Int
        progress_every_batches::Int
        early_stop_decline_epochs::Int
        early_stop_decline_batches::Int
        early_stop_min_batches::Int
        hidden::Int
        output_replicas::Int
        sweeps::T
        β::T
        nudge_tau::T
        lr::T
        w_normalization::S
        w_norm::T
        project_output_bias_prior::Bool
        positive_target_weight::T
        negative_target_weight::T
        w_input_normalization::S
        w_input_row_norm::T
        temp::T
        nudge_temp_peak::T
        stepsize::T
        weight_scale::T
        weight_decay::T
        adaptive_weight_decay::Bool
        adaptive_w_norm::T
        adaptive_w_input_norm::T
        adaptive_decay_gain::T
        adaptive_max_decay::T
        skip_worker_close::Bool
        seed::Int
        resume_from::S
        resume_epoch::Int
        outdir::S
    end

    checkpoint = deserialize(IOBuffer(read($(repr(path)))))
    names = fieldnames(typeof(checkpoint.config))
    config = NamedTuple{names}(Tuple(getfield(checkpoint.config, name) for name in names))
    open($(repr(recovered_path)), "w") do io
        serialize(io, (
            architecture = checkpoint.architecture,
            params = checkpoint.params,
            opt_state = checkpoint.opt_state,
            rows = checkpoint.rows,
            config = config,
        ))
    end
    """
    run(Cmd([julia_path, "--project=$(project_path)", "-e", source]))
    return recovered_path
end

"""Deserialize one retained checkpoint and verify that it exists."""
function load_checkpoint(path::P) where {P<:AbstractString}
    isfile(path) || throw(ArgumentError("resume checkpoint does not exist: $(path)"))
    try
        return deserialize(IOBuffer(read(path)))
    catch err
        recovered_path = recovered_checkpoint_path(path)
        if !isfile(recovered_path) || stat(recovered_path).mtime < stat(path).mtime
            @warn "recovering legacy checkpoint format" checkpoint = path recovered = recovered_path
            recover_legacy_checkpoint!(recovered_path, path)
        end
        try
            return deserialize(IOBuffer(read(recovered_path)))
        catch recover_err
            error("failed to load checkpoint $(path): $(sprint(showerror, err)); recovery also failed: $(sprint(showerror, recover_err))")
        end
    end
end

"""Return the best test accuracy retained inside one checkpoint."""
function checkpoint_best_accuracy(checkpoint)
    if hasproperty(checkpoint, :rows)
        best = -Inf
        for row in checkpoint.rows
            if hasproperty(row, :test_accuracy) && !ismissing(row.test_accuracy)
                best = max(best, Float64(row.test_accuracy))
            end
        end
        return best
    end
    return -Inf
end

"""Infer the first epoch to run after a restored checkpoint."""
function checkpoint_start_epoch(checkpoint, config::C) where {C<:InputFieldMNISTConfig}
    if config.resume_epoch >= 0
        return config.resume_epoch
    end
    if hasproperty(checkpoint, :rows) && !isempty(checkpoint.rows)
        return Int(checkpoint.rows[end].epoch) + 1
    end
    return 0
end

"""Restore graph parameters and optimiser state into an already constructed manager."""
function restore_manager_checkpoint!(manager::M, checkpoint) where {M<:StatefulAlgorithms.ProcessManager}
    manager.state.params[] = checkpoint.params
    manager.state.opt_state = checkpoint.opt_state
    sync_after_update!(manager, checkpoint.params)
    return manager
end

"""Rebuild Adam state after restoring checkpoint parameters when a resumed branch changes LR."""
function reset_manager_optimizer!(manager::M, config::C) where {M<:StatefulAlgorithms.ProcessManager,C<:InputFieldMNISTConfig}
    manager.state.opt_state = Optimisers.setup(Optimisers.Adam(config.lr), manager.state.params[])
    return manager
end

"""Return the default file paths used by one retained run directory."""
function session_paths(outdir::P) where {P<:AbstractString}
    return (
        csv_path = joinpath(outdir, "mnist_784_120_40_softplus_margin_nudge_adam.csv"),
        best_path = joinpath(outdir, "best_checkpoint.bin"),
        latest_path = joinpath(outdir, "latest_checkpoint.bin"),
        final_path = joinpath(outdir, "final_checkpoint.bin"),
    )
end

"""Update the session config and derived output paths without rebuilding the manager."""
function sync_session_config!(session::S, config::C) where {S<:InputFieldMNISTSession,C<:InputFieldMNISTConfig}
    session.config = config
    session.manager.config = config
    paths = session_paths(config.outdir)
    session.csv_path = paths.csv_path
    session.best_path = paths.best_path
    session.latest_path = paths.latest_path
    session.final_path = paths.final_path
    return session
end

"""Restore one live session from a checkpoint and continue in the same Julia process."""
function restart_from_checkpoint!(
    session::S,
    checkpoint_path::P;
    resume_epoch::Int = -1,
    beta::T = session.config.β,
    lr::T = session.config.lr,
    outdir::Q = session.config.outdir,
) where {S<:InputFieldMNISTSession,P<:AbstractString,Q<:AbstractString,T<:Real}
    checkpoint = load_checkpoint(checkpoint_path)
    config = updated_config(
        session.config;
        β = FT(beta),
        lr = FT(lr),
        resume_from = String(checkpoint_path),
        resume_epoch = resume_epoch,
        outdir = String(outdir),
    )
    mkpath(config.outdir)

    # Reset manager/optimizer state in place, then point the session at the new run directory.
    restore_manager_checkpoint!(session.manager, checkpoint)
    config.reset_opt_state_on_resume && reset_manager_optimizer!(session.manager, config)
    sync_session_config!(session, config)
    clear_manager_buffers!(session.manager)

    # Carry forward prior metrics so the resumed run stays continuous on disk.
    session.rows = collect(checkpoint.rows)
    session.best_accuracy[] = checkpoint_best_accuracy(checkpoint)
    write_rows_csv!(session.csv_path, session.rows)
    write_settings!(joinpath(config.outdir, "settings.md"), config, session.manager.state.layer.free_relaxation_steps)
    return checkpoint_start_epoch(checkpoint, config)
end

"""Plot train/test accuracy and loss curves for a retained run."""
function plot_metrics(path::P, rows::R) where {P<:AbstractString,R<:AbstractVector}
    CM = ensure_cairomakie()
    fig = CM.Figure(size = (1200, 760))
    ax_acc = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "MNIST 784-120-40 softplus-margin nudge accuracy")
    ax_loss = CM.Axis(fig[2, 1], xlabel = "epoch", ylabel = "loss", title = "Mean squared output error")
    train_rows = [row for row in rows if !ismissing(row.train_accuracy)]
    if !isempty(train_rows)
        CM.lines!(ax_acc, [row.epoch for row in train_rows], [row.train_accuracy for row in train_rows], label = "train", color = :steelblue)
        CM.lines!(ax_loss, [row.epoch for row in train_rows], [row.train_loss for row in train_rows], label = "train", color = :steelblue)
    end
    test_rows = [row for row in rows if !ismissing(row.test_accuracy)]
    if !isempty(test_rows)
        CM.lines!(ax_acc, [row.epoch for row in test_rows], [row.test_accuracy for row in test_rows], label = "test", color = :orange)
        CM.lines!(ax_loss, [row.epoch for row in test_rows], [row.test_loss for row in test_rows], label = "test", color = :orange)
    end
    CM.axislegend(ax_acc, position = :rb)
    CM.save(path, fig)
    return path
end

"""Write the run settings needed to reproduce the softplus-margin nudge baseline."""
function write_settings!(path::P, config::C, relaxation_steps::I) where {P<:AbstractString,C<:InputFieldMNISTConfig,I<:Integer}
    open(path, "w") do io
        println(io, "# MNIST 784-120-40 Softplus-Margin Gradient Quality Adam Diagnostic")
        println(io)
        println(io, "- paper: https://arxiv.org/pdf/2305.18321")
        println(io, "- nudging term: `InteractiveIsing.SoftplusMarginNudging`")
        println(io, "- architecture: `784 -> $(config.hidden) -> $(NCLASSES * config.output_replicas)`")
        println(io, "- sampled graph: hidden/output only, with no structural input layer")
        println(io, "- input handling: MNIST pixels in `[0, 1]` are projected through external `784 -> hidden` weights into a worker-local field")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- manager execution: `ChannelWorkers()` with one sample per manager job")
        println(io, "- progress print interval: `$(config.progress_every_batches)` batches")
        println(io, "- early stop decline epochs: `$(config.early_stop_decline_epochs)`")
        println(io, "- early stop decline batches: `$(config.early_stop_decline_batches)`")
        println(io, "- early stop minimum batches: `$(config.early_stop_min_batches)`")
        println(io, "- worker graph adjacency: pointer-shared with source graph")
        println(io, "- worker graph base bias: pointer-shared with source graph")
        println(io, "- learning step/nudge mode: `$(config.nudge_mode)` contrastive gradient from inline `LoopAlgorithm`")
        println(io, "- sparse coupling constraint: paired CSC entries are symmetrized before and after Adam updates")
        println(io, "- validation: `ChannelWorkers()` ProcessManager with worker-local stats")
        println(io, "- job buffers: preallocated per-sample jobs reused across minibatches/evaluations")
        println(io, "- optimiser: `Optimisers.Adam($(config.lr))`")
        println(io, "- epochs/batchsize: `$(config.epochs)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- train eval per class: `$(config.train_eval_per_class)`")
        println(io, "- sweeps/relaxation steps: `$(config.sweeps)` / `$(relaxation_steps)`")
        println(io, "- beta/nudge tau/temp/stepsize: `$(config.β)` / `$(config.nudge_tau)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- nudged temperature schedule/peak: `$(config.nudge_temp_schedule)` / `$(config.nudge_temp_peak)`")
        println(io, "- gradient quality min cosine: `$(config.gradient_quality_min_cosine)`")
        println(io, "- gradient quality rule: compare raw forward `plus-free` gradient with raw symmetric `plus-minus` gradient; accept and average symmetric gradients only when cosine is at least the threshold")
        println(io, "- recurrent w normalization/norm: `$(config.w_normalization)` / `$(config.w_norm)`")
        println(io, "- project output bias prior: `$(config.project_output_bias_prior)`")
        println(io, "- positive/negative target mask weights: `$(config.positive_target_weight)` / `$(config.negative_target_weight)`")
        println(io, "- w_input normalization/row norm: `$(config.w_input_normalization)` / `$(config.w_input_row_norm)`")
        println(io, "- weight scale/decay: `$(config.weight_scale)` / `$(config.weight_decay)`")
        println(io, "- adaptive weight decay: `$(config.adaptive_weight_decay)`")
        println(io, "- adaptive decay targets w/w_input: `$(config.adaptive_w_norm)` / `$(config.adaptive_w_input_norm)`")
        println(io, "- adaptive decay gain/max: `$(config.adaptive_decay_gain)` / `$(config.adaptive_max_decay)`")
        println(io, "- resume from: `$(isempty(config.resume_from) ? "none" : config.resume_from)`")
        println(io, "- resume epoch: `$(config.resume_epoch)`")
        println(io, "- reset optimizer state on resume: `$(config.reset_opt_state_on_resume)`")
    end
    return path
end

"""Validate softplus-margin nudge baseline configuration before building runtime state."""
function validate_config!(config::C) where {C<:InputFieldMNISTConfig}
    config.workers > 0 || throw(ArgumentError("ISING_MNIST_IF_WORKERS must be positive"))
    config.batchsize > 0 || throw(ArgumentError("ISING_MNIST_IF_BATCHSIZE must be positive"))
    config.epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EPOCHS must be nonnegative"))
    config.progress_every_batches >= 0 || throw(ArgumentError("ISING_MNIST_IF_PROGRESS_EVERY_BATCHES must be nonnegative"))
    config.early_stop_decline_epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS must be nonnegative"))
    config.early_stop_decline_batches >= 0 || throw(ArgumentError("ISING_MNIST_IF_EARLY_STOP_DECLINE_BATCHES must be nonnegative"))
    config.early_stop_min_batches >= 0 || throw(ArgumentError("ISING_MNIST_IF_EARLY_STOP_MIN_BATCHES must be nonnegative"))
    String(config.nudge_mode) in ("one_sided", "symmetric", "gradient_quality") ||
        throw(ArgumentError("ISING_MNIST_IF_NUDGE_MODE must be one_sided, symmetric, or gradient_quality"))
    String(config.nudge_temp_schedule) in ("fixed", "reverse_anneal") ||
        throw(ArgumentError("ISING_MNIST_IF_NUDGE_TEMP_SCHEDULE must be fixed or reverse_anneal"))
    config.nudge_tau > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_NUDGE_TAU must be positive"))
    -one(FT) <= config.gradient_quality_min_cosine <= one(FT) ||
        throw(ArgumentError("ISING_MNIST_IF_GRADIENT_QUALITY_MIN_COSINE must be in [-1, 1]"))
    String(config.w_normalization) in ("none", "global") ||
        throw(ArgumentError("ISING_MNIST_IF_W_NORMALIZATION must be one of none, global"))
    config.w_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_W_NORM must be positive"))
    String(config.w_input_normalization) in ("none", "row", "global") ||
        throw(ArgumentError("ISING_MNIST_IF_W_INPUT_NORMALIZATION must be one of none, row, global"))
    config.positive_target_weight > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_POSITIVE_TARGET_WEIGHT must be positive"))
    config.negative_target_weight >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_NEGATIVE_TARGET_WEIGHT must be nonnegative"))
    config.w_input_row_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_W_INPUT_ROW_NORM must be positive"))
    config.temp > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_TEMP must be positive for Langevin noise"))
    config.nudge_temp_peak >= config.temp ||
        throw(ArgumentError("ISING_MNIST_IF_NUDGE_TEMP_PEAK must be at least ISING_MNIST_IF_TEMP"))
    config.weight_decay >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_WEIGHT_DECAY must be nonnegative"))
    config.adaptive_w_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_ADAPTIVE_W_NORM must be positive"))
    config.adaptive_w_input_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_ADAPTIVE_W_INPUT_NORM must be positive"))
    config.adaptive_decay_gain >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_ADAPTIVE_DECAY_GAIN must be nonnegative"))
    config.adaptive_max_decay >= config.weight_decay ||
        throw(ArgumentError("ISING_MNIST_IF_ADAPTIVE_MAX_DECAY must be at least ISING_MNIST_IF_WEIGHT_DECAY"))
    config.hidden == 120 || @warn "baseline paper hidden count is 120" hidden = config.hidden
    config.output_replicas == 4 || @warn "baseline paper output count is 40, i.e. four replicas per digit" output_replicas = config.output_replicas
    config.train_per_class < 5421 && @warn "this run uses a subsampled balanced training split, so it is not the full balanced MNIST baseline" train_per_class = config.train_per_class
    config.test_per_class < 892 && @warn "this run uses a subsampled balanced test split, so reported accuracy will be noisy" test_per_class = config.test_per_class
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    has_resume_checkpoint(config) && !isfile(config.resume_from) &&
        throw(ArgumentError("ISING_MNIST_IF_RESUME_FROM does not point to a checkpoint file"))
    return config
end

"""Construct one reusable softplus-margin nudge session that can be rerun in-process."""
function create_session(config::C) where {C<:InputFieldMNISTConfig}
    validate_config!(config)
    mkpath(config.outdir)
    println("building MNIST 784-120-40 layer")
    flush(stdout)
    setup = build_layer(config)
    write_settings!(joinpath(config.outdir, "settings.md"), config, setup.relaxation_steps)
    println("loading training split")
    flush(stdout)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    println("training samples: ", size(xtrain, 2), " (", cld(size(xtrain, 2), config.batchsize), " batches/epoch)")
    flush(stdout)
    println("loading train-eval split")
    flush(stdout)
    xtrain_eval, ytrain_eval = config.train_eval_per_class > 0 ?
        balanced_mnist(:train, config.train_eval_per_class, config) :
        (zeros(FT, INPUT_DIM, 0), zeros(FT, NCLASSES * config.output_replicas, 0))
    println("train-eval samples: ", size(xtrain_eval, 2))
    flush(stdout)
    println("loading test split")
    flush(stdout)
    xtest, ytest = balanced_mnist(:test, config.test_per_class, config)
    println("test samples: ", size(xtest, 2))
    flush(stdout)
    input_hidden_w = Ref(setup.input_hidden_w)
    println("constructing ", config.workers, "-worker softplus-margin nudge Adam manager")
    flush(stdout)
    manager = input_field_manager(setup.layer, setup.graph, config, input_hidden_w)
    println("constructing ", config.workers, "-worker validation manager")
    flush(stdout)
    validator = input_field_validation_manager(setup.layer, setup.graph, config, input_hidden_w)
    train_jobs = InputFieldMNISTJobBuffer(config.batchsize, INPUT_DIM, NCLASSES * config.output_replicas)
    eval_capacity = max(size(xtrain_eval, 2), size(xtest, 2), 1)
    eval_jobs = InputFieldMNISTJobBuffer(eval_capacity, INPUT_DIM, NCLASSES * config.output_replicas)
    println("starting epochs")
    flush(stdout)
    paths = session_paths(config.outdir)
    return InputFieldMNISTSession(
        config,
        xtrain,
        ytrain,
        xtrain_eval,
        ytrain_eval,
        xtest,
        ytest,
        manager,
        validator,
        train_jobs,
        eval_jobs,
        NamedTuple[],
        Ref(-Inf),
        paths.csv_path,
        paths.best_path,
        paths.latest_path,
        paths.final_path,
    )
end

"""Run softplus-margin nudge baseline epochs on an already prepared session."""
function run_epochs!(session::S; start_epoch::Int = 0, stop_epoch::Int = session.config.epochs) where {S<:InputFieldMNISTSession}
    config = session.config
    prev_eval_accuracy = Ref{Union{Nothing,Float64}}(nothing)
    decline_streak = Ref(0)
    best_batch_accuracy = Ref(-Inf)
    prev_batch_accuracy = Ref{Union{Nothing,Float64}}(nothing)
    batch_decline_streak = Ref(0)
    trained_batches = Ref(0)
    for epoch in start_epoch:stop_epoch
        seconds = 0.0
        train_accuracy = missing
        train_loss = missing
        test_accuracy = missing
        test_loss = missing
        pred_counts = missing
        is_new_best = false
        should_stop_early = false
        epoch_quality_total = 0
        epoch_quality_accepted = 0
        epoch_quality_rejected = 0
        epoch_quality_sum_cosine = 0.0
        epoch_quality_sum_angle_degrees = 0.0
        if epoch > 0
            println("epoch ", epoch, " training")
            flush(stdout)
            order = Random.shuffle(Random.MersenneTwister(config.seed + epoch), collect(axes(session.xtrain, 2)))
            nbatches = cld(length(order), config.batchsize)
            progress_every = Int(config.progress_every_batches)
            progress = progress_every > 0 ? ProgressMeter.Progress(nbatches; desc = "epoch $(epoch) ", dt = 1.0) : nothing
            seconds = @elapsed begin
                epoch_start = time()
                for (batch_idx, first_idx) in enumerate(1:config.batchsize:length(order))
                    last_idx = min(first_idx + config.batchsize - 1, length(order))
                    batch_indices = @view order[first_idx:last_idx]
                    jobs = fill_jobs!(
                        session.train_jobs,
                        session.xtrain,
                        session.ytrain,
                        batch_indices,
                    )
                    run_minibatch!(session.manager, jobs)
                    if uses_gradient_quality_nudging(config)
                        quality = training_quality(session.manager)
                        epoch_quality_total += quality.gradient_quality_total
                        epoch_quality_accepted += quality.gradient_quality_accepted
                        epoch_quality_rejected += quality.gradient_quality_rejected
                        epoch_quality_sum_cosine += isnan(quality.gradient_quality_avg_cosine) ?
                            0.0 :
                            quality.gradient_quality_avg_cosine * quality.gradient_quality_total
                        epoch_quality_sum_angle_degrees += isnan(quality.gradient_quality_avg_angle_degrees) ?
                            0.0 :
                            quality.gradient_quality_avg_angle_degrees * quality.gradient_quality_total
                    end
                    trained_batches[] += 1
                    if config.early_stop_decline_batches > 0
                        batch_eval = evaluate_jobs(session.validator, jobs)
                        batch_accuracy = Float64(batch_eval.accuracy)
                        if batch_accuracy > best_batch_accuracy[]
                            best_batch_accuracy[] = batch_accuracy
                        end
                        if trained_batches[] >= config.early_stop_min_batches && !isnothing(prev_batch_accuracy[])
                            if batch_accuracy < prev_batch_accuracy[]
                                batch_decline_streak[] += 1
                            else
                                batch_decline_streak[] = 0
                            end
                        end
                        prev_batch_accuracy[] = batch_accuracy
                        println(
                            "epoch ", epoch, " batch ", batch_idx, "/", nbatches,
                            " accuracy ", round(batch_accuracy; digits = 4),
                            " loss ", round(Float64(batch_eval.loss); digits = 6),
                            " best_batch ", round(best_batch_accuracy[]; digits = 4),
                            " declines ", batch_decline_streak[], "/", config.early_stop_decline_batches,
                            " trained_batches ", trained_batches[],
                        )
                        flush(stdout)
                        if trained_batches[] >= config.early_stop_min_batches &&
                                batch_decline_streak[] >= config.early_stop_decline_batches
                            should_stop_early = true
                            println(
                                "epoch ", epoch, " early stop: batch accuracy declined for ",
                                batch_decline_streak[], " consecutive batches",
                            )
                            flush(stdout)
                            break
                        end
                    end
                    if progress !== nothing
                        ProgressMeter.next!(
                            progress;
                            showvalues = [
                                (:batch, "$(batch_idx)/$(nbatches)"),
                                (:samples, "$(last_idx)/$(length(order))"),
                            ],
                        )
                    end
                    if progress_every > 0 && (batch_idx == 1 || batch_idx == nbatches || mod(batch_idx, progress_every) == 0)
                        elapsed = time() - epoch_start
                        quality_suffix = ""
                        if uses_gradient_quality_nudging(config)
                            quality = training_quality(session.manager)
                            quality_suffix = " accepted $(quality.gradient_quality_accepted)/$(quality.gradient_quality_total) reject_fraction $(round(quality.gradient_quality_reject_fraction; digits = 3)) avg_angle $(round(quality.gradient_quality_avg_angle_degrees; digits = 2))"
                        end
                        println(
                            "epoch ", epoch, " batch ", batch_idx, "/", nbatches,
                            " samples ", last_idx, "/", length(order),
                            " elapsed ", round(elapsed; digits = 1), "s",
                            quality_suffix,
                        )
                        flush(stdout)
                    end
                end
            end
            println("epoch ", epoch, " training done in ", round(seconds; digits = 3), "s")
            flush(stdout)
        end

        # Validation is intentionally not run every epoch; on larger held-out sets it can dominate training time.
        should_eval_train = config.train_eval_per_class > 0 &&
            (epoch == 0 || epoch == config.epochs || (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0))
        if should_eval_train
            train = evaluate_with_progress(
                session.validator,
                session.xtrain_eval,
                session.ytrain_eval,
                config,
                session.eval_jobs,
                "epoch $(epoch) train-eval",
            )
            train_accuracy = train.accuracy
            train_loss = train.loss
            println(
                "epoch ", epoch,
                " train accuracy ", round(Float64(train_accuracy); digits = 4),
                " loss ", round(Float64(train_loss); digits = 6),
            )
            flush(stdout)
        end
        should_eval_test = epoch == 0 || epoch == config.epochs ||
            (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0)
        if should_eval_test
            println("epoch ", epoch, " evaluating")
            flush(stdout)
            test = evaluate_with_progress(
                session.validator,
                session.xtest,
                session.ytest,
                config,
                session.eval_jobs,
                "epoch $(epoch) test",
            )
            test_accuracy = test.accuracy
            test_loss = test.loss
            pred_counts = join(test.pred_counts, "-")
            println(
                "epoch ", epoch,
                " test accuracy ", round(Float64(test_accuracy); digits = 4),
                " loss ", round(Float64(test_loss); digits = 6),
                " best ", round(Float64(max(session.best_accuracy[], test_accuracy)); digits = 4),
            )
            flush(stdout)
            if test.accuracy > session.best_accuracy[]
                session.best_accuracy[] = test.accuracy
                is_new_best = true
            end
            if config.early_stop_decline_epochs > 0 && epoch > 0
                previous = prev_eval_accuracy[]
                if previous !== nothing && Float64(test.accuracy) < previous
                    decline_streak[] += 1
                else
                    decline_streak[] = 0
                end
                prev_eval_accuracy[] = Float64(test.accuracy)
                if decline_streak[] >= config.early_stop_decline_epochs
                    should_stop_early = true
                    println(
                        "epoch ", epoch, " early stop: test accuracy declined for ",
                        decline_streak[], " consecutive evaluated epochs",
                    )
                    flush(stdout)
                end
            elseif should_eval_test
                prev_eval_accuracy[] = Float64(test.accuracy)
            end
        end

        norms = training_norms(session.manager)
        gradient_quality_reject_fraction = epoch_quality_total == 0 ? 0.0 : epoch_quality_rejected / epoch_quality_total
        gradient_quality_avg_cosine = epoch_quality_total == 0 ? NaN : epoch_quality_sum_cosine / epoch_quality_total
        gradient_quality_avg_angle_degrees = epoch_quality_total == 0 ? NaN : epoch_quality_sum_angle_degrees / epoch_quality_total
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            epoch,
            seconds,
            train_accuracy,
            train_loss,
            test_accuracy,
            test_loss,
            pred_counts,
            norms.w_norm,
            norms.b_norm,
            norms.w_input_norm,
            norms.grad_w_norm,
            norms.grad_b_norm,
            norms.grad_w_input_norm,
            norms.effective_w_decay,
            norms.effective_w_input_decay,
            norms.symmetry_error,
            norms.grad_symmetry_error,
            gradient_quality_total = epoch_quality_total,
            gradient_quality_accepted = epoch_quality_accepted,
            gradient_quality_rejected = epoch_quality_rejected,
            gradient_quality_reject_fraction,
            gradient_quality_avg_cosine,
            gradient_quality_avg_angle_degrees,
            best_accuracy = session.best_accuracy[],
            best_path = session.best_path,
            final_path = epoch == config.epochs ? session.final_path : "",
        )
        append_row!(session.csv_path, row)
        push!(session.rows, row)
        should_eval_test && save_checkpoint(session.latest_path, session.manager, config, session.rows)
        is_new_best && save_checkpoint(session.best_path, session.manager, config, session.rows)
        println(row)
        flush(stdout)
        should_stop_early && break
    end
    return session
end

"""Persist final artifacts for a completed softplus-margin nudge baseline session."""
function finalize_session!(session::S) where {S<:InputFieldMNISTSession}
    save_checkpoint(session.final_path, session.manager, session.config, session.rows)
    plot_metrics(joinpath(session.config.outdir, "learning_summary.png"), session.rows)
    println("saved MNIST 784-120-40 softplus-margin nudge baseline in ", session.config.outdir)
    return (; config = session.config, rows = session.rows, best_accuracy = session.best_accuracy[])
end

"""Close manager-backed state held by one live softplus-margin nudge session."""
function close_session!(session::S) where {S<:InputFieldMNISTSession}
    session.config.skip_worker_close && return nothing
    close(session.manager)
    close(session.validator)
    return nothing
end

"""Run the full softplus-margin nudge baseline experiment."""
function run_config!(config::C) where {C<:InputFieldMNISTConfig}
    session = create_session(config)
    try
        start_epoch = 0
        if has_resume_checkpoint(config)
            println("restoring checkpoint state from ", config.resume_from)
            flush(stdout)
            start_epoch = restart_from_checkpoint!(
                session,
                config.resume_from;
                resume_epoch = config.resume_epoch,
                beta = config.β,
                lr = config.lr,
                outdir = config.outdir,
            )
        end
        run_epochs!(session; start_epoch = start_epoch, stop_epoch = session.config.epochs)
        return finalize_session!(session)
    finally
        close_session!(session)
    end
end

"""Entry point for command-line runs."""
function main()
    return run_config!(updated_config(InputFieldMNISTConfig(), nudge_mode = "gradient_quality"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
