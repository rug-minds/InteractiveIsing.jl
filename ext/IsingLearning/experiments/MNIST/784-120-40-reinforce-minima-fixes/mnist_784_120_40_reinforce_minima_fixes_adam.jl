using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

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
    chunk_size::Int = parse(Int, get(ENV, "ISING_MNIST_IF_CHUNK_SIZE", "0"))
    train_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_PER_CLASS", "5421"))
    test_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TEST_PER_CLASS", "892"))
    train_eval_per_class::Int = parse(Int, get(ENV, "ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS", "100"))
    eval_every::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EVAL_EVERY", "5"))
    progress_every_batches::Int = parse(Int, get(ENV, "ISING_MNIST_IF_PROGRESS_EVERY_BATCHES", "25"))
    early_stop_decline_epochs::Int = parse(Int, get(ENV, "ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS", "0"))
    hidden::Int = parse(Int, get(ENV, "ISING_MNIST_IF_HIDDEN", "120"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS", "4"))
    sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_SWEEPS", "500"))
    β::T = parse(FT, get(ENV, "ISING_MNIST_IF_BETA", "5.0"))
    covariance_samples::Int = parse(Int, get(ENV, "ISING_MNIST_IF_COVARIANCE_SAMPLES", "20"))
    covariance_sample_sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_SAMPLE_SWEEPS", "1.0"))
    covariance_kick_steps::Int = parse(Int, get(ENV, "ISING_MNIST_IF_COVARIANCE_KICK_STEPS", "5"))
    covariance_stepsize::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_STEPSIZE", "1.0"))
    covariance_noise_temp_factor::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_NOISE_TEMP_FACTOR", "1.0"))
    reward_mode::S = get(ENV, "ISING_MNIST_IF_ATTRACTOR_REWARD", "logprob")
    reward_baseline::S = get(ENV, "ISING_MNIST_IF_REWARD_BASELINE", "batch")
    lr::T = parse(FT, get(ENV, "ISING_MNIST_IF_LR", "0.003"))
    gradient_sign::T = parse(FT, get(ENV, "ISING_MNIST_IF_GRADIENT_SIGN", "1.0"))
    w_normalization::S = get(ENV, "ISING_MNIST_IF_W_NORMALIZATION", "none")
    w_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_W_NORM", "1.0"))
    project_output_bias_prior::Bool = parse(Bool, get(ENV, "ISING_MNIST_IF_PROJECT_OUTPUT_BIAS_PRIOR", "false"))
    w_input_normalization::S = get(ENV, "ISING_MNIST_IF_W_INPUT_NORMALIZATION", "none")
    w_input_row_norm::T = parse(FT, get(ENV, "ISING_MNIST_IF_W_INPUT_ROW_NORM", "0.14"))
    temp::T = parse(FT, get(ENV, "ISING_MNIST_IF_TEMP", "0.001"))
    stepsize::T = parse(FT, get(ENV, "ISING_MNIST_IF_STEPSIZE", "0.5"))
    weight_scale::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_SCALE", "0.005"))
    weight_decay::T = parse(FT, get(ENV, "ISING_MNIST_IF_WEIGHT_DECAY", "0.0"))
    seed::Int = parse(Int, get(ENV, "ISING_MNIST_IF_SEED", "20260526"))
    resume_from::S = get(ENV, "ISING_MNIST_IF_RESUME_FROM", "")
    resume_epoch::Int = parse(Int, get(ENV, "ISING_MNIST_IF_RESUME_EPOCH", "-1"))
    outdir::S = get(
        ENV,
        "ISING_MNIST_IF_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("mnist_784_120_40_reinforce_minima_fixes_adam")),
    )
end

struct InputFieldMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

struct InputFieldMNISTJobBuffer{J<:AbstractVector}
    jobs::J
end

mutable struct InputFieldMNISTManagerState{L,G,P,B,S,D,O,X,Y,W}
    layer::L
    source_graph::G
    params::Base.RefValue{P}
    batch_gradient::B
    batch_stat::S
    diagnostics::D
    nsamples::Base.RefValue{Int}
    opt_state::O
    current_x::Base.RefValue{X}
    current_y::Base.RefValue{Y}
    input_hidden_w::Base.RefValue{W}
end

mutable struct InputFieldMNISTEvalManagerState{L,G,P,X,Y,W}
    layer::L
    source_graph::G
    nsamples::Base.RefValue{Int}
    ncorrect::Base.RefValue{Int}
    total_loss::Base.RefValue{FT}
    pred_counts::P
    current_x::Base.RefValue{X}
    current_y::Base.RefValue{Y}
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
    hasproperty(buffer, :stat_w) && fill!(buffer.stat_w, zero(eltype(buffer.stat_w)))
    hasproperty(buffer, :stat_b) && fill!(buffer.stat_b, zero(eltype(buffer.stat_b)))
    hasproperty(buffer, :stat_w_input) && fill!(buffer.stat_w_input, zero(eltype(buffer.stat_w_input)))
    hasproperty(buffer, :reward_sum) && (buffer.reward_sum[] = zero(eltype(buffer.w)))
    hasproperty(buffer, :reward_sumsq) && (buffer.reward_sumsq[] = zero(eltype(buffer.w)))
    hasproperty(buffer, :attractor_samples) && (buffer.attractor_samples[] = 0)
    hasproperty(buffer, :w_by_label) && fill!(buffer.w_by_label, zero(eltype(buffer.w_by_label)))
    hasproperty(buffer, :b_by_label) && fill!(buffer.b_by_label, zero(eltype(buffer.b_by_label)))
    hasproperty(buffer, :w_input_by_label) && fill!(buffer.w_input_by_label, zero(eltype(buffer.w_input_by_label)))
    hasproperty(buffer, :stat_w_by_label) && fill!(buffer.stat_w_by_label, zero(eltype(buffer.stat_w_by_label)))
    hasproperty(buffer, :stat_b_by_label) && fill!(buffer.stat_b_by_label, zero(eltype(buffer.stat_b_by_label)))
    hasproperty(buffer, :stat_w_input_by_label) && fill!(buffer.stat_w_input_by_label, zero(eltype(buffer.stat_w_input_by_label)))
    hasproperty(buffer, :reward_sum_by_label) && fill!(buffer.reward_sum_by_label, zero(eltype(buffer.reward_sum_by_label)))
    hasproperty(buffer, :reward_sumsq_by_label) && fill!(buffer.reward_sumsq_by_label, zero(eltype(buffer.reward_sumsq_by_label)))
    hasproperty(buffer, :attractor_samples_by_label) && fill!(buffer.attractor_samples_by_label, 0)
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    return buffer
end

"""No-op reset hook for per-example surrogate-attractor state.

Worker buffers accumulate across the whole minibatch so that the manager can
subtract one batch-average reward baseline during synchronization.
"""
function clear_covariance_stats!(buffer::B) where {B}
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

"""Add one worker statistic sum buffer into a manager-owned statistic buffer."""
function add_stat_buffer!(dest::D, src::S) where {D,S}
    dest.w .+= src.stat_w
    dest.b .+= src.stat_b
    hasproperty(dest, :w_input) && (dest.w_input .+= src.stat_w_input)
    if hasproperty(dest, :w_by_label) && hasproperty(src, :stat_w_by_label)
        dest.w_by_label .+= src.stat_w_by_label
        dest.b_by_label .+= src.stat_b_by_label
        dest.w_input_by_label .+= src.stat_w_input_by_label
        dest.reward_sum_by_label .+= src.reward_sum_by_label
        dest.reward_sumsq_by_label .+= src.reward_sumsq_by_label
        dest.attractor_samples_by_label .+= src.attractor_samples_by_label
    end
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
    symmetry_error = sparse_parameter_symmetry_error(params.w, II.adj(manager.state.source_graph))
    grad_symmetry_error = sparse_parameter_symmetry_error(grad.w, II.adj(manager.state.source_graph))
    return (;
        w_norm = parameter_norm(params.w),
        b_norm = parameter_norm(params.b),
        w_input_norm,
        grad_w_norm = parameter_norm(grad.w),
        grad_b_norm = parameter_norm(grad.b),
        grad_w_input_norm,
        symmetry_error,
        grad_symmetry_error,
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

"""Point a manager at the matrices used by index-based jobs."""
function set_manager_inputs!(manager::M, x::X, y::Y) where {M<:StatefulAlgorithms.ProcessManager,X<:AbstractMatrix,Y<:AbstractMatrix}
    manager.state.current_x[] = x
    manager.state.current_y[] = y
    return manager
end

"""Copy one sample column into a worker-local input/target buffer."""
function load_sample_into_worker!(ctx::C, x::X, y::Y, sample_idx::I) where {C,X<:AbstractMatrix,Y<:AbstractMatrix,I<:Integer}
    copyto!(ctx.x[], view(x, :, Int(sample_idx)))
    copyto!(ctx.y[], view(y, :, Int(sample_idx)))
    return ctx
end

"""Create one gradient buffer matching the reduced field-input baseline parameters."""
function input_field_gradient_buffer(graph::G, input_hidden_w::W) where {G,W<:AbstractMatrix}
    return (;
        w = zeros(eltype(graph), length(SparseArrays.getnzval(II.adj(graph)))),
        b = zeros(eltype(graph), II.nstates(graph)),
        w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
    )
end

"""Create one batch statistic buffer matching the optimizer-facing parameters."""
function input_field_stat_buffer(graph::G, input_hidden_w::W) where {G,W<:AbstractMatrix}
    return (;
        w = zeros(eltype(graph), length(SparseArrays.getnzval(II.adj(graph)))),
        b = zeros(eltype(graph), II.nstates(graph)),
        w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
        w_by_label = zeros(eltype(graph), length(SparseArrays.getnzval(II.adj(graph))), NCLASSES),
        b_by_label = zeros(eltype(graph), II.nstates(graph), NCLASSES),
        w_input_by_label = zeros(eltype(input_hidden_w), length(input_hidden_w), NCLASSES),
        reward_sum_by_label = zeros(eltype(graph), NCLASSES),
        reward_sumsq_by_label = zeros(eltype(graph), NCLASSES),
        attractor_samples_by_label = zeros(Int, NCLASSES),
    )
end

"""Create scalar diagnostics for one synchronized minibatch gradient."""
function input_field_diagnostics_buffer(::Type{T}) where {T<:Real}
    return (;
        attractor_samples = Ref(0),
        reward_mean = Ref(zero(T)),
        reward_std = Ref(zero(T)),
    )
end

"""Create one worker buffer with online batch accumulators for attractor learning."""
function input_field_surrogate_worker_buffer(
    graph::G,
    input_hidden_w::W,
    sample_capacity::I,
) where {G,W<:AbstractMatrix,I<:Integer}
    _ = sample_capacity
    w_shape = length(SparseArrays.getnzval(II.adj(graph)))
    b_shape = II.nstates(graph)
    return (;
        w = zeros(eltype(graph), w_shape),
        b = zeros(eltype(graph), b_shape),
        w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
        stat_w = zeros(eltype(graph), w_shape),
        stat_b = zeros(eltype(graph), b_shape),
        stat_w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
        stat_w_by_label = zeros(eltype(graph), w_shape, NCLASSES),
        stat_b_by_label = zeros(eltype(graph), b_shape, NCLASSES),
        stat_w_input_by_label = zeros(eltype(input_hidden_w), length(input_hidden_w), NCLASSES),
        reward_sum_by_label = zeros(eltype(graph), NCLASSES),
        reward_sumsq_by_label = zeros(eltype(graph), NCLASSES),
        attractor_samples_by_label = zeros(Int, NCLASSES),
        reward_sum = Ref(zero(eltype(graph))),
        reward_sumsq = Ref(zero(eltype(graph))),
        attractor_samples = Ref(0),
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

"""Return the CSC storage pointer for the reverse edge `col -> row`."""
function reverse_sparse_ptr(rowvals::R, colptr::C, row::I, col::I) where {R<:AbstractVector,C<:AbstractVector,I<:Integer}
    @inbounds for rev_ptr in colptr[row]:(colptr[row + 1] - 1)
        rowvals[rev_ptr] == col && return rev_ptr
    end
    throw(ArgumentError("sparse adjacency is missing reverse edge for ($(row), $(col))"))
end

"""Force paired directed CSC values to exactly represent one symmetric Ising matrix."""
function symmetrize_sparse_parameter_values!(values::V, adjacency::A) where {V<:AbstractVector,A}
    rowvals = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    length(values) == length(SparseArrays.getnzval(adjacency)) ||
        throw(ArgumentError("parameter vector length does not match adjacency storage"))
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            row <= col && continue
            rev_ptr = reverse_sparse_ptr(rowvals, colptr, row, col)
            avg = (values[ptr] + values[rev_ptr]) / eltype(values)(2)
            values[ptr] = avg
            values[rev_ptr] = avg
        end
    end
    return values
end

"""Return the maximum absolute mismatch between paired symmetric sparse entries."""
function sparse_parameter_symmetry_error(values::V, adjacency::A) where {V<:AbstractVector,A}
    rowvals = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    err = zero(eltype(values))
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            row <= col && continue
            rev_ptr = reverse_sparse_ptr(rowvals, colptr, row, col)
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
        II.Clamping(
            β = II.UniformArray(zero(FT)),
            y = g -> II.filltype(Vector, zero(FT), II.statelen(g)),
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
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = dynamics,
        validation_algorithm = dynamics,
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

"""Return the object inside a `Ref`, or the object itself for direct values."""
@inline ref_value(x::Base.RefValue) = x[]
@inline ref_value(x) = x

# Apply a possibly reference-backed MNIST output target to one worker graph.
StatefulAlgorithms.@ProcessAlgorithm function ApplyTargetsRef!(isinggraph::G, y) where G
    IsingLearning.apply_targets(isinggraph, ref_value(y))
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

"""Return the direct quadratic output cost `1/2 * sum((s_output - y)^2)`."""
function quadratic_output_cost(
    y::Y,
    state::S,
    output_idxs::O,
) where {Y<:AbstractVector,S<:AbstractVector,O<:AbstractVector{Int}}
    cost = zero(FT)
    @inbounds for target_idx in eachindex(y)
        δ = FT(state[output_idxs[target_idx]]) - FT(y[target_idx])
        cost += δ * δ
    end
    return FT(0.5) * cost
end

"""Add Gaussian state noise and clamp continuous spins back into `[-1, 1]`."""
function inject_gaussian_state_noise!(
    isinggraph::G,
    rng::R,
    noise_scale::T,
) where {G,R<:Random.AbstractRNG,T<:Real}
    s = II.state(isinggraph)
    scale = eltype(s)(noise_scale)
    lower = -one(eltype(s))
    upper = one(eltype(s))
    @inbounds for idx in eachindex(s)
        s[idx] = clamp(s[idx] + scale * randn(rng, eltype(s)), lower, upper)
    end
    return isinggraph
end

"""Return a reward for one sampled attractor state and MNIST target."""
function attractor_reward(
    y::Y,
    state::S,
    output_idxs::O,
    output_replicas::I,
    reward_mode::M,
) where {Y<:AbstractVector,S<:AbstractVector,O<:AbstractVector{Int},I<:Integer,M<:AbstractString}
    replicas = Int(output_replicas)
    best_pred = 1
    best_truth = 1
    best_pred_score = typemin(FT)
    best_truth_score = typemin(FT)
    correct_score = zero(FT)
    max_other_score = typemin(FT)
    scores = ntuple(_ -> zero(FT), NCLASSES)

    @inbounds for digit in 1:NCLASSES
        pred_score = zero(FT)
        truth_score = zero(FT)
        first_idx = (digit - 1) * replicas + 1
        for replica_idx in first_idx:(first_idx + replicas - 1)
            pred_score += FT(state[output_idxs[replica_idx]])
            truth_score += FT(y[replica_idx])
        end
        pred_score /= FT(replicas)
        if pred_score > best_pred_score
            best_pred_score = pred_score
            best_pred = digit
        end
        if truth_score > best_truth_score
            best_truth_score = truth_score
            best_truth = digit
        end
        scores = Base.setindex(scores, pred_score, digit)
    end

    @inbounds for digit in 1:NCLASSES
        if digit == best_truth
            correct_score = scores[digit]
        else
            max_other_score = max(max_other_score, scores[digit])
        end
    end

    if reward_mode == "correct01"
        return best_pred == best_truth ? one(FT) : zero(FT)
    elseif reward_mode == "correctpm"
        return best_pred == best_truth ? one(FT) : -one(FT)
    elseif reward_mode == "margin"
        return correct_score - max_other_score
    elseif reward_mode == "logprob"
        max_score = maximum(scores)
        sum_exp = zero(FT)
        @inbounds for digit in 1:NCLASSES
            sum_exp += exp(scores[digit] - max_score)
        end
        return correct_score - (max_score + log(sum_exp))
    else
        throw(ArgumentError("unknown attractor reward mode: $(reward_mode)"))
    end
end

"""Return the one-based MNIST label encoded by a repeated `-1/+1` target vector."""
function target_label_index(y::Y, output_replicas::I) where {Y<:AbstractVector,I<:Integer}
    replicas = Int(output_replicas)
    best_label = 1
    best_score = typemin(FT)
    @inbounds for digit in 1:NCLASSES
        score = zero(FT)
        first_idx = (digit - 1) * replicas + 1
        for replica_idx in first_idx:(first_idx + replicas - 1)
            score += FT(y[replica_idx])
        end
        if score > best_score
            best_score = score
            best_label = digit
        end
    end
    return best_label
end

"""Accumulate one sampled attractor directly into reward-statistic sums."""
function accumulate_surrogate_attractor_sample!(
    isinggraph::G,
    y::Y,
    state::S,
    output_idxs::O,
    output_replicas::I,
    reward_mode::M,
    x::X,
    buffers::B,
) where {
    G,
    Y<:AbstractVector,
    S<:AbstractVector,
    O<:AbstractVector{Int},
    I<:Integer,
    M<:AbstractString,
    X<:AbstractVector,
    B,
}
    reward = attractor_reward(y, state, output_idxs, output_replicas, reward_mode)
    label_idx = target_label_index(y, output_replicas)
    buffers.reward_sum[] += reward
    buffers.reward_sumsq[] += reward * reward
    buffers.attractor_samples[] += 1
    buffers.reward_sum_by_label[label_idx] += reward
    buffers.reward_sumsq_by_label[label_idx] += reward * reward
    buffers.attractor_samples_by_label[label_idx] += 1

    adjacency = II.adj(isinggraph)
    nzvals = SparseArrays.getnzval(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    rowvals = SparseArrays.rowvals(adjacency)
    grad_w = buffers.w
    stat_w = buffers.stat_w
    stat_w_by_label = buffers.stat_w_by_label
    grad_b = buffers.b
    stat_b = buffers.stat_b
    stat_b_by_label = buffers.stat_b_by_label
    grad_w_input = vec(buffers.w_input)
    stat_w_input = vec(buffers.stat_w_input)
    stat_w_input_by_label = buffers.stat_w_input_by_label
    T = eltype(grad_w)
    length(nzvals) == length(grad_w) ||
        throw(ArgumentError("coupling gradient buffer does not match sparse graph storage"))

    # Walk the sparse CSC parameter storage directly; each ptr is one stored J.
    @inbounds for col in axes(adjacency, 2)
        state_col = T(state[col])
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            statistic = T(-0.5) * T(state[row]) * state_col
            grad_w[ptr] += reward * statistic
            stat_w[ptr] += statistic
            stat_w_by_label[ptr, label_idx] += statistic
        end
    end

    # The base field and image projection are both magnetic-field terms.
    @inbounds for state_idx in eachindex(state)
        statistic = -T(state[state_idx])
        grad_b[state_idx] += reward * statistic
        stat_b[state_idx] += statistic
        stat_b_by_label[state_idx, label_idx] += statistic
    end

    hidden_count = size(buffers.w_input, 1)
    @inbounds for input_idx in eachindex(x)
        xval = T(x[input_idx])
        for hidden_idx in 1:hidden_count
            flat_idx = hidden_idx + (input_idx - 1) * hidden_count
            statistic = -xval * T(state[hidden_idx])
            grad_w_input[flat_idx] += reward * statistic
            stat_w_input[flat_idx] += statistic
            stat_w_input_by_label[flat_idx, label_idx] += statistic
        end
    end

    return buffers
end

"""Finish one sample stream after online accumulation."""
function finish_surrogate_attractor_gradient!(buffers::B) where {B}
    return buffers
end

# Process wrapper that injects controlled state noise between stored samples.
StatefulAlgorithms.@ProcessAlgorithm function InjectGaussianStateNoiseRef!(
    isinggraph::G,
    rng::Random.AbstractRNG,
    noise_scale::Real,
) where {G}
    inject_gaussian_state_noise!(isinggraph, rng, noise_scale)
    return nothing
end

# Set a routed scalar target field to a routed scalar value.
StatefulAlgorithms.@ProcessAlgorithm function SetScalarTarget(target::T, value::T) where {T<:Real}
    # The `target` output name is intentionally the same as the routed input name.
    return (; target = T(value))
end

# Reset per-example attractor statistics before a sample stream.
StatefulAlgorithms.@ProcessAlgorithm function ResetSurrogateAttractorStats!(buffers::B) where {B}
    clear_covariance_stats!(buffers)
    return nothing
end

# Accumulate a surrogate attractor sample from reference-backed input and target state.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateSurrogateAttractorSampleRef!(
    isinggraph,
    y,
    sample_state::AbstractVector,
    output_idxs::AbstractVector{Int},
    output_replicas::Integer,
    reward_mode::AbstractString,
    x,
    buffers,
)
    accumulate_surrogate_attractor_sample!(
        isinggraph,
        ref_value(y),
        sample_state,
        output_idxs,
        output_replicas,
        reward_mode,
        ref_value(x),
        buffers,
    )
    return nothing
end

# Finalize one per-example surrogate estimate into the worker gradient buffer.
StatefulAlgorithms.@ProcessAlgorithm function FinishSurrogateAttractorGradient!(buffers::B) where {B}
    finish_surrogate_attractor_gradient!(buffers)
    return nothing
end

"""Build the free-phase input-field routine for one MNIST sample."""
function input_field_free_phase_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = layer.dynamics_algorithm
    free_steps = layer.free_relaxation_steps
    n_units = layer.nunits

    return StatefulAlgorithms.@Routine begin
        @state x
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state equilibrium_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Fold the image into the worker-local input field, then relax once.
        IsingLearning.initstate!(dynamics.model)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        model = @repeat free_steps dynamics()
        IsingLearning.CopyGraphState!(equilibrium_state, model)
    end
end

"""Build one surrogate-attractor transition measurement routine."""
function input_field_surrogate_measurement_algorithm(
    dynamics_algorithm::D,
    kick_steps::K,
    sample_interval_steps::I,
) where {D,K<:Integer,I<:Integer}
    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state buffers
        @state sample_state
        @state output_idxs
        @state output_replicas
        @state reward_mode
        @state noise_scale
        @state burnin_stepsize
        @state kick_stepsize
        @alias dynamics = dynamics_algorithm

        # Each read is a minimum reached after noise plus a relaxation window.
        InjectGaussianStateNoiseRef!(dynamics.model, dynamics.rng, noise_scale)
        SetScalarTarget(target = dynamics.stepsize, value = kick_stepsize)
        @repeat kick_steps dynamics()
        SetScalarTarget(target = dynamics.stepsize, value = burnin_stepsize)
        @repeat sample_interval_steps dynamics()
        IsingLearning.CopyGraphState!(sample_state, dynamics.model)
        AccumulateSurrogateAttractorSampleRef!(
            dynamics.model,
            y,
            sample_state,
            output_idxs,
            output_replicas,
            reward_mode,
            x,
            buffers,
        )
    end
end

"""Build the surrogate-attractor learning rule for one MNIST sample."""
function input_field_surrogate_attractor_algorithm(
    layer::L,
    config::C,
) where {L<:IsingLearning.LayeredIsingGraphLayer,C<:InputFieldMNISTConfig}
    dynamics_algorithm = layer.dynamics_algorithm
    free_steps = layer.free_relaxation_steps
    n_units = layer.nunits
    sample_count = max(1, Int(config.covariance_samples))
    sample_interval_steps = max(1, round(Int, config.covariance_sample_sweeps * n_units))
    kick_steps = max(1, Int(config.covariance_kick_steps))
    burnin_stepsize = FT(config.stepsize)
    kick_stepsize = FT(config.covariance_stepsize)
    noise_scale = FT(config.covariance_noise_temp_factor * sqrt(config.temp))
    output_idxs = collect(Int, layer.output_layer)
    output_replicas = length(layer.output_layer) ÷ NCLASSES
    reward_mode = String(config.reward_mode)
    measurement = input_field_surrogate_measurement_algorithm(
        dynamics_algorithm,
        kick_steps,
        sample_interval_steps,
    )

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state sample_state = zeros(FT, n_units)
        @state output_idxs = output_idxs
        @state output_replicas = output_replicas
        @state reward_mode = reward_mode
        @state burnin_stepsize = burnin_stepsize
        @state kick_stepsize = kick_stepsize
        @state noise_scale = noise_scale
        @alias dynamics = dynamics_algorithm
        @alias measurement = measurement

        # Targets enter only through the reward; dynamics remains beta-zero.
        ResetSurrogateAttractorStats!(buffers)
        IsingLearning.initstate!(dynamics.model)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        SetScalarTarget(target = dynamics.stepsize, value = burnin_stepsize)
        @repeat free_steps dynamics()
        @context measurement_context = @repeat sample_count measurement()
        @bind x => measurement_context.x
        @bind y => measurement_context.y
        @bind buffers => measurement_context.buffers
        @bind sample_state => measurement_context.sample_state
        @bind output_idxs => measurement_context.output_idxs
        @bind output_replicas => measurement_context.output_replicas
        @bind reward_mode => measurement_context.reward_mode
        @bind noise_scale => measurement_context.noise_scale
        @bind burnin_stepsize => measurement_context.burnin_stepsize
        @bind kick_stepsize => measurement_context.kick_stepsize
        FinishSurrogateAttractorGradient!(buffers)
    end
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
    sample_count::I,
) where {A,L<:IsingLearning.LayeredIsingGraphLayer,G,R<:Base.RefValue,I<:Integer}
    state = II.state(graph)
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:_state;
            x = Ref(zeros(eltype(graph), INPUT_DIM)),
            y = Ref(zeros(eltype(graph), length(layer.output_layer))),
            input_hidden_w = input_hidden_w,
            buffers = input_field_surrogate_worker_buffer(graph, input_hidden_w[], sample_count),
            equilibrium_state = copy(state),
            nudged_state = similar(state),
        ),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""Build the reusable validation algorithm used by manager-owned validation workers."""
function input_field_validation_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = layer.validation_algorithm
    relaxation_steps = layer.free_relaxation_steps
    n_units = layer.nunits
    replica_count = length(layer.output_layer) ÷ NCLASSES

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state equilibrium_state = zeros(n_units)
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
    clear_buffer!(manager.state.batch_stat)
    clear_diagnostics!(manager.state.diagnostics)
    manager.state.nsamples[] = 0
    for worker in StatefulAlgorithms.workers(manager)
        clear_buffer!(worker_context(worker).buffers)
    end
    return manager
end

"""Reset scalar diagnostics retained from the previous minibatch."""
function clear_diagnostics!(diagnostics::D) where {D}
    diagnostics.attractor_samples[] = 0
    diagnostics.reward_mean[] = zero(FT)
    diagnostics.reward_std[] = zero(FT)
    return diagnostics
end

"""Return a nonnegative standard deviation from raw sums."""
function mean_std(sum_x::T, sum_x2::T, n::I) where {T<:Real,I<:Integer}
    n_int = Int(n)
    n_int > 0 || return zero(FT), zero(FT)
    mean_x = FT(sum_x) / FT(n_int)
    var_x = max(zero(FT), FT(sum_x2) / FT(n_int) - mean_x * mean_x)
    return mean_x, sqrt(var_x)
end

"""Subtract the selected reward baseline from an accumulated feature covariance."""
function apply_reward_baseline!(manager::M, batch_reward_mean::T) where {M<:StatefulAlgorithms.ProcessManager,T<:Real}
    mode = String(manager.config.reward_baseline)
    gradient = manager.state.batch_gradient
    stat = manager.state.batch_stat
    if mode == "none"
        return gradient
    elseif mode == "batch"
        gradient.w .-= FT(batch_reward_mean) .* stat.w
        gradient.b .-= FT(batch_reward_mean) .* stat.b
        gradient.w_input .-= FT(batch_reward_mean) .* stat.w_input
    elseif mode == "label"
        @inbounds for label_idx in 1:NCLASSES
            count = stat.attractor_samples_by_label[label_idx]
            count == 0 && continue
            label_mean = stat.reward_sum_by_label[label_idx] / FT(count)
            gradient.w .-= label_mean .* view(stat.w_by_label, :, label_idx)
            gradient.b .-= label_mean .* view(stat.b_by_label, :, label_idx)
            vec(gradient.w_input) .-= label_mean .* view(stat.w_input_by_label, :, label_idx)
        end
    else
        throw(ArgumentError("unknown reward baseline mode: $(mode)"))
    end
    return gradient
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

"""Flush worker gradients into one Adam-ready minibatch gradient."""
function flush_manager_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    clear_buffer!(manager.state.batch_stat)
    clear_diagnostics!(manager.state.diagnostics)
    reward_sum = zero(FT)
    reward_sumsq = zero(FT)
    attractor_samples = 0
    for worker in StatefulAlgorithms.workers(manager)
        buffers = worker_context(worker).buffers
        add_buffer!(manager.state.batch_gradient, buffers)
        add_stat_buffer!(manager.state.batch_stat, buffers)
        reward_sum += FT(buffers.reward_sum[])
        reward_sumsq += FT(buffers.reward_sumsq[])
        attractor_samples += buffers.attractor_samples[]
        clear_buffer!(buffers)
    end

    nsamples = manager.state.nsamples[]
    nsamples > 0 || throw(ArgumentError("cannot flush an empty MNIST minibatch"))
    attractor_samples > 0 || throw(ArgumentError("cannot flush a minibatch without attractor samples"))
    batch_reward_mean = reward_sum / FT(attractor_samples)
    apply_reward_baseline!(manager, batch_reward_mean)
    scale_buffer!(manager.state.batch_gradient, FT(manager.config.gradient_sign) / FT(attractor_samples))
    manager.config.project_output_bias_prior &&
        project_output_bias_prior!(manager.state.batch_gradient, manager.state.layer, manager.config.output_replicas)

    reward_mean, reward_std = mean_std(reward_sum, reward_sumsq, attractor_samples)
    diagnostics = manager.state.diagnostics
    diagnostics.attractor_samples[] = attractor_samples
    diagnostics.reward_mean[] = reward_mean
    diagnostics.reward_std[] = reward_std
    if manager.config.weight_decay > zero(FT)
        manager.state.batch_gradient.w .+= manager.config.weight_decay .* manager.state.params[].w
        manager.state.batch_gradient.w_input .+= manager.config.weight_decay .* manager.state.params[].w_input
    end
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
    params = input_field_params(source, input_hidden_w[])
    symmetrize_sparse_parameter_values!(params.w, II.adj(source))
    normalize_sparse_couplings!(params.w, config)
    normalize_input_projection!(params.w_input, config)
    input_hidden_w[] = params.w_input
    optimiser = Optimisers.Adam(config.lr)
    algorithm = StatefulAlgorithms.resolve(input_field_surrogate_attractor_algorithm(layer, config))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        input_field_gradient_buffer(source, input_hidden_w[]),
        input_field_stat_buffer(source, input_hidden_w[]),
        input_field_diagnostics_buffer(FT),
        Ref(0),
        Optimisers.setup(optimiser, params),
        Ref(zeros(FT, INPUT_DIM, 0)),
        Ref(zeros(FT, NCLASSES * config.output_replicas, 0)),
        input_hidden_w,
    )
    recipe = (;
        makeworker = (idx, manager) -> input_field_worker(
            algorithm,
            manager.state.layer,
            shared_worker_graph(manager.state.source_graph),
            manager.state.input_hidden_w,
            manager.config.covariance_samples,
        ),
        loadjob! = load_input_field_training_job!,
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
        Ref(zeros(FT, INPUT_DIM, 0)),
        Ref(zeros(FT, NCLASSES * config.output_replicas, 0)),
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

"""Split a minibatch index view into concrete manager jobs."""
function batch_jobs(x::X, y::Y, indices::V) where {X<:AbstractMatrix,Y<:AbstractMatrix,V<:AbstractVector{Int}}
    jobs = InputFieldMNISTJob{Vector{FT},Vector{FT}}[]
    for sample_idx in indices
        push!(jobs, InputFieldMNISTJob(copy(view(x, :, sample_idx)), copy(view(y, :, sample_idx))))
    end
    return jobs
end

"""Run one minibatch through the manager and apply one Adam update."""
function run_minibatch!(manager::M, jobs::J) where {M<:StatefulAlgorithms.ProcessManager,J<:AbstractVector}
    clear_manager_buffers!(manager)
    manager.state.nsamples[] = length(jobs)
    StatefulAlgorithms.run!(manager, jobs)
    symmetrize_sparse_parameter_values!(manager.state.batch_gradient.w, II.adj(manager.state.source_graph))
    manager.state.opt_state, params = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
    symmetrize_sparse_parameter_values!(params.w, II.adj(manager.state.source_graph))
    normalize_sparse_couplings!(params.w, manager.config)
    normalize_input_projection!(params.w_input, manager.config)
    manager.state.params[] = params
    sync_after_update!(manager, params)
    return params
end

"""Evaluate accuracy and squared error using preallocated validation-manager jobs."""
function evaluate(
    manager::M,
    x::X,
    y::Y,
    config::C,
    jobs::B,
) where {M<:StatefulAlgorithms.ProcessManager,X<:AbstractMatrix,Y<:AbstractMatrix,C<:InputFieldMNISTConfig,B<:InputFieldMNISTJobBuffer}
    clear_validation_buffers!(manager)
    active_jobs = fill_jobs!(jobs, x, y, collect(axes(x, 2)))
    StatefulAlgorithms.run!(manager, active_jobs)
    nsamples = manager.state.nsamples[]
    return (;
        accuracy = nsamples == 0 ? 0.0 : manager.state.ncorrect[] / nsamples,
        loss = nsamples == 0 ? zero(FT) : manager.state.total_loss[] / nsamples,
        pred_counts = copy(manager.state.pred_counts),
    )
end

"""Evaluate accuracy and print a timed status line for long validation passes."""
function evaluate_with_progress(
    manager::M,
    x::X,
    y::Y,
    config::C,
    jobs::B,
    label::L,
) where {
    M<:StatefulAlgorithms.ProcessManager,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
    C<:InputFieldMNISTConfig,
    B<:InputFieldMNISTJobBuffer,
    L<:AbstractString,
}
    nsamples = size(x, 2)
    println("$(label): evaluating $(nsamples) samples")
    flush(stdout)
    seconds = @elapsed result = evaluate(manager, x, y, config, jobs)
    println("$(label): done in $(round(seconds; digits = 3))s accuracy=$(round(result.accuracy; digits = 5)) loss=$(result.loss)")
    flush(stdout)
    return result
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
            config,
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

"""Return the sidecar path used for legacy checkpoint recovery."""
function recovered_checkpoint_path(path::P) where {P<:AbstractString}
    stem, ext = splitext(path)
    return stem * "_recovered" * ext
end

"""Recover one pre-resume baseline checkpoint into a neutral serialized payload."""
function recover_legacy_checkpoint!(recovered_path::P, path::Q) where {P<:AbstractString,Q<:AbstractString}
    project_path = abspath(joinpath(@__DIR__, "..", "..", ".."))
    julia_path = joinpath(Sys.BINDIR, Base.julia_exename())
    source = """
    using Optimisers
    using Serialization

    Base.@kwdef struct InputFieldMNISTConfig{T<:AbstractFloat,S<:AbstractString}
        workers::Int = 0
        epochs::Int = 0
        batchsize::Int = 0
        scheduler::S = ""
        chunk_size::Int = 0
        train_per_class::Int = 0
        test_per_class::Int = 0
        train_eval_per_class::Int = 0
        eval_every::Int = 0
        progress_every_batches::Int = 25
        early_stop_decline_epochs::Int = 0
        hidden::Int = 0
        output_replicas::Int = 0
        sweeps::T = zero(T)
        β::T = zero(T)
        covariance_samples::Int = 0
        covariance_sample_sweeps::T = zero(T)
        covariance_kick_steps::Int = 0
        covariance_stepsize::T = zero(T)
        covariance_noise_temp_factor::T = zero(T)
        reward_mode::S = "logprob"
        reward_baseline::S = "batch"
        lr::T = zero(T)
        gradient_sign::T = one(T)
        w_normalization::S = "none"
        w_norm::T = one(T)
        project_output_bias_prior::Bool = false
        w_input_normalization::S = "none"
        w_input_row_norm::T = T(0.14)
        temp::T = zero(T)
        stepsize::T = zero(T)
        weight_scale::T = zero(T)
        weight_decay::T = zero(T)
        seed::Int = 0
        outdir::S = ""
    end

    checkpoint = deserialize(IOBuffer(read($(repr(path)))))
    config = (
        workers = checkpoint.config.workers,
        epochs = checkpoint.config.epochs,
        batchsize = checkpoint.config.batchsize,
        scheduler = hasproperty(checkpoint.config, :scheduler) ? String(checkpoint.config.scheduler) : "spawn",
        chunk_size = hasproperty(checkpoint.config, :chunk_size) ? checkpoint.config.chunk_size : 0,
        train_per_class = checkpoint.config.train_per_class,
        test_per_class = checkpoint.config.test_per_class,
        train_eval_per_class = checkpoint.config.train_eval_per_class,
        eval_every = checkpoint.config.eval_every,
        progress_every_batches = hasproperty(checkpoint.config, :progress_every_batches) ? checkpoint.config.progress_every_batches : 25,
        early_stop_decline_epochs = hasproperty(checkpoint.config, :early_stop_decline_epochs) ? checkpoint.config.early_stop_decline_epochs : 0,
        hidden = checkpoint.config.hidden,
        output_replicas = checkpoint.config.output_replicas,
        sweeps = checkpoint.config.sweeps,
        β = checkpoint.config.β,
        covariance_samples = hasproperty(checkpoint.config, :covariance_samples) ? checkpoint.config.covariance_samples : 1,
        covariance_sample_sweeps = hasproperty(checkpoint.config, :covariance_sample_sweeps) ? checkpoint.config.covariance_sample_sweeps : one(Float32),
        covariance_kick_steps = hasproperty(checkpoint.config, :covariance_kick_steps) ? checkpoint.config.covariance_kick_steps : 5,
        covariance_stepsize = hasproperty(checkpoint.config, :covariance_stepsize) ? checkpoint.config.covariance_stepsize : checkpoint.config.stepsize,
        covariance_noise_temp_factor = hasproperty(checkpoint.config, :covariance_noise_temp_factor) ? checkpoint.config.covariance_noise_temp_factor : one(Float32),
        reward_mode = hasproperty(checkpoint.config, :reward_mode) ? String(checkpoint.config.reward_mode) : "logprob",
        reward_baseline = hasproperty(checkpoint.config, :reward_baseline) ? String(checkpoint.config.reward_baseline) : "batch",
        lr = checkpoint.config.lr,
        gradient_sign = hasproperty(checkpoint.config, :gradient_sign) ? checkpoint.config.gradient_sign : one(Float32),
        w_normalization = hasproperty(checkpoint.config, :w_normalization) ? String(checkpoint.config.w_normalization) : "none",
        w_norm = hasproperty(checkpoint.config, :w_norm) ? checkpoint.config.w_norm : one(Float32),
        project_output_bias_prior = hasproperty(checkpoint.config, :project_output_bias_prior) ? checkpoint.config.project_output_bias_prior : false,
        w_input_normalization = hasproperty(checkpoint.config, :w_input_normalization) ? String(checkpoint.config.w_input_normalization) : "none",
        w_input_row_norm = hasproperty(checkpoint.config, :w_input_row_norm) ? checkpoint.config.w_input_row_norm : Float32(0.14),
        temp = checkpoint.config.temp,
        stepsize = checkpoint.config.stepsize,
        weight_scale = checkpoint.config.weight_scale,
        weight_decay = checkpoint.config.weight_decay,
        seed = checkpoint.config.seed,
        outdir = String(checkpoint.config.outdir),
    )

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
    symmetrize_sparse_parameter_values!(manager.state.params[].w, II.adj(manager.state.source_graph))
    normalize_sparse_couplings!(manager.state.params[].w, manager.config)
    manager.state.opt_state = checkpoint.opt_state
    sync_after_update!(manager, manager.state.params[])
    return manager
end

"""Return the default file paths used by one retained run directory."""
function session_paths(outdir::P) where {P<:AbstractString}
    return (
        csv_path = joinpath(outdir, "mnist_784_120_40_surrogate_attractor_adam.csv"),
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
    ax_acc = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "MNIST 784-120-40 surrogate-attractor accuracy")
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

"""Write the run settings needed to reproduce the surrogate-attractor run."""
function write_settings!(path::P, config::C, relaxation_steps::I) where {P<:AbstractString,C<:InputFieldMNISTConfig,I<:Integer}
    open(path, "w") do io
        println(io, "# MNIST 784-120-40 REINFORCE Minima-Fixes Adam")
        println(io)
        println(io, "- learning note: local `derivation.md`")
        println(io, "- architecture: `784 -> $(config.hidden) -> $(NCLASSES * config.output_replicas)`")
        println(io, "- sampled graph: hidden/output only, with no structural input layer")
        println(io, "- input handling: MNIST pixels in `[0, 1]` are projected through external `784 -> hidden` weights into a worker-local field")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- manager execution: `ChannelWorkers()` with one sample per manager job")
        println(io, "- progress print interval: `$(config.progress_every_batches)` batches")
        println(io, "- early stop decline epochs: `$(config.early_stop_decline_epochs)`")
        println(io, "- worker graph adjacency: pointer-shared with source graph")
        println(io, "- worker graph base bias: pointer-shared with source graph")
        println(io, "- learning step: configurable REINFORCE-style reward covariance over sampled minima")
        println(io, "- sparse coupling constraint: paired CSC entries are symmetrized before and after Adam updates")
        println(io, "- validation: `ChannelWorkers()` ProcessManager with worker-local stats")
        println(io, "- job buffers: preallocated per-sample jobs reused across minibatches/evaluations")
        println(io, "- optimiser: `Optimisers.Adam($(config.lr))`")
        println(io, "- epochs/batchsize: `$(config.epochs)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- train eval per class: `$(config.train_eval_per_class)`")
        println(io, "- sweeps/relaxation steps: `$(config.sweeps)` / `$(relaxation_steps)`")
        println(io, "- covariance samples/sample sweeps: `$(config.covariance_samples)` / `$(config.covariance_sample_sweeps)`")
        println(io, "- covariance kick steps/noise temp factor/stepsize: `$(config.covariance_kick_steps)` / `$(config.covariance_noise_temp_factor)` / `$(config.covariance_stepsize)`")
        println(io, "- reward mode: `$(config.reward_mode)`")
        println(io, "- reward baseline: `$(config.reward_baseline)`")
        println(io, "- gradient sign: `$(config.gradient_sign)`")
        println(io, "- recurrent w normalization/norm: `$(config.w_normalization)` / `$(config.w_norm)`")
        println(io, "- project output bias prior: `$(config.project_output_bias_prior)`")
        println(io, "- w_input normalization/row norm: `$(config.w_input_normalization)` / `$(config.w_input_row_norm)`")
        println(io, "- beta/temp/stepsize: `$(config.β)` unused for training / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- weight scale/decay: `$(config.weight_scale)` / `$(config.weight_decay)`")
        println(io, "- resume from: `$(isempty(config.resume_from) ? "none" : config.resume_from)`")
        println(io, "- resume epoch: `$(config.resume_epoch)`")
    end
    return path
end

"""Validate surrogate-attractor configuration before building long-lived runtime state."""
function validate_config!(config::C) where {C<:InputFieldMNISTConfig}
    config.workers > 0 || throw(ArgumentError("ISING_MNIST_IF_WORKERS must be positive"))
    config.batchsize > 0 || throw(ArgumentError("ISING_MNIST_IF_BATCHSIZE must be positive"))
    config.epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EPOCHS must be nonnegative"))
    config.progress_every_batches >= 0 || throw(ArgumentError("ISING_MNIST_IF_PROGRESS_EVERY_BATCHES must be nonnegative"))
    config.early_stop_decline_epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS must be nonnegative"))
    config.covariance_samples > 0 || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_SAMPLES must be positive"))
    config.covariance_sample_sweeps > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_SAMPLE_SWEEPS must be positive"))
    config.covariance_kick_steps > 0 || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_KICK_STEPS must be positive"))
    config.covariance_stepsize > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_STEPSIZE must be positive"))
    config.covariance_noise_temp_factor >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_NOISE_TEMP_FACTOR must be nonnegative"))
    String(config.reward_mode) in ("correct01", "correctpm", "margin", "logprob") ||
        throw(ArgumentError("ISING_MNIST_IF_ATTRACTOR_REWARD must be one of correct01, correctpm, margin, logprob"))
    String(config.reward_baseline) in ("none", "batch", "label") ||
        throw(ArgumentError("ISING_MNIST_IF_REWARD_BASELINE must be one of none, batch, label"))
    String(config.w_normalization) in ("none", "global") ||
        throw(ArgumentError("ISING_MNIST_IF_W_NORMALIZATION must be one of none, global"))
    config.w_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_W_NORM must be positive"))
    String(config.w_input_normalization) in ("none", "row", "global") ||
        throw(ArgumentError("ISING_MNIST_IF_W_INPUT_NORMALIZATION must be one of none, row, global"))
    config.w_input_row_norm > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_W_INPUT_ROW_NORM must be positive"))
    config.temp > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_TEMP must be positive for Langevin noise"))
    config.hidden == 120 || @warn "baseline paper hidden count is 120" hidden = config.hidden
    config.output_replicas == 4 || @warn "baseline paper output count is 40, i.e. four replicas per digit" output_replicas = config.output_replicas
    config.train_per_class < 5421 && @warn "this run uses a subsampled balanced training split, so it is not the full balanced MNIST baseline" train_per_class = config.train_per_class
    config.test_per_class < 892 && @warn "this run uses a subsampled balanced test split, so reported accuracy will be noisy" test_per_class = config.test_per_class
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    has_resume_checkpoint(config) && !isfile(config.resume_from) &&
        throw(ArgumentError("ISING_MNIST_IF_RESUME_FROM does not point to a checkpoint file"))
    return config
end

"""Construct one reusable surrogate-attractor session that can be rerun or checkpoint-restarted in-process."""
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
    println("constructing ", config.workers, "-worker surrogate-attractor Adam manager")
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

"""Run surrogate-attractor epochs on an already prepared session."""
function run_epochs!(session::S; start_epoch::Int = 0, stop_epoch::Int = session.config.epochs) where {S<:InputFieldMNISTSession}
    config = session.config
    prev_eval_accuracy = Ref{Union{Nothing,Float64}}(nothing)
    decline_streak = Ref(0)
    for epoch in start_epoch:stop_epoch
        seconds = 0.0
        train_accuracy = missing
        train_loss = missing
        test_accuracy = missing
        test_loss = missing
        pred_counts = missing
        is_new_best = false
        should_stop_early = false
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
                    diagnostics = session.manager.state.diagnostics
                    if progress !== nothing
                        ProgressMeter.next!(
                            progress;
                            showvalues = [
                                (:batch, "$(batch_idx)/$(nbatches)"),
                                (:samples, "$(last_idx)/$(length(order))"),
                                (:reward_mean, round(Float64(diagnostics.reward_mean[]); digits = 5)),
                                (:reward_std, round(Float64(diagnostics.reward_std[]); digits = 5)),
                                (:minima, diagnostics.attractor_samples[]),
                            ],
                        )
                    end
                    if progress_every > 0 && (batch_idx == 1 || batch_idx == nbatches || mod(batch_idx, progress_every) == 0)
                        elapsed = time() - epoch_start
                        println(
                            "epoch ", epoch, " batch ", batch_idx, "/", nbatches,
                            " samples ", last_idx, "/", length(order),
                            " elapsed ", round(elapsed; digits = 1), "s",
                            " reward_mean ", round(Float64(diagnostics.reward_mean[]); digits = 5),
                            " reward_std ", round(Float64(diagnostics.reward_std[]); digits = 5),
                            " minima ", diagnostics.attractor_samples[],
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
        diagnostics = session.manager.state.diagnostics
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
            norms.symmetry_error,
            norms.grad_symmetry_error,
            attractor_samples = diagnostics.attractor_samples[],
            reward_mean = diagnostics.reward_mean[],
            reward_std = diagnostics.reward_std[],
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

"""Persist final artifacts for a completed surrogate-attractor session."""
function finalize_session!(session::S) where {S<:InputFieldMNISTSession}
    save_checkpoint(session.final_path, session.manager, session.config, session.rows)
    plot_metrics(joinpath(session.config.outdir, "learning_summary.png"), session.rows)
    println("saved MNIST 784-120-40 surrogate-attractor run in ", session.config.outdir)
    return (; config = session.config, rows = session.rows, best_accuracy = session.best_accuracy[])
end

"""Close manager-backed state held by one live surrogate-attractor session."""
function close_session!(session::S) where {S<:InputFieldMNISTSession}
    close(session.manager)
    close(session.validator)
    return nothing
end

"""Run the full surrogate-attractor experiment."""
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
    return run_config!(InputFieldMNISTConfig())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
