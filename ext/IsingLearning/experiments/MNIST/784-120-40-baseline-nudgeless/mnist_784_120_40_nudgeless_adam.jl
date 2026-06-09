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
    hidden::Int = parse(Int, get(ENV, "ISING_MNIST_IF_HIDDEN", "120"))
    output_replicas::Int = parse(Int, get(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS", "4"))
    sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_SWEEPS", "500"))
    β::T = parse(FT, get(ENV, "ISING_MNIST_IF_BETA", "5.0"))
    covariance_samples::Int = parse(Int, get(ENV, "ISING_MNIST_IF_COVARIANCE_SAMPLES", "20"))
    covariance_sample_sweeps::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_SAMPLE_SWEEPS", "1.0"))
    covariance_kick_steps::Int = parse(Int, get(ENV, "ISING_MNIST_IF_COVARIANCE_KICK_STEPS", "5"))
    covariance_stepsize::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_STEPSIZE", "1.0"))
    covariance_noise_temp_factor::T = parse(FT, get(ENV, "ISING_MNIST_IF_COVARIANCE_NOISE_TEMP_FACTOR", "1.0"))
    lr::T = parse(FT, get(ENV, "ISING_MNIST_IF_LR", "0.003"))
    gradient_sign::T = parse(FT, get(ENV, "ISING_MNIST_IF_GRADIENT_SIGN", "1.0"))
    gradient_temp_power::T = parse(FT, get(ENV, "ISING_MNIST_IF_GRADIENT_TEMP_POWER", "1.0"))
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
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("mnist_784_120_40_nudgeless_adam")),
    )
end

struct InputFieldMNISTJob{X<:AbstractVector,Y<:AbstractVector}
    x::X
    y::Y
end

struct InputFieldMNISTJobBuffer{J<:AbstractVector}
    jobs::J
end

mutable struct InputFieldMNISTManagerState{L,G,P,B,O,X,Y,W}
    layer::L
    source_graph::G
    params::Base.RefValue{P}
    batch_gradient::B
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
    hasproperty(buffer, :α) && fill!(buffer.α, zero(eltype(buffer.α)))
    if hasproperty(buffer, :sample_w)
        fill!(buffer.sample_w, zero(eltype(buffer.sample_w)))
        fill!(buffer.sample_b, zero(eltype(buffer.sample_b)))
        fill!(buffer.sample_w_input, zero(eltype(buffer.sample_w_input)))
        fill!(buffer.cost_samples, zero(eltype(buffer.cost_samples)))
        buffer.covariance_samples[] = 0
    end
    return buffer
end

"""Clear only the per-example covariance accumulators inside a worker buffer."""
function clear_covariance_stats!(buffer::B) where {B}
    fill!(buffer.sample_w, zero(eltype(buffer.sample_w)))
    fill!(buffer.sample_b, zero(eltype(buffer.sample_b)))
    fill!(buffer.sample_w_input, zero(eltype(buffer.sample_w_input)))
    fill!(buffer.cost_samples, zero(eltype(buffer.cost_samples)))
    buffer.covariance_samples[] = 0
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
    return (;
        w_norm = parameter_norm(params.w),
        b_norm = parameter_norm(params.b),
        w_input_norm,
        grad_w_norm = parameter_norm(grad.w),
        grad_b_norm = parameter_norm(grad.b),
        grad_w_input_norm,
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

"""Create one worker buffer with sample-history fields for beta-zero learning."""
function input_field_nudgeless_worker_buffer(
    graph::G,
    input_hidden_w::W,
    sample_capacity::I,
) where {G,W<:AbstractMatrix,I<:Integer}
    w_shape = length(SparseArrays.getnzval(II.adj(graph)))
    b_shape = II.nstates(graph)
    w_input_shape = length(input_hidden_w)
    sample_count = Int(sample_capacity)
    return (;
        w = zeros(eltype(graph), w_shape),
        b = zeros(eltype(graph), b_shape),
        w_input = zeros(eltype(input_hidden_w), size(input_hidden_w)),
        sample_w = zeros(eltype(graph), w_shape, sample_count),
        sample_b = zeros(eltype(graph), b_shape, sample_count),
        sample_w_input = zeros(eltype(input_hidden_w), w_input_shape, sample_count),
        cost_samples = zeros(eltype(graph), sample_count),
        covariance_samples = Ref(0),
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

"""Accumulate one stored sample column into a destination gradient by covariance."""
function add_sample_covariance_gradient!(
    dest::D,
    samples::S,
    costs::C,
    nsamples::I,
) where {D<:AbstractArray,S<:AbstractMatrix,C<:AbstractVector,I<:Integer}
    n = Int(nsamples)
    inv_samples = inv(FT(n))
    mean_cost = zero(FT)
    @inbounds for sample_idx in 1:n
        mean_cost += FT(costs[sample_idx])
    end
    mean_cost *= inv_samples

    @inbounds for param_idx in axes(samples, 1)
        mean_stat = zero(FT)
        mean_cost_stat = zero(FT)
        for sample_idx in 1:n
            statistic = FT(samples[param_idx, sample_idx])
            mean_stat += statistic
            mean_cost_stat += FT(costs[sample_idx]) * statistic
        end
        mean_stat *= inv_samples
        mean_cost_stat *= inv_samples
        dest[param_idx] += -(mean_cost_stat - mean_cost * mean_stat)
    end
    return dest
end

"""Accumulate one beta-zero sample into per-example covariance sums."""
function accumulate_nudgeless_covariance_sample!(
    isinggraph::G,
    y::Y,
    state::S,
    output_idxs::O,
    x::X,
    buffers::B,
) where {G,Y<:AbstractVector,S<:AbstractVector,O<:AbstractVector{Int},X<:AbstractVector,B}
    sample_idx = buffers.covariance_samples[] + 1
    sample_idx <= length(buffers.cost_samples) ||
        throw(ArgumentError("nudgeless covariance sample buffer is full"))
    cost = quadratic_output_cost(y, state, output_idxs)
    buffers.cost_samples[sample_idx] = cost
    adjacency = II.adj(isinggraph)
    colptr = SparseArrays.getcolptr(adjacency)
    rowvals = SparseArrays.rowvals(adjacency)

    # Store dE/dtheta; for Bilinear, dE/dJ_ij = -s_i*s_j/2.
    @inbounds for col in axes(adjacency, 2)
        for ptr in colptr[col]:(colptr[col + 1] - 1)
            row = rowvals[ptr]
            buffers.sample_w[ptr, sample_idx] = FT(-0.5) * FT(state[row]) * FT(state[col])
        end
    end

    # The base field and image projection are both magnetic-field terms.
    @inbounds for state_idx in eachindex(state)
        buffers.sample_b[state_idx, sample_idx] = -FT(state[state_idx])
    end

    hidden_count = size(buffers.w_input, 1)
    @inbounds for input_idx in eachindex(x)
        xval = FT(x[input_idx])
        for hidden_idx in 1:hidden_count
            flat_idx = hidden_idx + (input_idx - 1) * hidden_count
            buffers.sample_w_input[flat_idx, sample_idx] = -xval * FT(state[hidden_idx])
        end
    end

    buffers.covariance_samples[] = sample_idx
    return buffers
end

"""Convert per-example covariance sums into an Adam-ready loss-gradient contribution."""
function finish_nudgeless_covariance_gradient!(buffers::B) where {B}
    nsamples = buffers.covariance_samples[]
    nsamples > 0 || throw(ArgumentError("cannot finish nudgeless covariance gradient without samples"))
    # dL/dtheta = -cov(C, dE/dtheta) / T; division by T happens at minibatch flush.
    add_sample_covariance_gradient!(buffers.w, buffers.sample_w, buffers.cost_samples, nsamples)
    add_sample_covariance_gradient!(buffers.b, buffers.sample_b, buffers.cost_samples, nsamples)
    add_sample_covariance_gradient!(vec(buffers.w_input), buffers.sample_w_input, buffers.cost_samples, nsamples)
    clear_covariance_stats!(buffers)
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

# Reset per-example covariance statistics before a beta-zero sample stream.
StatefulAlgorithms.@ProcessAlgorithm function ResetNudgelessCovarianceStats!(buffers::B) where {B}
    clear_covariance_stats!(buffers)
    return nothing
end

# Accumulate a beta-zero sample from reference-backed input and target state.
StatefulAlgorithms.@ProcessAlgorithm function AccumulateNudgelessCovarianceSampleRef!(
    isinggraph::G,
    y,
    sample_state::AbstractVector,
    output_idxs::AbstractVector{Int},
    x,
    buffers,
) where {G}
    accumulate_nudgeless_covariance_sample!(
        isinggraph,
        ref_value(y),
        sample_state,
        output_idxs,
        ref_value(x),
        buffers,
    )
    return nothing
end

# Finalize one per-example covariance estimate into the worker gradient buffer.
StatefulAlgorithms.@ProcessAlgorithm function FinishNudgelessCovarianceGradient!(buffers::B) where {B}
    finish_nudgeless_covariance_gradient!(buffers)
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

"""Build one beta-zero covariance measurement routine."""
function input_field_nudgeless_measurement_algorithm(
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
        @state noise_scale
        @state burnin_stepsize
        @state kick_stepsize
        @alias dynamics = dynamics_algorithm

        # Record after normal relaxation, then kick and re-relax before the next read.
        IsingLearning.CopyGraphState!(sample_state, dynamics.model)
        AccumulateNudgelessCovarianceSampleRef!(dynamics.model, y, sample_state, output_idxs, x, buffers)
        InjectGaussianStateNoiseRef!(dynamics.model, dynamics.rng, noise_scale)
        SetScalarTarget(target = dynamics.stepsize, value = kick_stepsize)
        @repeat kick_steps dynamics()
        SetScalarTarget(target = dynamics.stepsize, value = burnin_stepsize)
        @repeat sample_interval_steps dynamics()
    end
end

"""Build the beta-zero covariance learning rule for one MNIST sample."""
function input_field_nudgeless_algorithm(
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
    measurement = input_field_nudgeless_measurement_algorithm(dynamics_algorithm, kick_steps, sample_interval_steps)

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state sample_state = zeros(FT, n_units)
        @state output_idxs = output_idxs
        @state burnin_stepsize = burnin_stepsize
        @state kick_stepsize = kick_stepsize
        @state noise_scale = noise_scale
        @alias dynamics = dynamics_algorithm
        @alias measurement = measurement

        # Sample only from beta-zero dynamics; targets enter only through C(s,y).
        ResetNudgelessCovarianceStats!(buffers)
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
        @bind noise_scale => measurement_context.noise_scale
        @bind burnin_stepsize => measurement_context.burnin_stepsize
        @bind kick_stepsize => measurement_context.kick_stepsize
        FinishNudgelessCovarianceGradient!(buffers)
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
            buffers = input_field_nudgeless_worker_buffer(graph, input_hidden_w[], sample_count),
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
    manager.state.nsamples[] = 0
    for worker in StatefulAlgorithms.workers(manager)
        clear_buffer!(worker_context(worker).buffers)
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

"""Flush worker gradients into one Adam-ready minibatch gradient."""
function flush_manager_buffers!(manager::M) where {M<:StatefulAlgorithms.ProcessManager}
    clear_buffer!(manager.state.batch_gradient)
    for worker in StatefulAlgorithms.workers(manager)
        add_buffer!(manager.state.batch_gradient, worker_context(worker).buffers)
        clear_buffer!(worker_context(worker).buffers)
    end

    nsamples = manager.state.nsamples[]
    nsamples > 0 || throw(ArgumentError("cannot flush an empty MNIST minibatch"))
    temp_scale = FT(manager.config.temp)^FT(manager.config.gradient_temp_power)
    scale_buffer!(manager.state.batch_gradient, FT(manager.config.gradient_sign) / (temp_scale * FT(nsamples)))
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
    optimiser = Optimisers.Adam(config.lr)
    algorithm = StatefulAlgorithms.resolve(input_field_nudgeless_algorithm(layer, config))
    state = InputFieldMNISTManagerState(
        layer,
        source,
        Ref(params),
        input_field_gradient_buffer(source, input_hidden_w[]),
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
    manager.state.opt_state, params = Optimisers.update(manager.state.opt_state, manager.state.params[], manager.state.batch_gradient)
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
        hidden::Int = 0
        output_replicas::Int = 0
        sweeps::T = zero(T)
        β::T = zero(T)
        covariance_samples::Int = 0
        covariance_sample_sweeps::T = zero(T)
        covariance_kick_steps::Int = 0
        covariance_stepsize::T = zero(T)
        covariance_noise_temp_factor::T = zero(T)
        lr::T = zero(T)
        gradient_sign::T = one(T)
        gradient_temp_power::T = one(T)
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
        hidden = checkpoint.config.hidden,
        output_replicas = checkpoint.config.output_replicas,
        sweeps = checkpoint.config.sweeps,
        β = checkpoint.config.β,
        covariance_samples = hasproperty(checkpoint.config, :covariance_samples) ? checkpoint.config.covariance_samples : 1,
        covariance_sample_sweeps = hasproperty(checkpoint.config, :covariance_sample_sweeps) ? checkpoint.config.covariance_sample_sweeps : one(FT),
        covariance_kick_steps = hasproperty(checkpoint.config, :covariance_kick_steps) ? checkpoint.config.covariance_kick_steps : 5,
        covariance_stepsize = hasproperty(checkpoint.config, :covariance_stepsize) ? checkpoint.config.covariance_stepsize : checkpoint.config.stepsize,
        covariance_noise_temp_factor = hasproperty(checkpoint.config, :covariance_noise_temp_factor) ? checkpoint.config.covariance_noise_temp_factor : one(FT),
        lr = checkpoint.config.lr,
        gradient_sign = hasproperty(checkpoint.config, :gradient_sign) ? checkpoint.config.gradient_sign : one(FT),
        gradient_temp_power = hasproperty(checkpoint.config, :gradient_temp_power) ? checkpoint.config.gradient_temp_power : one(FT),
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
    manager.state.opt_state = checkpoint.opt_state
    sync_after_update!(manager, checkpoint.params)
    return manager
end

"""Return the default file paths used by one retained run directory."""
function session_paths(outdir::P) where {P<:AbstractString}
    return (
        csv_path = joinpath(outdir, "mnist_784_120_40_nudgeless_adam.csv"),
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
    ax_acc = CM.Axis(fig[1, 1], xlabel = "epoch", ylabel = "accuracy", title = "MNIST 784-120-40 nudgeless accuracy")
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

"""Write the run settings needed to reproduce the nudgeless baseline."""
function write_settings!(path::P, config::C, relaxation_steps::I) where {P<:AbstractString,C<:InputFieldMNISTConfig,I<:Integer}
    open(path, "w") do io
        println(io, "# MNIST 784-120-40 Nudgeless Adam Baseline")
        println(io)
        println(io, "- paper: https://arxiv.org/pdf/2305.18321")
        println(io, "- architecture: `784 -> $(config.hidden) -> $(NCLASSES * config.output_replicas)`")
        println(io, "- sampled graph: hidden/output only, with no structural input layer")
        println(io, "- input handling: MNIST pixels in `[0, 1]` are projected through external `784 -> hidden` weights into a worker-local field")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- manager execution: `ChannelWorkers()` with one sample per manager job")
        println(io, "- worker graph adjacency: pointer-shared with source graph")
        println(io, "- worker graph base bias: pointer-shared with source graph")
        println(io, "- learning step: beta-zero covariance estimator from local `@Routine` measurements")
        println(io, "- validation: `ChannelWorkers()` ProcessManager with worker-local stats")
        println(io, "- job buffers: preallocated per-sample jobs reused across minibatches/evaluations")
        println(io, "- optimiser: `Optimisers.Adam($(config.lr))`")
        println(io, "- epochs/batchsize: `$(config.epochs)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- train eval per class: `$(config.train_eval_per_class)`")
        println(io, "- sweeps/relaxation steps: `$(config.sweeps)` / `$(relaxation_steps)`")
        println(io, "- covariance samples/sample sweeps: `$(config.covariance_samples)` / `$(config.covariance_sample_sweeps)`")
        println(io, "- covariance kick steps/noise temp factor/stepsize: `$(config.covariance_kick_steps)` / `$(config.covariance_noise_temp_factor)` / `$(config.covariance_stepsize)`")
        println(io, "- gradient sign: `$(config.gradient_sign)`")
        println(io, "- gradient temp power: `$(config.gradient_temp_power)`")
        println(io, "- beta/temp/stepsize: `$(config.β)` unused for training / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- weight scale/decay: `$(config.weight_scale)` / `$(config.weight_decay)`")
        println(io, "- resume from: `$(isempty(config.resume_from) ? "none" : config.resume_from)`")
        println(io, "- resume epoch: `$(config.resume_epoch)`")
    end
    return path
end

"""Validate baseline configuration before building long-lived runtime state."""
function validate_config!(config::C) where {C<:InputFieldMNISTConfig}
    config.workers > 0 || throw(ArgumentError("ISING_MNIST_IF_WORKERS must be positive"))
    config.batchsize > 0 || throw(ArgumentError("ISING_MNIST_IF_BATCHSIZE must be positive"))
    config.epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EPOCHS must be nonnegative"))
    config.covariance_samples > 0 || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_SAMPLES must be positive"))
    config.covariance_sample_sweeps > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_SAMPLE_SWEEPS must be positive"))
    config.gradient_temp_power >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_GRADIENT_TEMP_POWER must be nonnegative"))
    config.covariance_kick_steps > 0 || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_KICK_STEPS must be positive"))
    config.covariance_stepsize > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_STEPSIZE must be positive"))
    config.covariance_noise_temp_factor >= zero(FT) || throw(ArgumentError("ISING_MNIST_IF_COVARIANCE_NOISE_TEMP_FACTOR must be nonnegative"))
    config.temp > zero(FT) || throw(ArgumentError("ISING_MNIST_IF_TEMP must be positive for covariance scaling"))
    config.hidden == 120 || @warn "baseline paper hidden count is 120" hidden = config.hidden
    config.output_replicas == 4 || @warn "baseline paper output count is 40, i.e. four replicas per digit" output_replicas = config.output_replicas
    config.train_per_class < 5421 && @warn "this run uses a subsampled balanced training split, so it is not the full balanced MNIST baseline" train_per_class = config.train_per_class
    config.test_per_class < 892 && @warn "this run uses a subsampled balanced test split, so reported accuracy will be noisy" test_per_class = config.test_per_class
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    has_resume_checkpoint(config) && !isfile(config.resume_from) &&
        throw(ArgumentError("ISING_MNIST_IF_RESUME_FROM does not point to a checkpoint file"))
    return config
end

"""Construct one reusable nudgeless session that can be rerun or checkpoint-restarted in-process."""
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
    println("loading train-eval split")
    flush(stdout)
    xtrain_eval, ytrain_eval = config.train_eval_per_class > 0 ?
        balanced_mnist(:train, config.train_eval_per_class, config) :
        (zeros(FT, INPUT_DIM, 0), zeros(FT, NCLASSES * config.output_replicas, 0))
    println("loading test split")
    flush(stdout)
    xtest, ytest = balanced_mnist(:test, config.test_per_class, config)
    input_hidden_w = Ref(setup.input_hidden_w)
    println("constructing ", config.workers, "-worker nudgeless Adam manager")
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

"""Run baseline epochs on an already prepared session."""
function run_epochs!(session::S; start_epoch::Int = 0, stop_epoch::Int = session.config.epochs) where {S<:InputFieldMNISTSession}
    config = session.config
    for epoch in start_epoch:stop_epoch
        seconds = 0.0
        train_accuracy = missing
        train_loss = missing
        test_accuracy = missing
        test_loss = missing
        pred_counts = missing
        is_new_best = false
        if epoch > 0
            println("epoch ", epoch, " training")
            flush(stdout)
            order = Random.shuffle(Random.MersenneTwister(config.seed + epoch), collect(axes(session.xtrain, 2)))
            seconds = @elapsed begin
                for first_idx in 1:config.batchsize:length(order)
                    last_idx = min(first_idx + config.batchsize - 1, length(order))
                    batch_indices = @view order[first_idx:last_idx]
                    jobs = fill_jobs!(
                        session.train_jobs,
                        session.xtrain,
                        session.ytrain,
                        batch_indices,
                    )
                    run_minibatch!(session.manager, jobs)
                end
            end
        end

        # Validation is intentionally not run every epoch; on larger held-out sets it can dominate training time.
        should_eval_train = config.train_eval_per_class > 0 &&
            (epoch == 0 || epoch == config.epochs || (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0))
        if should_eval_train
            train = evaluate(session.validator, session.xtrain_eval, session.ytrain_eval, config, session.eval_jobs)
            train_accuracy = train.accuracy
            train_loss = train.loss
        end
        should_eval_test = epoch == 0 || epoch == config.epochs ||
            (epoch > 0 && config.eval_every > 0 && mod(epoch, config.eval_every) == 0)
        if should_eval_test
            println("epoch ", epoch, " evaluating")
            flush(stdout)
            test = evaluate(session.validator, session.xtest, session.ytest, config, session.eval_jobs)
            test_accuracy = test.accuracy
            test_loss = test.loss
            pred_counts = join(test.pred_counts, "-")
            if test.accuracy > session.best_accuracy[]
                session.best_accuracy[] = test.accuracy
                is_new_best = true
            end
        end

        norms = training_norms(session.manager)
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
    end
    return session
end

"""Persist final artifacts for a completed baseline session."""
function finalize_session!(session::S) where {S<:InputFieldMNISTSession}
    save_checkpoint(session.final_path, session.manager, session.config, session.rows)
    plot_metrics(joinpath(session.config.outdir, "learning_summary.png"), session.rows)
    println("saved MNIST 784-120-40 nudgeless baseline in ", session.config.outdir)
    return (; config = session.config, rows = session.rows, best_accuracy = session.best_accuracy[])
end

"""Close manager-backed state held by one live baseline session."""
function close_session!(session::S) where {S<:InputFieldMNISTSession}
    close(session.manager)
    close(session.validator)
    return nothing
end

"""Run the full baseline experiment."""
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
