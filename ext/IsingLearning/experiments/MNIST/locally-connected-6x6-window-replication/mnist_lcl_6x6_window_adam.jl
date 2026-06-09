include(joinpath(@__DIR__, "..", "784-120-40-baseline", "mnist_784_120_40_adam.jl"))

const LCL_WINDOW = parse(Int, get(ENV, "ISING_MNIST_LCL_WINDOW", "6"))
const LCL_STRIDE = parse(Int, get(ENV, "ISING_MNIST_LCL_STRIDE", "1"))
const LCL_TANGENT_NUDGE = parse(Bool, lowercase(get(ENV, "ISING_MNIST_LCL_TANGENT_NUDGE", "true")))
const LCL_PATCH_INPUT_IDXS = Ref(Vector{Vector{Int}}())
const LCL_INPUT_MASK = Ref(falses(0, 0))

"""Return the number of hidden rows/columns for a valid locally connected window."""
function lcl_hidden_side(window::I, stride::J) where {I<:Integer,J<:Integer}
    window > 0 || throw(ArgumentError("LCL window must be positive"))
    stride > 0 || throw(ArgumentError("LCL stride must be positive"))
    side = IsingLearning.D_MNIST
    window <= side || throw(ArgumentError("LCL window $(window) exceeds MNIST side $(side)"))
    return div(side - Int(window), Int(stride)) + 1
end

"""Build the hidden-to-input patch index lists for a valid local receptive field."""
function lcl_patch_input_indices(window::I, stride::J) where {I<:Integer,J<:Integer}
    hidden_side = lcl_hidden_side(window, stride)
    patches = Vector{Vector{Int}}(undef, hidden_side * hidden_side)
    ptr = 1
    @inbounds for hrow in 1:hidden_side
        row0 = (hrow - 1) * Int(stride) + 1
        for hcol in 1:hidden_side
            col0 = (hcol - 1) * Int(stride) + 1
            patch = Vector{Int}(undef, Int(window) * Int(window))
            p = 1
            for irow in row0:(row0 + Int(window) - 1)
                for icol in col0:(col0 + Int(window) - 1)
                    patch[p] = (icol - 1) * IsingLearning.D_MNIST + irow
                    p += 1
                end
            end
            patches[ptr] = patch
            ptr += 1
        end
    end
    return patches
end

"""Construct a dense mask whose `true` entries are trainable LCL input weights."""
function lcl_input_mask(patches::P) where {P<:AbstractVector}
    mask = falses(length(patches), INPUT_DIM)
    @inbounds for hidden_idx in eachindex(patches)
        for input_idx in patches[hidden_idx]
            mask[hidden_idx, input_idx] = true
        end
    end
    return mask
end

"""Initialize the local `784 -> hidden` weights and store their structural mask."""
function lcl_input_hidden_weights(
    rng::R,
    ::Type{T},
    scale::S,
    window::I,
    stride::J,
) where {R<:AbstractRNG,T<:AbstractFloat,S<:Real,I<:Integer,J<:Integer}
    patches = lcl_patch_input_indices(window, stride)
    weights = zeros(T, length(patches), INPUT_DIM)
    @inbounds for hidden_idx in eachindex(patches)
        for input_idx in patches[hidden_idx]
            weights[hidden_idx, input_idx] = T(scale) * randn(rng, T)
        end
    end
    LCL_PATCH_INPUT_IDXS[] = patches
    LCL_INPUT_MASK[] = lcl_input_mask(patches)
    return weights
end

"""Create the 6x6-window LCL hidden/output graph with external image-field input."""
function build_layer(config::C) where {C<:InputFieldMNISTConfig}
    hidden_units = Int(config.hidden)
    output_units = NCLASSES * Int(config.output_replicas)
    hidden_rows, hidden_cols = mnist_layer_shape(hidden_units)
    output_rows, output_cols = mnist_layer_shape(output_units)
    side = IsingLearning.D_MNIST
    rng = Random.MersenneTwister(config.seed)
    scale = FT(config.weight_scale)

    expected_hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2
    hidden_units == expected_hidden ||
        throw(ArgumentError("LCL hidden count $(hidden_units) does not match window=$(LCL_WINDOW), stride=$(LCL_STRIDE), expected $(expected_hidden)"))

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

    input_hidden_w = lcl_input_hidden_weights(rng, FT, scale, LCL_WINDOW, LCL_STRIDE)

    # Paper-style first target: no hidden intralayer couplings, dense hidden-to-output readout.
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
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return (; graph, layer, relaxation_steps, input_hidden_w)
end

"""Accumulate the LCL gradient without touching nonlocal `784 -> hidden` weights."""
function accumulate_input_field_gradient!(
    isinggraph::G,
    nudged_state::N,
    equilibrium_state::E,
    x::X,
    buffers::B,
    β::R,
) where {G,N<:AbstractVector,E<:AbstractVector,X<:AbstractVector,B,R<:Real}
    IsingLearning.contrastive_gradient(isinggraph, nudged_state, equilibrium_state, β; buffers = buffers)
    patches = LCL_PATCH_INPUT_IDXS[]
    length(patches) == size(buffers.w_input, 1) || error("LCL patch table does not match hidden count")
    @inbounds for hidden_idx in eachindex(patches)
        state_delta = nudged_state[hidden_idx] - equilibrium_state[hidden_idx]
        for input_idx in patches[hidden_idx]
            buffers.w_input[hidden_idx, input_idx] += -x[input_idx] * state_delta
        end
    end
    return
end

"""Install a projected image field plus fixed tangent output force into one worker graph."""
function install_tangent_projected_input_field!(
    isinggraph::G,
    input_hidden_w::W,
    x::X,
    y::Y,
    equilibrium_state::E,
    pattern::P,
    output_idxs::O,
    phase_beta::T,
) where {
    G,
    W<:AbstractMatrix,
    X<:AbstractVector,
    Y<:AbstractVector,
    E<:AbstractVector,
    P<:AbstractVector,
    O<:AbstractVector{<:Integer},
    T<:Real,
}
    project_input_field_pattern!(pattern, input_hidden_w, x)
    β = FT(phase_beta)
    @inbounds for target_idx in eachindex(y)
        graph_idx = Int(output_idxs[target_idx])
        pattern[graph_idx] += β * (y[target_idx] - equilibrium_state[graph_idx])
    end
    install_input_field_pattern!(isinggraph, pattern)
    return isinggraph
end

# Install paper-style tangent nudging without routing the whole process context into helper code.
StatefulAlgorithms.@ProcessAlgorithm function ApplyTangentProjectedInputFieldRef!(
    isinggraph::G,
    input_hidden_w,
    x,
    y,
    equilibrium_state::AbstractVector,
    input_pattern::AbstractVector,
    output_idxs::AbstractVector,
    phase_beta::Float32,
) where G
    install_tangent_projected_input_field!(
        isinggraph,
        ref_value(input_hidden_w),
        ref_value(x),
        ref_value(y),
        equilibrium_state,
        input_pattern,
        output_idxs,
        phase_beta,
    )
    return nothing
end

"""Build the LCL nudged routine, optionally using paper-style tangent nudging."""
function input_field_nudged_phase_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    nudged_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
    default_β = layer.β
    output_idxs = collect(layer.output_layer)

    if LCL_TANGENT_NUDGE
        return StatefulAlgorithms.@Routine begin
            @state x
            @state y
            @state input_hidden_w
            @state input_pattern
            @state phase_beta = default_β
            @state equilibrium_state
            @state nudged_state = zeros(n_units)
            @state output_idxs = output_idxs
            @alias dynamics = dynamics_algorithm

            # Tangent nudging keeps the free-equilibrium output error fixed as a magnetic field.
            IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
            ApplyTangentProjectedInputFieldRef!(
                dynamics.model,
                input_hidden_w,
                x,
                y,
                equilibrium_state,
                input_pattern,
                output_idxs,
                phase_beta,
            )
            model = @repeat nudged_steps dynamics()
            IsingLearning.CopyGraphState!(nudged_state, model)
        end
    end

    return StatefulAlgorithms.@Routine begin
        @state x
        @state y
        @state input_hidden_w
        @state input_pattern
        @state phase_beta = default_β
        @state equilibrium_state
        @state nudged_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Fallback to the baseline clamp-style nudge when explicitly requested.
        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        ApplyTargetsRef!(dynamics.model, y)
        SetInputFieldClampingBeta!(dynamics.model, phase_beta)
        model = @repeat nudged_steps dynamics()
        IsingLearning.CopyGraphState!(nudged_state, model)
        SetInputFieldClampingBeta!(dynamics.model, 0f0)
    end
end

"""Synchronize updated graph parameters and enforce the structural LCL input mask."""
function sync_after_update!(manager::M, params::P) where {M<:StatefulAlgorithms.ProcessManager,P}
    mask = LCL_INPUT_MASK[]
    !isempty(mask) && (params.w_input .*= mask)
    IsingLearning.sync_graph_params!(manager.state.source_graph, (; w = params.w, b = params.b))
    manager.state.input_hidden_w[] = params.w_input
    for worker in StatefulAlgorithms.workers(manager)
        IsingLearning._sync_worker_graph_params!(worker_graph(worker), manager.state.source_graph, (; w = params.w, b = params.b))
    end
    return manager
end

"""Validate the LCL replication configuration before allocating workers."""
function validate_config!(config::C) where {C<:InputFieldMNISTConfig}
    config.workers > 0 || throw(ArgumentError("ISING_MNIST_IF_WORKERS must be positive"))
    config.batchsize > 0 || throw(ArgumentError("ISING_MNIST_IF_BATCHSIZE must be positive"))
    config.epochs >= 0 || throw(ArgumentError("ISING_MNIST_IF_EPOCHS must be nonnegative"))
    expected_hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2
    config.hidden == expected_hidden ||
        throw(ArgumentError("LCL hidden count must be $(expected_hidden) for window=$(LCL_WINDOW), stride=$(LCL_STRIDE)"))
    config.output_replicas == 1 || @warn "paper LCL target uses 10 output nodes; replicas make this a deliberate variant" output_replicas = config.output_replicas
    config.train_per_class < 5421 && @warn "this run uses a subsampled balanced training split, so it is diagnostic, not a full-paper replication" train_per_class = config.train_per_class
    config.test_per_class < 892 && @warn "this run uses a subsampled balanced test split, so reported accuracy will be noisy" test_per_class = config.test_per_class
    Threads.nthreads() < config.workers && @warn "Julia was started with fewer threads than requested manager workers" threads = Threads.nthreads() workers = config.workers
    has_resume_checkpoint(config) && !isfile(config.resume_from) &&
        throw(ArgumentError("ISING_MNIST_IF_RESUME_FROM does not point to a checkpoint file"))
    return config
end

"""Write the run settings needed to reproduce one LCL replication run."""
function write_settings!(path::P, config::C, relaxation_steps::I) where {P<:AbstractString,C<:InputFieldMNISTConfig,I<:Integer}
    open(path, "w") do io
        println(io, "# MNIST Locally Connected 6x6 Window Replication")
        println(io)
        println(io, "- source paper: https://arxiv.org/abs/2601.21945")
        println(io, "- target section: layered LCL MNIST, Figure 5")
        println(io, "- architecture: `28x28 input field -> $(config.hidden) hidden LCL units -> $(NCLASSES * config.output_replicas) output units`")
        println(io, "- LCL window/stride: `$(LCL_WINDOW)` / `$(LCL_STRIDE)`")
        println(io, "- hidden grid: `$(lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)) x $(lcl_hidden_side(LCL_WINDOW, LCL_STRIDE))`")
        println(io, "- trainable input weights: `$(count(LCL_INPUT_MASK[]))`")
        println(io, "- hidden intralayer couplings: none")
        println(io, "- hidden/output readout: dense bidirectional couplings")
        println(io, "- input handling: pixels projected through masked local weights into worker-local magnetic field")
        println(io, "- nudging: `$(LCL_TANGENT_NUDGE ? "tangent fixed free-equilibrium error force" : "baseline clamp-style nudge")`")
        println(io, "- workers: `$(config.workers)`")
        println(io, "- optimiser: `Optimisers.Adam($(config.lr))`")
        println(io, "- epochs/batchsize: `$(config.epochs)` / `$(config.batchsize)`")
        println(io, "- train/test per class: `$(config.train_per_class)` / `$(config.test_per_class)`")
        println(io, "- train eval per class: `$(config.train_eval_per_class)`")
        println(io, "- sweeps/relaxation steps: `$(config.sweeps)` / `$(relaxation_steps)`")
        println(io, "- beta/temp/stepsize: `$(config.β)` / `$(config.temp)` / `$(config.stepsize)`")
        println(io, "- weight scale/decay: `$(config.weight_scale)` / `$(config.weight_decay)`")
        println(io, "- resume from: `$(isempty(config.resume_from) ? "none" : config.resume_from)`")
        println(io, "- note: this is an Ising/Langevin implementation of the paper's LCL topology, not an exact XY-angle simulator.")
    end
    return path
end

"""Serialize optimizer-facing LCL parameters and run metadata."""
function save_checkpoint(path::P, manager::M, config::C, rows::R) where {P<:AbstractString,M<:StatefulAlgorithms.ProcessManager,C<:InputFieldMNISTConfig,R<:AbstractVector}
    mkpath(dirname(path))
    epoch_tag = isempty(rows) ? "unknown" : string(getproperty(rows[end], :epoch))
    tmp = path * "." * string(rand(UInt)) * ".tmp"
    open(tmp, "w"; lock = false) do io
        serialize(io, (;
            architecture = "lcl-window$(LCL_WINDOW)-stride$(LCL_STRIDE)-$(config.hidden)-$(NCLASSES * config.output_replicas)",
            params = manager.state.params[],
            opt_state = manager.state.opt_state,
            rows,
            config,
            lcl_window = LCL_WINDOW,
            lcl_stride = LCL_STRIDE,
        ))
    end
    for attempt in 1:5
        try
            mv(tmp, path; force = true)
            return path
        catch err
            attempt == 5 || (sleep(0.2 * attempt); continue)
            fallback = joinpath(dirname(path), first(splitext(basename(path))) * "_epoch$(epoch_tag)_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".bin")
            try
                mv(tmp, fallback; force = true)
                @warn "checkpoint target was locked; wrote fallback checkpoint instead" target = path fallback exception = (err, catch_backtrace())
                return fallback
            catch fallback_err
                @warn "failed to move checkpoint fallback; leaving temp checkpoint in place" target = path temp = tmp exception = (fallback_err, catch_backtrace())
                return tmp
            end
        end
    end
    return path
end

"""Run the LCL replication defaults while still allowing environment overrides."""
function main()
    hidden_default = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2
    base = InputFieldMNISTConfig()
    config = updated_config(
        base;
        workers = haskey(ENV, "ISING_MNIST_IF_WORKERS") ? base.workers : 32,
        epochs = haskey(ENV, "ISING_MNIST_IF_EPOCHS") ? base.epochs : 200,
        batchsize = haskey(ENV, "ISING_MNIST_IF_BATCHSIZE") ? base.batchsize : 200,
        hidden = haskey(ENV, "ISING_MNIST_IF_HIDDEN") ? base.hidden : hidden_default,
        output_replicas = haskey(ENV, "ISING_MNIST_IF_OUTPUT_REPLICAS") ? base.output_replicas : 1,
        lr = haskey(ENV, "ISING_MNIST_IF_LR") ? base.lr : FT(1f-4),
        β = haskey(ENV, "ISING_MNIST_IF_BETA") ? base.β : FT(0.1),
        outdir = haskey(ENV, "ISING_MNIST_IF_OUTDIR") ? base.outdir :
            joinpath(@__DIR__, "experiments", "current", default_run_dirname("mnist_lcl_6x6_stride1_adam")),
    )
    return run_config!(config)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
