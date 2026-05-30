using Dates
using Random
using SparseArrays

const BESPOKE_BASELINE_FILE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(BESPOKE_BASELINE_FILE)

const BESPOKE_OUTDIR = @__DIR__

"""Print one timestamped line from the bespoke baseline timing diagnostic."""
function bespoke_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Create the baseline config used by the single-sample bespoke diagnostic."""
function bespoke_single_example_config()
    return InputFieldMNISTConfig(;
        workers = 1,
        epochs = 1,
        batchsize = 1,
        train_per_class = 2,
        test_per_class = 1,
        train_eval_per_class = 1,
        eval_every = 1,
        β = FT(5),
        lr = FT(0.0015),
        weight_decay = FT(0),
        outdir = BESPOKE_OUTDIR,
    )
end

"""Build the baseline graph but force Metropolis dynamics in the layer metadata."""
function build_bespoke_metropolis_layer(config::C) where {C<:InputFieldMNISTConfig}
    graph = IsingLearning.MNISTArchitecture(
        hidden = config.hidden,
        output_replicas = config.output_replicas,
        precision = FT,
        weight_scale = config.weight_scale,
        rng = Random.MersenneTwister(config.seed),
    )
    II.temp!(graph, config.temp)
    relaxation_steps = max(1, round(Int, config.sweeps * active_units(graph)))
    dynamics = II.Metropolis()
    layer = IsingLearning.MNISTLayer(
        graph = graph,
        β = config.β,
        free_relaxation_steps = relaxation_steps,
        nudged_relaxation_steps = relaxation_steps,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return (; graph, layer, relaxation_steps)
end

"""Return the active non-input state indices used by the input-field baseline."""
function baseline_active_indices(graph::G) where {G}
    active = Int[]
    for layer_idx in 2:length(graph)
        append!(active, collect(Int, II.layerrange(graph[layer_idx])))
    end
    return active
end

"""Precompute CSC pointer bounds that skip input rows already represented by `input_b`."""
function baseline_active_ptr_bounds(graph::G, active_idxs::A) where {G,A<:AbstractVector{<:Integer}}
    adjacency = II.adj(graph)
    rows = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    starts = fill(1, size(adjacency, 2))
    stops = fill(0, size(adjacency, 2))
    input_last = last(II.layerrange(graph[1]))

    @inbounds for idx in active_idxs
        lo = colptr[idx]
        hi = colptr[idx + 1] - 1
        while lo <= hi && rows[lo] <= input_last
            lo += 1
        end
        starts[idx] = lo
        stops[idx] = hi
    end
    return (; starts, stops)
end

"""Randomize hidden/output state while keeping input spins zeroed for field-input mode."""
function randomize_baseline_state!(graph::G, rng::R) where {G,R<:Random.AbstractRNG}
    state = II.state(graph)
    @inbounds for idx in eachindex(state)
        state[idx] = FT(2) * rand(rng, FT) - one(FT)
    end
    fill!(II.state(graph[1]), zero(FT))
    return graph
end

"""Return the local field from optimized adjacency contraction plus MNIST fields."""
@inline function baseline_local_field(
    idx::I,
    rows::R,
    nzvals::Z,
    ptr_bounds::PB,
    state::S,
    base_b::B,
    input_b::P,
) where {
    I<:Integer,
    R<:AbstractVector{<:Integer},
    Z<:AbstractVector,
    PB<:NamedTuple,
    S<:AbstractVector,
    B<:AbstractVector,
    P<:AbstractVector,
}
    field = base_b[idx] + input_b[idx]
    @inbounds for ptr in ptr_bounds.starts[idx]:ptr_bounds.stops[idx]
        field += nzvals[ptr] * state[rows[ptr]]
    end
    return field
end

"""Return unclamped direct Metropolis energy change for one proposal."""
@inline function baseline_unclamped_delta_energy(
    idx::I,
    proposed::T,
    rows::R,
    nzvals::Z,
    ptr_bounds::PB,
    state::S,
    base_b::B,
    input_b::P,
) where {
    I<:Integer,
    T<:Real,
    R<:AbstractVector{<:Integer},
    Z<:AbstractVector,
    PB<:NamedTuple,
    S<:AbstractVector,
    B<:AbstractVector,
    P<:AbstractVector,
}
    old = state[idx]
    local_field = baseline_local_field(idx, rows, nzvals, ptr_bounds, state, base_b, input_b)
    return -(proposed - old) * local_field
end

"""Return clamped direct Metropolis energy change for one proposal."""
@inline function baseline_clamped_delta_energy(
    idx::I,
    proposed::T,
    rows::R,
    nzvals::Z,
    ptr_bounds::PB,
    state::S,
    base_b::B,
    input_b::P,
    clamp_y::Y,
    clamp_mask::M,
    clamp_beta::Q,
) where {
    I<:Integer,
    T<:Real,
    R<:AbstractVector{<:Integer},
    Z<:AbstractVector,
    PB<:NamedTuple,
    S<:AbstractVector,
    B<:AbstractVector,
    P<:AbstractVector,
    Y<:AbstractVector,
    M<:AbstractVector,
    Q<:Real,
}
    old = state[idx]
    delta = proposed - old
    local_field = baseline_local_field(idx, rows, nzvals, ptr_bounds, state, base_b, input_b)
    clamp_delta = clamp_beta * clamp_mask[idx] *
        (proposed * proposed - old * old - FT(2) * clamp_y[idx] * delta) / FT(2)
    return -delta * local_field + clamp_delta
end

"""Run unclamped direct continuous-state Metropolis updates over active spins."""
function baseline_bespoke_metropolis_free!(
    graph::G,
    active_idxs::A,
    ptr_bounds::PB,
    nsteps::I,
    rng::R,
) where {G,A<:AbstractVector{<:Integer},PB<:NamedTuple,I<:Integer,R<:Random.AbstractRNG}
    adjacency = II.adj(graph)
    rows = SparseArrays.rowvals(adjacency)
    nzvals = SparseArrays.nonzeros(adjacency)
    state = II.state(graph)
    base_b = IsingLearning._mnist_base_magfield(graph).b
    input_b = IsingLearning._mnist_input_magfield(graph).b
    temperature = II.temp(graph)
    accepted = 0
    nactive = length(active_idxs)

    @inbounds for _ in 1:Int(nsteps)
        idx = active_idxs[rand(rng, 1:nactive)]
        proposed = FT(2) * rand(rng, FT) - one(FT)
        ΔE = baseline_unclamped_delta_energy(
            idx,
            proposed,
            rows,
            nzvals,
            ptr_bounds,
            state,
            base_b,
            input_b,
        )
        if ΔE <= zero(FT) || rand(rng, FT) < exp(-ΔE / temperature)
            state[idx] = proposed
            accepted += 1
        end
    end
    return accepted
end

"""Run target-clamped direct continuous-state Metropolis updates over active spins."""
function baseline_bespoke_metropolis_nudged!(
    graph::G,
    active_idxs::A,
    ptr_bounds::PB,
    nsteps::I,
    rng::R,
) where {G,A<:AbstractVector{<:Integer},PB<:NamedTuple,I<:Integer,R<:Random.AbstractRNG}
    adjacency = II.adj(graph)
    rows = SparseArrays.rowvals(adjacency)
    nzvals = SparseArrays.nonzeros(adjacency)
    state = II.state(graph)
    base_b = IsingLearning._mnist_base_magfield(graph).b
    input_b = IsingLearning._mnist_input_magfield(graph).b
    clamping = IsingLearning._learning_clamping(graph)
    clamp_y = clamping.y
    clamp_mask = clamping.mask
    clamp_beta = clamping.β[]
    temperature = II.temp(graph)
    accepted = 0
    nactive = length(active_idxs)

    @inbounds for _ in 1:Int(nsteps)
        idx = active_idxs[rand(rng, 1:nactive)]
        proposed = FT(2) * rand(rng, FT) - one(FT)
        ΔE = baseline_clamped_delta_energy(
            idx,
            proposed,
            rows,
            nzvals,
            ptr_bounds,
            state,
            base_b,
            input_b,
            clamp_y,
            clamp_mask,
            clamp_beta,
        )
        if ΔE <= zero(FT) || rand(rng, FT) < exp(-ΔE / temperature)
            state[idx] = proposed
            accepted += 1
        end
    end
    return accepted
end

"""Run one complete direct free/nudged MNIST contrastive sample."""
function baseline_bespoke_contrastive_sample!(
    graph::G,
    x::X,
    y::Y,
    buffers::B,
    free_state::S,
    nudged_state::S,
    active_idxs::A,
    relaxation_steps::I,
    beta::T,
    rng::R,
    ptr_bounds::PB = baseline_active_ptr_bounds(graph, active_idxs),
) where {
    G,
    X<:AbstractVector,
    Y<:AbstractVector,
    B,
    S<:AbstractVector,
    A<:AbstractVector{<:Integer},
    I<:Integer,
    T<:Real,
    R<:Random.AbstractRNG,
    PB<:NamedTuple,
}
    clear_buffer!(buffers)

    # Free phase: random state, image as input field, no output clamping.
    randomize_baseline_state!(graph, rng)
    IsingLearning.apply_input(graph, x)
    IsingLearning.set_clamping_beta!(graph, zero(FT))
    free_seconds = @elapsed accepted_free = baseline_bespoke_metropolis_free!(graph, active_idxs, ptr_bounds, relaxation_steps, rng)
    free_state .= II.state(graph)

    # Nudged phase: restart from free state, keep image field, install target clamp.
    II.state(graph) .= free_state
    IsingLearning.apply_input(graph, x)
    IsingLearning.apply_targets(graph, y)
    IsingLearning.set_clamping_beta!(graph, beta)
    nudged_seconds = @elapsed accepted_nudged = baseline_bespoke_metropolis_nudged!(graph, active_idxs, ptr_bounds, relaxation_steps, rng)
    nudged_state .= II.state(graph)

    gradient_seconds = @elapsed accumulate_input_field_gradient!(
        graph,
        nudged_state,
        free_state,
        x,
        buffers,
        beta,
    )
    normalize_seconds = @elapsed scale_buffer!(buffers, inv(FT(beta)))
    IsingLearning.set_clamping_beta!(graph, zero(FT))

    return (;
        free_seconds,
        nudged_seconds,
        gradient_seconds,
        normalize_seconds,
        accepted_free,
        accepted_nudged,
    )
end

"""Append one CSV row for the bespoke direct-Metropolis diagnostic."""
function append_bespoke_row!(row::R) where {R<:NamedTuple}
    path = joinpath(BESPOKE_OUTDIR, "bespoke_metropolis_single_example.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run the warmup and measured direct-Metropolis baseline timing."""
function main()
    mkpath(BESPOKE_OUTDIR)
    config = bespoke_single_example_config()
    bespoke_log("building bespoke Metropolis baseline"; threads = Threads.nthreads())
    setup_seconds = @elapsed setup = build_bespoke_metropolis_layer(config)
    graph = shared_worker_graph(setup.graph)
    II.temp!(graph, config.temp)
    active_idxs = baseline_active_indices(graph)
    buffers = IsingLearning.gradient_buffer(graph)
    free_state = similar(II.state(graph))
    nudged_state = similar(II.state(graph))
    rng = Random.MersenneTwister(config.seed + 90_001)

    bespoke_log("loading tiny MNIST split")
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    warmup = baseline_bespoke_contrastive_sample!(
        graph,
        view(xtrain, :, 1),
        view(ytrain, :, 1),
        buffers,
        free_state,
        nudged_state,
        active_idxs,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    measured_seconds = @elapsed measured = baseline_bespoke_contrastive_sample!(
        graph,
        view(xtrain, :, 2),
        view(ytrain, :, 2),
        buffers,
        free_state,
        nudged_state,
        active_idxs,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "784-120-40",
        implementation = "bespoke_direct_continuous_metropolis",
        measured_examples = 1,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        beta = config.β,
        temp = config.temp,
        relaxation_steps = setup.relaxation_steps,
        work_steps_per_example = 2 * setup.relaxation_steps,
        active_units = length(active_idxs),
        setup_seconds,
        data_seconds,
        warmup_total_seconds = warmup.free_seconds + warmup.nudged_seconds + warmup.gradient_seconds + warmup.normalize_seconds,
        measured_wall_seconds = measured_seconds,
        free_seconds = measured.free_seconds,
        nudged_seconds = measured.nudged_seconds,
        gradient_seconds = measured.gradient_seconds,
        normalize_seconds = measured.normalize_seconds,
        accepted_free = measured.accepted_free,
        accepted_nudged = measured.accepted_nudged,
        acceptance_rate = (measured.accepted_free + measured.accepted_nudged) / (2 * setup.relaxation_steps),
        steps_per_second = (2 * setup.relaxation_steps) / measured_seconds,
    )
    csv_path = append_bespoke_row!(row)
    bespoke_log("bespoke Metropolis summary"; row..., csv = csv_path)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
