using Dates
using LinearAlgebra
using Random
using SparseArrays

include(joinpath(@__DIR__, "bespoke_metropolis_single_example.jl"))

const FIELD_INPUT_OUTDIR = @__DIR__

"""Print one timestamped line from the hidden/output field-input diagnostic."""
function field_input_log(message::S; kwargs...) where {S<:AbstractString}
    print("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] ", message)
    for (key, value) in kwargs
        print(" ", key, "=", value)
    end
    println()
    flush(stdout)
    return nothing
end

"""Return CSC pointer bounds over all rows for the reduced hidden/output graph."""
function reduced_active_ptr_bounds(graph::G, active_idxs::A) where {G,A<:AbstractVector{<:Integer}}
    adjacency = II.adj(graph)
    colptr = SparseArrays.getcolptr(adjacency)
    starts = fill(1, size(adjacency, 2))
    stops = fill(0, size(adjacency, 2))

    @inbounds for idx in active_idxs
        starts[idx] = colptr[idx]
        stops[idx] = colptr[idx + 1] - 1
    end
    return (; starts, stops)
end

"""Randomize every state in the reduced hidden/output graph."""
function randomize_reduced_state!(graph::G, rng::R) where {G,R<:Random.AbstractRNG}
    state = II.state(graph)
    @inbounds for idx in eachindex(state)
        state[idx] = FT(2) * rand(rng, FT) - one(FT)
    end
    return graph
end

"""Extract input-hidden weights and hidden-output adjacency from the full baseline graph."""
function reduced_field_input_components(source_graph::G) where {G}
    source_layers = getfield(source_graph, :layers)
    input_idxs = collect(Int, II.layerrange(source_graph[1]))
    hidden_idxs = collect(Int, II.layerrange(source_graph[2]))
    output_idxs = collect(Int, II.layerrange(source_graph[3]))
    reduced_old_idxs = vcat(hidden_idxs, output_idxs)
    old_to_new = zeros(Int, maximum(reduced_old_idxs))
    @inbounds for (new_idx, old_idx) in enumerate(reduced_old_idxs)
        old_to_new[old_idx] = new_idx
    end

    source_adj = II.adj(source_graph)
    source_rows = SparseArrays.rowvals(source_adj)
    source_colptr = SparseArrays.getcolptr(source_adj)
    source_nzvals = SparseArrays.nonzeros(source_adj)
    input_first = first(input_idxs)
    input_last = last(input_idxs)

    # Dense image projection is the whole input layer's only role in this reduced model.
    input_hidden_w = zeros(FT, length(hidden_idxs), length(input_idxs))
    @inbounds for (hpos, hidx) in enumerate(hidden_idxs)
        for ptr in source_colptr[hidx]:(source_colptr[hidx + 1] - 1)
            row = source_rows[ptr]
            input_first <= row <= input_last || continue
            input_hidden_w[hpos, row - input_first + 1] = source_nzvals[ptr]
        end
    end

    # Keep only hidden/output couplings in the sampled graph.
    rows = Int[]
    cols = Int[]
    vals = FT[]
    @inbounds for old_col in reduced_old_idxs
        new_col = old_to_new[old_col]
        for ptr in source_colptr[old_col]:(source_colptr[old_col + 1] - 1)
            old_row = source_rows[ptr]
            old_row <= length(old_to_new) || continue
            new_row = old_to_new[old_row]
            new_row == 0 && continue
            push!(rows, new_row)
            push!(cols, new_col)
            push!(vals, source_nzvals[ptr])
        end
    end

    reduced_adj = II.UndirectedAdjacency(sparse(rows, cols, vals, length(reduced_old_idxs), length(reduced_old_idxs)))
    return (;
        hidden_layer = source_layers[2],
        output_layer = source_layers[3],
        input_hidden_w,
        reduced_adj,
        hidden_count = length(hidden_idxs),
        output_count = length(output_idxs),
    )
end

"""Build a hidden/output graph with separate base-bias and image-field magnetic terms."""
function build_reduced_field_input_layer(config::C) where {C<:InputFieldMNISTConfig}
    setup = build_bespoke_metropolis_layer(config)
    components = reduced_field_input_components(setup.graph)
    nstates_reduced = components.hidden_count + components.output_count
    base_bias = zeros(FT, nstates_reduced)
    image_field = zeros(FT, nstates_reduced)
    hamiltonian = II.Bilinear() +
        II.MagField(b = II.Force(base_bias)) +
        II.MagField(b = II.Force(image_field)) +
        II.Clamping(
            β = II.UniformArray(zero(FT)),
            y = g -> II.filltype(Vector, zero(FT), II.statelen(g)),
        )

    graph = II.IsingGraph(
        components.hidden_layer,
        components.output_layer,
        hamiltonian;
        precision = FT,
        adj = components.reduced_adj,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    active_idxs = collect(Int, 1:II.nstates(graph))
    relaxation_steps = max(1, round(Int, config.sweeps * length(active_idxs)))
    return (;
        graph,
        input_hidden_w = components.input_hidden_w,
        active_idxs,
        ptr_bounds = reduced_active_ptr_bounds(graph, active_idxs),
        relaxation_steps,
    )
end

"""Write the image-induced hidden field into the graph's second magnetic field."""
function install_reduced_image_field!(
    graph::G,
    input_hidden_w::W,
    x::X,
) where {G,W<:AbstractMatrix,X<:AbstractVector}
    image_b = IsingLearning._mnist_input_magfield(graph).b
    fill!(image_b, zero(eltype(image_b)))
    hidden_view = @view image_b[1:size(input_hidden_w, 1)]
    mul!(hidden_view, input_hidden_w, x)
    return graph
end

"""Accumulate raw input-hidden gradients for the field-only input representation."""
function accumulate_reduced_input_weight_gradient!(
    input_w_gradient::GW,
    x::X,
    nudged_state::S,
    free_state::S,
    β::T,
) where {GW<:AbstractMatrix,X<:AbstractVector,S<:AbstractVector,T<:Real}
    invβ = inv(FT(β))
    hidden_count = size(input_w_gradient, 1)

    # External weights are stored once, so this has no symmetric-edge 1/2 factor.
    @inbounds for input_idx in eachindex(x)
        xval = x[input_idx] * invβ
        for hidden_idx in 1:hidden_count
            input_w_gradient[hidden_idx, input_idx] += -xval * (nudged_state[hidden_idx] - free_state[hidden_idx])
        end
    end
    return input_w_gradient
end

"""Run one complete reduced-graph contrastive sample with image input as a magnetic field."""
function reduced_field_input_contrastive_sample!(
    graph::G,
    input_hidden_w::W,
    x::X,
    y::Y,
    buffers::B,
    input_w_gradient::GW,
    free_state::S,
    nudged_state::S,
    active_idxs::A,
    ptr_bounds::PB,
    relaxation_steps::I,
    beta::T,
    rng::R,
) where {
    G,
    W<:AbstractMatrix,
    X<:AbstractVector,
    Y<:AbstractVector,
    B,
    GW<:AbstractMatrix,
    S<:AbstractVector,
    A<:AbstractVector{<:Integer},
    PB<:NamedTuple,
    I<:Integer,
    T<:Real,
    R<:Random.AbstractRNG,
}
    clear_buffer!(buffers)
    fill!(input_w_gradient, zero(eltype(input_w_gradient)))

    projection_seconds = @elapsed install_reduced_image_field!(graph, input_hidden_w, x)

    # Free phase: the image is a fixed hidden field, with no output clamping.
    randomize_reduced_state!(graph, rng)
    IsingLearning.set_clamping_beta!(graph, zero(FT))
    free_seconds = @elapsed accepted_free = baseline_bespoke_metropolis_free!(
        graph,
        active_idxs,
        ptr_bounds,
        relaxation_steps,
        rng,
    )
    free_state .= II.state(graph)

    # Nudged phase: restart, keep the same image field, and clamp only output units.
    II.state(graph) .= free_state
    IsingLearning.apply_targets(graph, y)
    IsingLearning.set_clamping_beta!(graph, beta)
    nudged_seconds = @elapsed accepted_nudged = baseline_bespoke_metropolis_nudged!(
        graph,
        active_idxs,
        ptr_bounds,
        relaxation_steps,
        rng,
    )
    nudged_state .= II.state(graph)

    gradient_seconds = @elapsed begin
        IsingLearning.contrastive_gradient(graph, nudged_state, free_state, beta; buffers)
        accumulate_reduced_input_weight_gradient!(input_w_gradient, x, nudged_state, free_state, beta)
    end
    normalize_seconds = @elapsed scale_buffer!(buffers, inv(FT(beta)))
    IsingLearning.set_clamping_beta!(graph, zero(FT))

    return (;
        projection_seconds,
        free_seconds,
        nudged_seconds,
        gradient_seconds,
        normalize_seconds,
        accepted_free,
        accepted_nudged,
    )
end

"""Append one CSV row for the reduced hidden/output field-input diagnostic."""
function append_field_input_row!(row::R) where {R<:NamedTuple}
    path = joinpath(FIELD_INPUT_OUTDIR, "hidden_output_field_input_single_example.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run the field-input hidden/output diagnostic for one warmed MNIST sample."""
function main()
    mkpath(FIELD_INPUT_OUTDIR)
    config = bespoke_single_example_config()
    field_input_log("building reduced hidden/output field-input graph"; threads = Threads.nthreads())
    setup_seconds = @elapsed setup = build_reduced_field_input_layer(config)
    graph = setup.graph
    buffers = IsingLearning.gradient_buffer(graph)
    input_w_gradient = similar(setup.input_hidden_w)
    free_state = similar(II.state(graph))
    nudged_state = similar(II.state(graph))
    rng = Random.MersenneTwister(config.seed + 123_451)

    field_input_log("loading tiny MNIST split")
    data_seconds = @elapsed xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    warmup = reduced_field_input_contrastive_sample!(
        graph,
        setup.input_hidden_w,
        view(xtrain, :, 1),
        view(ytrain, :, 1),
        buffers,
        input_w_gradient,
        free_state,
        nudged_state,
        setup.active_idxs,
        setup.ptr_bounds,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    measured_seconds = @elapsed measured = reduced_field_input_contrastive_sample!(
        graph,
        setup.input_hidden_w,
        view(xtrain, :, 2),
        view(ytrain, :, 2),
        buffers,
        input_w_gradient,
        free_state,
        nudged_state,
        setup.active_idxs,
        setup.ptr_bounds,
        setup.relaxation_steps,
        config.β,
        rng,
    )

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "120-40_field_input_from_784",
        implementation = "bespoke_reduced_hidden_output_two_magfields",
        measured_examples = 1,
        threads = Threads.nthreads(),
        sweeps = config.sweeps,
        beta = config.β,
        temp = config.temp,
        relaxation_steps = setup.relaxation_steps,
        work_steps_per_example = 2 * setup.relaxation_steps,
        active_units = length(setup.active_idxs),
        input_projection_weights = length(setup.input_hidden_w),
        sampled_nonzeros = length(SparseArrays.nonzeros(II.adj(graph))),
        setup_seconds,
        data_seconds,
        warmup_total_seconds = warmup.projection_seconds + warmup.free_seconds +
            warmup.nudged_seconds + warmup.gradient_seconds + warmup.normalize_seconds,
        measured_wall_seconds = measured_seconds,
        projection_seconds = measured.projection_seconds,
        free_seconds = measured.free_seconds,
        nudged_seconds = measured.nudged_seconds,
        gradient_seconds = measured.gradient_seconds,
        normalize_seconds = measured.normalize_seconds,
        accepted_free = measured.accepted_free,
        accepted_nudged = measured.accepted_nudged,
        acceptance_rate = (measured.accepted_free + measured.accepted_nudged) / (2 * setup.relaxation_steps),
        steps_per_second = (2 * setup.relaxation_steps) / measured_seconds,
    )
    csv_path = append_field_input_row!(row)
    field_input_log("field-input reduced summary"; row..., csv = csv_path)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
