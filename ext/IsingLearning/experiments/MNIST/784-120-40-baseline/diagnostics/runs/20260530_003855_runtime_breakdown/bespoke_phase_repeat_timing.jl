include(joinpath(@__DIR__, "bespoke_metropolis_single_example.jl"))

"""Append one repeated direct-phase timing row."""
function append_bespoke_phase_row!(row::R) where {R<:NamedTuple}
    path = joinpath(BESPOKE_OUTDIR, "bespoke_phase_repeat_timing.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run repeated direct free/nudged phase timings for the bespoke baseline loop."""
function main()
    config = bespoke_single_example_config()
    repeat_count = parse(Int, get(ENV, "ISING_MNIST_BESPOKE_PHASE_REPEATS", "8"))
    setup = build_bespoke_metropolis_layer(config)
    xtrain, ytrain = balanced_mnist(:train, max(config.train_per_class, repeat_count + 1), config)
    graph = shared_worker_graph(setup.graph)
    II.temp!(graph, config.temp)
    active_idxs = baseline_active_indices(graph)
    buffers = IsingLearning.gradient_buffer(graph)
    free_state = similar(II.state(graph))
    nudged_state = similar(II.state(graph))
    rng = Random.MersenneTwister(config.seed + 92_001)

    bespoke_log("warming direct phase repeat diagnostic")
    baseline_bespoke_contrastive_sample!(
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

    rows = NamedTuple[]
    for repeat_idx in 1:repeat_count
        sample_idx = repeat_idx + 1
        wall = @elapsed stats = baseline_bespoke_contrastive_sample!(
            graph,
            view(xtrain, :, sample_idx),
            view(ytrain, :, sample_idx),
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
            repeat_idx,
            sample_idx,
            wall_seconds = wall,
            free_seconds = stats.free_seconds,
            nudged_seconds = stats.nudged_seconds,
            gradient_seconds = stats.gradient_seconds,
            normalize_seconds = stats.normalize_seconds,
            accepted_free = stats.accepted_free,
            accepted_nudged = stats.accepted_nudged,
            free_acceptance = stats.accepted_free / setup.relaxation_steps,
            nudged_acceptance = stats.accepted_nudged / setup.relaxation_steps,
            nudged_over_free = stats.nudged_seconds / stats.free_seconds,
        )
        push!(rows, row)
        append_bespoke_phase_row!(row)
        bespoke_log("direct phase repeat"; row...)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
