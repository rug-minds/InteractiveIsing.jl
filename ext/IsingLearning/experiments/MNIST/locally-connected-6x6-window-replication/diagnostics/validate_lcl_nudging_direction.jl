using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using Random

include(joinpath(@__DIR__, "..", "mnist_lcl_6x6_window_adam.jl"))

"""Return output-local force from a Hamiltonian derivative at one graph state."""
function output_force(graph::G, output_idxs::O) where {G,O<:AbstractVector{<:Integer}}
    return [-II.calculate(II.d_iH(), graph.hamiltonian, graph, idx) for idx in output_idxs]
end

"""Validate that target nudges add force toward the target class and away from others."""
function main()
    rng = Random.MersenneTwister(20260607)
    config = updated_config(
        InputFieldMNISTConfig();
        workers = 1,
        epochs = 0,
        batchsize = 1,
        train_per_class = 1,
        test_per_class = 1,
        train_eval_per_class = 0,
        hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2,
        output_replicas = 1,
        β = 0.3f0,
        outdir = joinpath(@__DIR__, "tmp_validate_lcl_nudging_direction"),
    )
    setup = build_layer(config)
    graph = setup.graph
    output_idxs = collect(setup.layer.output_layer)
    x = rand(rng, Float32, INPUT_DIM)
    y = fill(-1f0, length(output_idxs))
    y[3] = 1f0
    free_state = rand(rng, Float32, II.nstates(graph)) .* 2f0 .- 1f0

    II.graphstate(graph) .= free_state
    pattern = zeros(Float32, II.nstates(graph))
    install_tangent_projected_input_field!(
        graph,
        setup.input_hidden_w,
        x,
        y,
        free_state,
        pattern,
        output_idxs,
        config.β,
    )
    tangent_force = output_force(graph, output_idxs)
    expected_tangent_delta = config.β .* (y .- free_state[output_idxs])
    input_only_graph = setup.graph
    project_input_field_pattern!(pattern, setup.input_hidden_w, x)
    install_input_field_pattern!(input_only_graph, pattern)
    input_only_force = output_force(input_only_graph, output_idxs)
    tangent_delta = tangent_force .- input_only_force
    tangent_error = maximum(abs.(tangent_delta .- expected_tangent_delta))

    IsingLearning.apply_targets(graph, y)
    IsingLearning.set_clamping_beta!(graph, config.β)
    clamp_force = output_force(graph, output_idxs)
    IsingLearning.set_clamping_beta!(graph, 0f0)
    clamp_base_force = output_force(graph, output_idxs)
    expected_clamp_delta = .-config.β .* (free_state[output_idxs] .- y)
    clamp_delta = clamp_force .- clamp_base_force
    clamp_error = maximum(abs.(clamp_delta .- expected_clamp_delta))

    println("tangent_delta = ", tangent_delta)
    println("expected_tangent_delta = ", expected_tangent_delta)
    println("clamp_delta = ", clamp_delta)
    println("expected_clamp_delta = ", expected_clamp_delta)
    println("tangent_error = ", tangent_error)
    println("clamp_error = ", clamp_error)

    tangent_error < 1f-6 || error("tangent nudge direction mismatch")
    clamp_error < 1f-6 || error("clamp nudge direction mismatch")
    println("LCL nudging direction validation passed")
end

main()
