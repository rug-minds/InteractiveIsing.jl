using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

include(joinpath(@__DIR__, "..", "mnist_lcl_6x6_window_adam.jl"))

"""Assert that two scalar values agree up to Float32 roundoff."""
function assert_close(name::AbstractString, got::T, expected::S) where {T<:Real,S<:Real}
    err = abs(Float64(got) - Float64(expected))
    err <= 1e-6 || error("$(name) mismatch: got $(got), expected $(expected)")
    return err
end

"""Return the internal dynamics temperature from one resolved LCL training worker."""
function worker_dynamics_temperatures(worker::W) where {W}
    ctx = StatefulAlgorithms.context(worker)
    return (;
        dynamics_T = ctx.dynamics.T,
    )
end

"""Validate static LCL temperature, stepsize, beta, and worker parameter plumbing."""
function main()
    config = updated_config(
        InputFieldMNISTConfig();
        workers = 2,
        epochs = 0,
        batchsize = 2,
        train_per_class = 1,
        test_per_class = 1,
        train_eval_per_class = 0,
        hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2,
        output_replicas = 1,
        sweeps = 2f0,
        β = 0.3f0,
        lr = 1f-3,
        temp = 0.0123f0,
        stepsize = 0.25f0,
        weight_scale = 0.005f0,
        outdir = joinpath(@__DIR__, "tmp_validate_lcl_temperature_and_parameters"),
    )

    setup = build_layer(config)
    input_hidden_w = Ref(copy(setup.input_hidden_w))
    manager = input_field_manager(setup.layer, setup.graph, config, input_hidden_w)

    try
        println("config.temp = ", config.temp)
        println("config.stepsize = ", config.stepsize)
        println("config.beta = ", config.β)
        println("relaxation_steps = ", setup.relaxation_steps)
        println("active_units = ", active_units(setup.graph))

        assert_close("source graph temp", setup.graph.temp, config.temp)
        setup.layer.β == config.β || error("layer beta mismatch")
        setup.layer.free_relaxation_steps == setup.relaxation_steps || error("free relaxation step mismatch")
        setup.layer.nudged_relaxation_steps == setup.relaxation_steps || error("nudged relaxation step mismatch")
        setup.layer.dynamics_algorithm.stepsize == config.stepsize || error("free dynamics stepsize mismatch")
        setup.layer.nudged_dynamics_algorithm.stepsize == config.stepsize || error("nudged dynamics stepsize mismatch")

        for (idx, worker) in enumerate(StatefulAlgorithms.workers(manager))
            graph = worker_graph(worker)
            temps = worker_dynamics_temperatures(worker)
            assert_close("worker $(idx) graph temp", graph.temp, config.temp)
            assert_close("worker $(idx) dynamics T", temps.dynamics_T, config.temp)
            worker_context(worker).input_hidden_w === manager.state.input_hidden_w ||
                error("worker $(idx) does not hold the manager input_hidden_w Ref")
        end

        params = manager.state.params[]
        manager.state.input_hidden_w[] === input_hidden_w[] ||
            error("manager input_hidden_w Ref is not the supplied matrix before update")
        params.w_input == input_hidden_w[] ||
            error("optimizer params and input-hidden Ref values diverge before training")

        println("LCL temperature and parameter plumbing validation passed")
    finally
        close(manager)
        rm(config.outdir; force = true, recursive = true)
    end
end

main()
