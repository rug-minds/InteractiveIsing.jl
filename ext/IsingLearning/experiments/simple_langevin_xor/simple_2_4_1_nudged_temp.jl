using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "simple_2_4_1_langevin.jl"))

"""
    nudged_temp(config)

Return the temperature used only during the plus/minus nudged phases.
The free phase and validation phase still use `config.temp`.
"""
function nudged_temp(config::SimpleXorConfig)
    factor = parse(FT, get(ENV, "ISING_SIMPLE_XOR_NUDGED_TEMP_FACTOR", "1.0"))
    return config.temp * factor
end

"""
    simple_nudged(layer, config)

Override the scalar experiment nudged phase so the clamped branches can run
hotter than the free branch. This tests whether a stable free attractor is too
deep for the target perturbation to escape at the base temperature.
"""
function simple_nudged(layer, config::SimpleXorConfig)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    plus_dynamics_algorithm = simple_dynamics(config)
    minus_dynamics_algorithm = simple_dynamics(config)
    Tn = nudged_temp(config)

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, beta)
        II.temp!(dynamics.model, Tn)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, config.temp)
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, -beta)
        II.temp!(dynamics.model, Tn)
        model = @repeat relaxation_steps dynamics()
        II.temp!(dynamics.model, config.temp)
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""
    run_nudged_temp_main()

Run the normal scalar `2 -> 4 -> 1` experiment with a nudged-only temperature
override supplied by `ISING_SIMPLE_XOR_NUDGED_TEMP_FACTOR`.
"""
function run_nudged_temp_main()
    config = SimpleXorConfig()
    outdir = get(
        ENV,
        "ISING_SIMPLE_XOR_DIR",
        joinpath(@__DIR__, "runs", "simple_2_4_1_nudged_temp_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(outdir)
    route_mode = Symbol(get(ENV, "ISING_SIMPLE_XOR_ROUTE", "normal"))
    all_rows = Dict{String,Any}[]
    results = []
    if route_mode in (:both, :normal)
        normal = run_route(config, "normal"; split = false)
        append!(all_rows, normal.rows)
        push!(results, (route = "normal", best_mse = normal.best_mse, best_acc = normal.best_acc))
    end
    if route_mode in (:both, :split)
        split = run_route(config, "split"; split = true)
        append!(all_rows, split.rows)
        push!(results, (route = "split", best_mse = split.best_mse, best_acc = split.best_acc))
    end
    route_mode in (:both, :normal, :split) || throw(ArgumentError("ISING_SIMPLE_XOR_ROUTE must be both, normal, or split"))
    csv_path = write_csv(joinpath(outdir, "simple_2_4_1_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "simple_2_4_1_progress.png"), all_rows)
    md_path = write_readme(joinpath(outdir, "README.md"), config, results, csv_path, png_path)
    println("Nudged temperature factor: ", get(ENV, "ISING_SIMPLE_XOR_NUDGED_TEMP_FACTOR", "1.0"))
    println("Saved metrics: ", csv_path)
    println("Saved plot: ", png_path)
    println("Saved docs: ", md_path)
    return (; outdir, results, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_nudged_temp_main()
end
