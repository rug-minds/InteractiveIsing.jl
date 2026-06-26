include(joinpath(@__DIR__, "Basefile.jl"))

cfg = config_from_environment()
anneal_start_kBT = 2.0f0u"meV"
anneal_end_kBT = 0.0f0u"meV"
anneal_initial_field = 0.5f0u"meV"

cfg = override_config(
    cfg;
    initial_kBT = anneal_start_kBT,
    anneal_max_kBT = anneal_start_kBT,
    initial_field = anneal_initial_field,
    time_factor = 10,
)

model = build_physical_model(cfg)
graph = model.graph
charges = model.defects
print_physical_summary(cfg, model)
reduced_summary = reduced_parameter_summary(cfg, model)

loggers = make_loggers(cfg)
point_repeat = loggers.sizes.point_repeat
anneal_time = loggers.sizes.experiment_time
top_mean = LayerMean{cfg.nz}()
mid_mean = LayerMean{cld(cfg.nz, 2)}()
bottom_mean = LayerMean{1}()

dynamics = select_dynamics(cfg)
charge_dynamics = Metropolis()
temperature_cycle = TemperatureCycle(
    internal_energy(anneal_start_kBT, cfg),
    internal_energy(anneal_end_kBT, cfg),
)

anneal_monte_carlo = @CompositeAlgorithm begin
    @alias dynamics = dynamics
    @alias charge_metro = charge_dynamics
    @alias anneal_temperature = temperature_cycle
    @replace anneal_temperature.temperature => dynamics.T
    @replace dynamics.T => charge_metro.T

    @every point_repeat anneal_temperature(model = dynamics.model)
    proposal = dynamics()
    @every cfg.defect_step_interval charge_metro()
    @every 1 loggers.polarization(
        value = @transform(accepted_proposal_delta, proposal),
    )
    @every point_repeat loggers.field(
        value = @transform(field_value, dynamics.hamiltonian),
    )
    @every point_repeat loggers.temperature(
        value = @transform(temp, dynamics.model),
    )
    @every point_repeat loggers.depol(
        model = dynamics.model,
        hamiltonian = dynamics.hamiltonian,
    )
    @every point_repeat loggers.staggered(
        value = @transform(staggered_z_polarization, dynamics.model),
    )
    @every point_repeat loggers.top(
        value = @transform(top_mean, dynamics.model),
    )
    @every point_repeat loggers.middle(
        value = @transform(mid_mean, dynamics.model),
    )
    @every point_repeat loggers.bottom(
        value = @transform(bottom_mean, dynamics.model),
    )
end

anneal_step = @CompositeAlgorithm begin
    @context monte_carlo = anneal_monte_carlo()
end

anneal_experiment = @Routine begin
    @repeat anneal_time anneal_step()
end

process = Process(
    anneal_experiment,
    StatefulAlgorithms.Init(:dynamics; model = graph),
    StatefulAlgorithms.Init(:charge_metro; model = charges),
    Init(loggers.polarization, initialvalue = sum(state(graph)));
    repeats = 1,
)
run(process)

context = fetch(process)
result = collect_logged_result(context, loggers; include_temperature = true)
temperature_K = kBT_to_kelvin(result.temperature, cfg)
result = merge(result, (; temperature_K))

figures = Dict{String,Any}(
    "temperature_polarization" => make_series_figure(
        temperature_K,
        result.polarization;
        xlabel = "Temperature (K)",
        ylabel = "Polarization",
        title = "Annealing",
    ),
    "step_polarization" => make_series_figure(
        eachindex(result.polarization),
        result.polarization;
        xlabel = "Logged step",
        ylabel = "Polarization",
        title = "Annealing",
    ),
    "step_total_energy" => make_series_figure(
        eachindex(result.total_energy),
        result.total_energy;
        xlabel = "Logged step",
        ylabel = "Total H",
        title = "Annealing Hamiltonian",
    ),
    "step_hamiltonian_terms" => make_multi_series_figure(
        eachindex(result.total_energy),
        [
            "H_J" => result.interaction_energy,
            "H_field" => result.field_energy,
            "H_poly" => result.polynomial_energy,
            "H_dep" => result.coulomb_energy,
        ];
        xlabel = "Logged step",
        ylabel = "Hamiltonian terms",
        title = "Annealing Hamiltonian Terms",
    ),
    "temperature_hamiltonian_terms" => make_multi_series_figure(
        temperature_K,
        [
            "H_J" => result.interaction_energy,
            "H_field" => result.field_energy,
            "H_poly" => result.polynomial_energy,
            "H_dep" => result.coulomb_energy,
        ];
        xlabel = "Temperature (K)",
        ylabel = "Hamiltonian terms",
        title = "Annealing Hamiltonian Terms",
    ),
    "step_depolarization_diagnostic" => make_multi_series_figure(
        eachindex(result.depol_mean),
        [
            "mean" => result.depol_mean,
            "median" => result.depol_median,
            "max" => result.depol_max,
        ];
        xlabel = "Logged step",
        ylabel = "Depolarization diagnostic",
        title = "Annealing Depolarization Diagnostic",
    ),
)

for figure in values(figures)
    cfg.show_figures && display(figure)
end

extra_sheets = Pair{String,DataFrame}[
    "coefficients" => coefficient_dataframe(model),
]

saved = save_experiment("anneal", cfg, result, figures, reduced_summary, extra_sheets)
isnothing(saved) || println("Saved anneal output: ", saved.xlsx_path)
