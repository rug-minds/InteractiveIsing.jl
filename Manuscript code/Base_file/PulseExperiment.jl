include(joinpath(@__DIR__, "Basefile.jl"))

cfg = config_from_environment()
model = build_physical_model(cfg)
g = model.graph
charges = model.defects
print_physical_summary(cfg, model)
reduced_summary = reduced_parameter_summary(cfg, model)

loggers = make_loggers(cfg)
point_repeat = loggers.sizes.point_repeat
pulse_time = loggers.sizes.experiment_time
relax_time = loggers.sizes.relax_time

dynamics = select_dynamics(cfg)
charge_dynamics = Metropolis()
pulse = TrianglePulse(
    internal_energy(cfg.pulse_amplitude, cfg),
    cfg.pulse_repeats,
)

pulse_monte_carlo = @CompositeAlgorithm begin
    @alias dynamics = dynamics
    @alias charge_metro = charge_dynamics
    @replace dynamics.T => charge_metro.T

    proposal = dynamics()
    @every cfg.defect_step_interval charge_metro()
    @every 1 loggers.polarization(
        Δvalue = @transform(accepted_proposal_delta, proposal),
    )
    @every point_repeat loggers.field(
        value = @transform(x -> x.b[], dynamics.hamiltonian),
    )
    @every point_repeat loggers.depol(
        model = dynamics.model,
        hamiltonian = dynamics.hamiltonian,
    )
    @every point_repeat loggers.staggered(
        value = @transform(staggered_z_polarization, dynamics.model),
    )
    @every point_repeat loggers.top(
        value = @transform(m -> mean_polarization_zlayer(m, cfg.nz), dynamics.model),
    )
    @every point_repeat loggers.middle(
        value = @transform(m -> mean_polarization_zlayer(m, cld(cfg.nz, 2)), dynamics.model),
    )
    @every point_repeat loggers.bottom(
        value = @transform(m -> mean_polarization_zlayer(m, 1), dynamics.model),
    )
end

pulse_step = @CompositeAlgorithm begin
    @context monte_carlo = pulse_monte_carlo()
    @every point_repeat pulse(
        hamiltonian = monte_carlo.dynamics.hamiltonian,
        M = monte_carlo.dynamics.M,
    )
end

relax_step = @CompositeAlgorithm begin
    @context monte_carlo = pulse_monte_carlo()
end

pulse_experiment = @Routine begin
    @repeat pulse_time pulse_step()
    @repeat relax_time relax_step()
end

createProcessManual(
    g,
    pulse_experiment,
    StatefulAlgorithms.Init(:dynamics; model = g),
    StatefulAlgorithms.Init(:charge_metro; model = charges),
    Init(loggers.polarization, initialvalue = sum(state(g)));
    lifetime = 1,
)

context = fetch(process(g))
result = collect_logged_result(context, loggers)
field_meV = result.field .* Unitful.ustrip(u"meV", cfg.energy_scale)
result = merge(result, (; field_meV))

figures = Dict{String,Any}(
    "field_polarization" => make_series_figure(
        field_meV,
        result.polarization;
        xlabel = "Field energy (meV)",
        ylabel = "Polarization",
        title = "Pulse scan",
    ),
    "step_polarization" => make_series_figure(
        eachindex(result.polarization),
        result.polarization;
        xlabel = "Logged step",
        ylabel = "Polarization",
        title = "Pulse scan",
    ),
)

for figure in values(figures)
    cfg.show_figures && display(figure)
end

saved = save_experiment("pulse", cfg, result, figures, reduced_summary)
isnothing(saved) || println("Saved pulse output: ", saved.xlsx_path)
