include(joinpath(@__DIR__, "Basefile.jl"))

################################################################################
# Experiment setup
################################################################################

cfg = config_from_environment()

forc_kBT = 0.15f0u"meV"
forc_initial_field = 0.0f0u"meV"
forc_max_field = 6.0f0u"meV"
forc_preset_edge_points = 1001
forc_test_field_step = 2.0f0u"meV"
## 每个forc正向脉冲后保持的点，不能太大哦
forc_zero_hold_points = 201

cfg = override_config(
    cfg;
    initial_kBT = forc_kBT,
    initial_field = forc_initial_field,
)

model = build_physical_model(cfg)
g = model.graph
charges = model.defects
print_physical_summary(cfg, model)
reduced_summary = reduced_parameter_summary(cfg, model)

loggers = make_loggers(cfg)
point_repeat = loggers.sizes.point_repeat

forc_pulse = FORCPulse(
    internal_energy(forc_max_field, cfg),
    forc_preset_edge_points,
    internal_energy(forc_test_field_step, cfg),
    forc_zero_hold_points,
)
protocol = forc_protocol(forc_pulse)
forc_time = point_repeat * length(protocol.fields)
relax_time = loggers.sizes.relax_time

println("FORC field samples: ", length(protocol.fields))
println("FORC curves: ", length(protocol.end_fields))
println("FORC edge step: ", protocol.edge_step)
println("FORC test field step: ", protocol.test_field_step)
println("FORC zero-hold samples per curve: ", protocol.zero_hold_points)

dynamics = select_dynamics(cfg)
charge_dynamics = Metropolis()

forc_monte_carlo = @CompositeAlgorithm begin
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

forc_step = @CompositeAlgorithm begin
    @context monte_carlo = forc_monte_carlo()
    @every point_repeat forc_pulse(
        hamiltonian = monte_carlo.dynamics.hamiltonian,
    )
end

relax_step = @CompositeAlgorithm begin
    @context monte_carlo = forc_monte_carlo()
end

forc_experiment = @Routine begin
    @repeat forc_time forc_step()
    @repeat relax_time relax_step()
end

createProcessManual(
    g,
    forc_experiment,
    StatefulAlgorithms.Init(:dynamics; model = g),
    StatefulAlgorithms.Init(:charge_metro; model = charges),
    Init(loggers.polarization, initialvalue = sum(state(g)));
    lifetime = 1,
)

context = fetch(process(g))
result = collect_logged_result(context, loggers)
field_meV = result.field .* Unitful.ustrip(u"meV", cfg.energy_scale)
label_count = length(result.field)
forc_curve = zeros(Int, label_count)
forc_end_field = zeros(Float32, label_count)
protocol_count = min(label_count, length(protocol.curve))
forc_curve[1:protocol_count] .= protocol.curve[1:protocol_count]
forc_end_field[1:protocol_count] .= Float32.(protocol.curve_end_field[1:protocol_count])
result = merge(
    result,
    (;
        field_meV,
        forc_curve,
        forc_end_field,
    ),
)

figures = Dict{String,Any}(
    "field_polarization" => make_series_figure(
        field_meV,
        result.polarization;
        xlabel = "Field energy (meV)",
        ylabel = "Polarization",
        title = "FORC pulse",
    ),
    "step_field" => make_series_figure(
        eachindex(result.field),
        field_meV;
        xlabel = "Logged step",
        ylabel = "Field energy (meV)",
        title = "FORC field protocol",
    ),
    "step_polarization" => make_series_figure(
        eachindex(result.polarization),
        result.polarization;
        xlabel = "Logged step",
        ylabel = "Polarization",
        title = "FORC pulse",
    ),
)

for figure in values(figures)
    cfg.show_figures && display(figure)
end

extra_sheets = Pair{String,DataFrame}[
    "forc_protocol" => forc_protocol_dataframe(protocol, cfg),
]

saved = save_experiment("Forc_pulse", cfg, result, figures, reduced_summary, extra_sheets)
isnothing(saved) || println("Saved FORC pulse output: ", saved.xlsx_path)
