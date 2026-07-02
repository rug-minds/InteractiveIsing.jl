include(joinpath(@__DIR__, "Basefile.jl"))

################################################################################
# Experiment setup
################################################################################

cfg = config_from_environment()

forc_kBT = 0.15f0u"meV"
forc_initial_field = 0.0f0u"meV"
forc_max_field = 3.0f0u"meV"
forc_preset_edge_points = 1001
forc_test_field_step = 0.5f0u"meV"
# Hold points after each positive FORC branch. Keep this modest for quick tests.
forc_zero_hold_points = 101

cfg = override_config(
    cfg;
    initial_kBT = forc_kBT,
    initial_field = forc_initial_field,
    # Landau polynomial coefficients
    landau_a = -0.3f0,
    landau_b = -2.1f0,
    landau_c = 1.5f0,
    landau_d = 0.1f0,
    landau_e = 0.1f0,
    include_landau_8 = true,
    include_landau_10 = true,

    # Additive Gaussian Landau disorder:
    # coeff_i[site] = landau_i + landau_i_disorder_scale * randn().
    apply_landau_disorder = false,
    landau_disorder_seed = 43,
    landau_a_disorder_scale = 1.0f0,
    landau_b_disorder_scale = 1.0f0,
    landau_c_disorder_scale = 1.5f0,
    landau_d_disorder_scale = 1.5f0,
    landau_e_disorder_scale = 1.5f0,
    
)

job = build_forc_job(
    cfg;
    max_field = forc_max_field,
    preset_edge_points = forc_preset_edge_points,
    test_field_step = forc_test_field_step,
    zero_hold_points = forc_zero_hold_points,
)
print_physical_summary(job.cfg, job.model)

context = fetch(job.process)
result = collect_logged_result(context, job.loggers)
field_meV = result.field .* Unitful.ustrip(u"meV", job.cfg.energy_scale)
label_count = length(result.field)
forc_curve = zeros(Int, label_count)
forc_end_field = zeros(Float32, label_count)
protocol_count = min(label_count, length(job.protocol.curve))
forc_curve[1:protocol_count] .= job.protocol.curve[1:protocol_count]
forc_end_field[1:protocol_count] .= Float32.(job.protocol.curve_end_field[1:protocol_count])
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
    job.cfg.show_figures && display(figure)
end

extra_sheets = Pair{String,DataFrame}[
    "forc_protocol" => forc_protocol_dataframe(job.protocol, job.cfg),
    "coefficients" => coefficient_dataframe(job.model),
]

saved = save_experiment(
    "Forc_pulse",
    job.cfg,
    result,
    figures,
    job.reduced_summary,
    extra_sheets,
)
isnothing(saved) || println("Saved FORC pulse output: ", saved.xlsx_path)
