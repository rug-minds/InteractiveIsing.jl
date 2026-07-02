include(joinpath(@__DIR__, "Basefile.jl"))

const DEFAULT_ANNEAL_SWEEP_SCALES = Float32[1, 2, 3, 4, 6, 8, 10, 12]

function parse_anneal_sweep_scales()
    raw = get(ENV, "ISING_ANNEAL_SCALES", "")
    isempty(strip(raw)) && return DEFAULT_ANNEAL_SWEEP_SCALES

    values = Float32[]
    for item in split(raw, [',', ';', ' ']; keepempty = false)
        push!(values, parse(Float32, item))
    end
    isempty(values) && error("ISING_ANNEAL_SCALES did not contain any scale values.")
    return values
end

scale_label(scale) =
    replace(replace(string(round(Float64(scale); sigdigits = 4)), "." => "p"), "-" => "m")

function scaled_anneal_config(
    base_cfg::ExperimentConfig,
    scale::Real;
    start_kBT,
    end_kBT,
    initial_field,
    outdir,
)
    s = Float32(scale)
    return override_config(
        base_cfg;
        exchange_energy = base_cfg.exchange_energy * s,
        coulomb_dipole_scale = base_cfg.coulomb_dipole_scale * s,
        # initial_kBT = start_kBT * s,
        # anneal_max_kBT = start_kBT * s,
        # initial_field = initial_field * s,
        landau_a = base_cfg.landau_a * s,
        landau_b = base_cfg.landau_b * s,
        landau_c = base_cfg.landau_c * s,
        landau_d = base_cfg.landau_d * s,
        landau_e = base_cfg.landau_e * s,
        # landau_a_disorder_scale = base_cfg.landau_a_disorder_scale * s,
        # landau_b_disorder_scale = base_cfg.landau_b_disorder_scale * s,
        # landau_c_disorder_scale = base_cfg.landau_c_disorder_scale * s,
        # landau_d_disorder_scale = base_cfg.landau_d_disorder_scale * s,
        # landau_e_disorder_scale = base_cfg.landau_e_disorder_scale * s,
        linear_defect_strength = base_cfg.linear_defect_strength * s,
        linear_defect_disorder_scale = base_cfg.linear_defect_disorder_scale * s,
        # vacancy_quadratic_shift = base_cfg.vacancy_quadratic_shift * s,
        # vacancy_quartic_shift = base_cfg.vacancy_quartic_shift * s,
        show_interfaces = false,
        show_figures = false,
        outdir,
    )
end

function estimate_transition_temperature(result)
    length(result.temperature_K) < 2 && return missing

    best_index = 0
    best_slope = -Inf
    for i in 1:(length(result.temperature_K) - 1)
        dT = abs(result.temperature_K[i + 1] - result.temperature_K[i])
        dT == 0 && continue
        slope = abs(result.polarization[i + 1] - result.polarization[i]) / dT
        if slope > best_slope
            best_slope = slope
            best_index = i
        end
    end

    best_index == 0 && return missing
    return (result.temperature_K[best_index] + result.temperature_K[best_index + 1]) / 2
end

function finish_anneal_sweep_case!(name, scale, job)
    context = fetch(job.process)
    result = collect_logged_result(context, job.loggers; include_temperature = true)
    temperature_K = kBT_to_kelvin(result.temperature, job.cfg)
    result = merge(result, (; temperature_K))

    figures = Dict{String,Any}(
        "temperature_polarization" => make_series_figure(
            temperature_K,
            result.polarization;
            xlabel = "Temperature (K)",
            ylabel = "Polarization",
            title = "Anneal energy scale $(scale)",
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
            title = "Anneal energy scale $(scale) Hamiltonian Terms",
        ),
        "step_polarization" => make_series_figure(
            eachindex(result.polarization),
            result.polarization;
            xlabel = "Logged step",
            ylabel = "Polarization",
            title = "Anneal energy scale $(scale)",
        ),
    )

    extra_sheets = Pair{String,DataFrame}[
        "coefficients" => coefficient_dataframe(job.model),
    ]
    saved = save_experiment(
        name,
        job.cfg,
        result,
        figures,
        job.reduced_summary,
        extra_sheets,
    )

    return (; result, saved, transition_K = estimate_transition_temperature(result))
end

function run_anneal_energy_sweep(;
    scales = parse_anneal_sweep_scales(),
    start_kBT = 8.0f0u"meV",
    end_kBT = 0.0f0u"meV",
    initial_field = 0.5f0u"meV",
)
    base_cfg = config_from_environment()
    root_outdir = joinpath(base_cfg.outdir, "anneal_sweep")
    summary_rows = NamedTuple[]
    results = Any[]

    for scale in scales
        label = scale_label(scale)
        name = "anneal_sweep_scale_$(label)"
        outdir = joinpath(root_outdir, name)
        cfg = scaled_anneal_config(
            base_cfg,
            scale;
            start_kBT,
            end_kBT,
            initial_field,
            outdir,
        )

        println("Starting anneal scale ", scale)
        println("  start kBT: ", start_kBT * scale, " (", only(kBT_to_kelvin([internal_energy(start_kBT * scale, cfg)], cfg)), " K)")
        println("  initial field coefficient: ", initial_field * scale)

        job = build_anneal_job(cfg; start_kBT = start_kBT * scale, end_kBT)
        finished = finish_anneal_sweep_case!(name, scale, job)
        push!(results, (; scale, cfg, finished...))
        push!(
            summary_rows,
            (;
                scale = Float64(scale),
                start_kBT = string(start_kBT * scale),
                start_temperature_K = only(kBT_to_kelvin([internal_energy(start_kBT * scale, cfg)], cfg)),
                transition_K = finished.transition_K,
                exchange_energy = string(cfg.exchange_energy),
                initial_field = string(cfg.initial_field),
                landau_a = cfg.landau_a,
                landau_b = cfg.landau_b,
                landau_c = cfg.landau_c,
                coulomb_dipole_scale = string(cfg.coulomb_dipole_scale),
                saved_xlsx = isnothing(finished.saved) ? "" : finished.saved.xlsx_path,
            ),
        )
    end

    summary = DataFrame(summary_rows)
    summary_path = nothing
    if base_cfg.save_outputs && base_cfg.save_excel
        mkpath(root_outdir)
        summary_path = joinpath(root_outdir, "anneal_sweep_summary.xlsx")
        XLSX.openxlsx(summary_path, mode = "w") do workbook
            workbook[1].name = "summary"
            write_dataframe_sheet!(workbook["summary"], summary)
        end
        println("Saved anneal sweep summary: ", summary_path)
    else
        println("Skipping anneal sweep summary save because output or Excel saving is disabled.")
    end

    return (; summary, results, summary_path)
end

sweep = Base.invokelatest(run_anneal_energy_sweep);
println("Finished ", length(sweep.results), " anneal sweep cases.")
