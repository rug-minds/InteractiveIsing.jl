include(joinpath(@__DIR__, "Basefile.jl"))

Base.@kwdef struct SweepCase
    name::String
    kind::Symbol = :pulse
    overrides::NamedTuple = (;)
    outdir::Union{Nothing,String} = nothing
end

function experiment_overrides(overrides::NamedTuple)
    allowed = fieldnames(ExperimentConfig)
    return (; (name => value for (name, value) in pairs(overrides) if name in allowed)...)
end

function cfg_for_case(base_cfg::ExperimentConfig, case::SweepCase)
    cfg = override_config(base_cfg; experiment_overrides(case.overrides)...)
    isnothing(case.outdir) || (cfg = override_config(cfg; outdir = case.outdir))
    return cfg
end

function finish_pulse_job!(case::SweepCase, job)
    context = fetch(job.process)
    result = collect_logged_result(context, job.loggers)
    field_meV = result.field .* Unitful.ustrip(u"meV", job.cfg.energy_scale)
    result = merge(result, (; field_meV))

    figures = Dict{String,Any}(
        "field_polarization" => make_series_figure(
            field_meV,
            result.polarization;
            xlabel = "Field energy (meV)",
            ylabel = "Polarization",
            title = case.name,
        ),
        "step_polarization" => make_series_figure(
            eachindex(result.polarization),
            result.polarization;
            xlabel = "Logged step",
            ylabel = "Polarization",
            title = case.name,
        ),
    )

    extra_sheets = Pair{String,DataFrame}[
        "coefficients" => coefficient_dataframe(job.model),
    ]

    saved = save_experiment(
        "sweep_$(case.name)",
        job.cfg,
        result,
        figures,
        job.reduced_summary,
        extra_sheets,
    )
    return (; case, result, saved)
end

function finish_anneal_job!(case::SweepCase, job)
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
            title = case.name,
        ),
        "step_polarization" => make_series_figure(
            eachindex(result.polarization),
            result.polarization;
            xlabel = "Logged step",
            ylabel = "Polarization",
            title = case.name,
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
            title = "$(case.name) Hamiltonian Terms",
        ),
    )

    extra_sheets = Pair{String,DataFrame}[
        "coefficients" => coefficient_dataframe(job.model),
    ]

    saved = save_experiment(
        "sweep_$(case.name)",
        job.cfg,
        result,
        figures,
        job.reduced_summary,
        extra_sheets,
    )
    return (; case, result, saved)
end

function start_sweep_batch(base_cfg::ExperimentConfig, cases)
    return map(cases) do case
        cfg = cfg_for_case(base_cfg, case)
        println("Starting ", case.kind, " case: ", case.name)
        job = case.kind == :pulse ? build_pulse_job(cfg) :
              case.kind == :anneal ? build_anneal_job(
                  cfg;
                  start_kBT = get(case.overrides, :anneal_start_kBT, cfg.anneal_max_kBT),
                  end_kBT = get(case.overrides, :anneal_end_kBT, 0.0f0u"meV"),
              ) :
              error("Unknown sweep case kind $(case.kind). Use :pulse or :anneal.")
        return (; case, job)
    end
end

function finish_sweep_batch!(running)
    return map(running) do item
        println("Fetching case: ", item.case.name)
        item.case.kind == :pulse ?
            finish_pulse_job!(item.case, item.job) :
            finish_anneal_job!(item.case, item.job)
    end
end

function run_sweep_batched(cases; max_inflight = Threads.nthreads())
    base_cfg = override_config(
        config_from_environment();
        show_interfaces = false,
        show_figures = false,
    )

    finished = Any[]
    for first in 1:max_inflight:length(cases)
        last = min(first + max_inflight - 1, length(cases))
        running = start_sweep_batch(base_cfg, cases[first:last])
        append!(finished, finish_sweep_batch!(running))
    end
    return finished
end

mutable struct SweepManagerState
    cases::Vector{Any}
    jobs::Vector{Any}
    results::Vector{Any}
end

SweepManagerState(n::Integer) =
    SweepManagerState(fill(nothing, n), fill(nothing, n), fill(nothing, n))

function indexed_sweep_jobs(base_cfg::ExperimentConfig, cases)
    return [
        (; index = i, case, base_cfg)
        for (i, case) in pairs(cases)
    ]
end

function make_sweep_worker(idx::Integer, manager::ProcessManager, job)
    case = job.case
    cfg = cfg_for_case(job.base_cfg, case)
    println("Starting ", case.kind, " case: ", case.name)

    built_job = case.kind == :pulse ? build_pulse_job(cfg; start = false) :
                case.kind == :anneal ? build_anneal_job(
                    cfg;
                    start_kBT = get(case.overrides, :anneal_start_kBT, cfg.anneal_max_kBT),
                    end_kBT = get(case.overrides, :anneal_end_kBT, 0.0f0u"meV"),
                    start = false,
                ) :
                error("Unknown sweep case kind $(case.kind). Use :pulse or :anneal.")

    manager.state.cases[job.index] = case
    manager.state.jobs[job.index] = built_job
    return built_job.process
end

sweep_worker_name(idx::Integer, manager::ProcessManager, job) =
    Symbol("sweep_", job.case.name)

function build_sweep_manager(jobs; nworkers = min(length(jobs), max(1, Threads.nthreads())))
    recipe = (;
        makeworker = make_sweep_worker,
        workername = sweep_worker_name,
    )

    return ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        state = SweepManagerState(length(jobs)),
        worker_lifecycle = OnDemandWorkers(destroy_after_finalize = false),
        worker_type = Process,
        sync_policy = NoSync(),
        execution = ThreadedWorkers(Dynamic()),
        job_type = eltype(jobs),
        result_type = Any,
    )
end

function finish_sweep_manager!(manager::ProcessManager)
    results = Any[]
    for i in eachindex(manager.state.cases)
        case = manager.state.cases[i]
        job = manager.state.jobs[i]
        isnothing(case) && continue
        isnothing(job) && continue

        println("Saving case: ", case.name)
        result = case.kind == :pulse ?
            finish_pulse_job!(case, job) :
            finish_anneal_job!(case, job)
        manager.state.results[i] = result
        push!(results, result)
    end
    return results
end

function run_sweep_manager(cases; nworkers = min(length(cases), max(1, Threads.nthreads())))
    base_cfg = override_config(
        config_from_environment();
        show_interfaces = false,
        show_figures = false,
    )

    jobs = indexed_sweep_jobs(base_cfg, cases)
    manager = build_sweep_manager(jobs; nworkers)
    run!(manager, jobs)
    results = finish_sweep_manager!(manager)
    return (; manager, results)
end

function run_sweep(cases; max_inflight = min(length(cases), max(1, Threads.nthreads())))
    sweep = Base.invokelatest(run_sweep_manager, cases; nworkers = max_inflight)
    return sweep.results
end

cases = [
    SweepCase(
        name = "pulse_J_0p5",
        kind = :pulse,
        overrides = (;
            exchange_energy = 0.5f0u"meV",
            initial_kBT = 0.15f0u"meV",
            initial_field = 0.0f0u"meV",
            pulse_amplitude = 10.0f0u"meV",
            steps = 1000,
        ),
        outdir = raw"D:\Code\data\20260623\sweep\pulse_J_0p5",
    ),
    SweepCase(
        name = "pulse_J_1p0",
        kind = :pulse,
        overrides = (;
            exchange_energy = 1.0f0u"meV",
            initial_kBT = 0.15f0u"meV",
            initial_field = 0.0f0u"meV",
            pulse_amplitude = 10.0f0u"meV",
            steps = 1000,
        ),
        outdir = raw"D:\Code\data\20260623\sweep\pulse_J_1p0",
    ),
    SweepCase(
        name = "anneal_T_2meV",
        kind = :anneal,
        overrides = (;
            anneal_start_kBT = 6.0f0u"meV",
            anneal_end_kBT = 0.0f0u"meV",
            initial_field = 0.0f0u"meV",
            steps = 1000,
        ),
        outdir = raw"D:\Code\data\20260623\sweep\anneal_T_2meV",
    ),
]

function main(; max_inflight = min(Threads.nthreads(), length(cases)))
    println("Julia threads: ", Threads.nthreads())
    return run_sweep(cases; max_inflight)
end

results = Base.invokelatest(main);
println("Finished ", length(results), " sweep cases.")



# ------------------------------------------------------------------------------
# Sweep override cheat sheet
# ------------------------------------------------------------------------------
# Copy any of these keys into a SweepCase `overrides = (; ... )`.
# The defaults live in Basefile.jl's ExperimentConfig.
#
# overrides = (;
#     # System size
#     nx = 10,
#     ny = 10,
#     nz = 10,
#
#     # Physical unit scales
#     energy_scale = 1.0f0u"meV",
#     length_scale = 1.0f0u"nm",
#     elementary_charge = 1.602176634f-19u"C",
#
#     # Lattice constants
#     lattice_x = 1.0f0u"nm",
#     lattice_y = 1.0f0u"nm",
#     lattice_z = 1.0f0u"nm",
#
#     # Interactions
#     exchange_energy = 1.0f0u"meV",
#     coulomb_dipole_scale = elementary_charge * 0f0u"nm",
#     coulomb_screening = 0.0001f0u"nm",
#     coulomb_recalc_interval = 1000,
#
#     # Initial condition / drive
#     initial_kBT = 0.15f0u"meV",
#     anneal_max_kBT = 10.0f0u"meV",
#     initial_field = 0f0u"meV",
#     pulse_amplitude = 10.0f0u"meV",
#
#     # Sweep-only anneal route keys. These are read by Exp_MultiCoreSweep.jl,
#     # not stored in ExperimentConfig.
#     anneal_start_kBT = 2.0f0u"meV",
#     anneal_end_kBT = 0.0f0u"meV",
#
#     # Landau polynomial coefficients
#     landau_a = -0.3f0,
#     landau_b = -2.1f0,
#     landau_c = 1.5f0,
#     landau_d = 0.0f0,
#     landau_e = 0.0f0,
#     include_landau_8 = false,
#     include_landau_10 = false,
#
#     # Additive Gaussian Landau disorder:
#     # coeff_i[site] = landau_i + landau_i_disorder_scale * randn().
#     apply_landau_disorder = false,
#     landau_disorder_seed = 43,
#     landau_a_disorder_scale = 0.0f0,
#     landau_b_disorder_scale = 0.0f0,
#     landau_c_disorder_scale = 0.0f0,
#     landau_d_disorder_scale = 0.0f0,
#     landau_e_disorder_scale = 0.0f0,
#
#     # Static built-in-field defects, implemented as a local first-order term.
#     linear_defect_count = 0,
#     linear_defect_strength = 0.0f0u"meV",
#     linear_defect_disorder_scale = 0.0f0u"meV",
#     linear_defect_random_sign = false,
#     linear_defect_rng_seed = 44,
#
#     # Mobile vacancies / electrons
#     vacancy_count = 10,
#     electron_count = 20,
#     vacancy_charge_number = 2.0f0,
#     electron_charge_number = 1.0f0,
#     electron_attempt_rate = 10.0f0,
#     defect_step_interval = 1000,
#     free_charge_split = 0.5f0,
#     vacancy_quadratic_shift = 0.012f0,
#     vacancy_quartic_shift = 0.004f0,
#     defect_rng_seed = 42,
#
#     # Runtime
#     steps = 4000,
#     time_factor = 1,
#     pulse_repeats = 3,
#     relax_fraction = 0.5,
#
#     # Dynamics
#     algorithm_name = :local_langevin,  # or :metropolis
#     langevin_stepsize = 0.02f0,
#     langevin_adjusted = true,
#     proposer_delta = 0.1f0,
#
#     # Weight function options for the Ising J term.
#     # Built-in modes: :shell, :nearest, :xy_nearest, :z_nearest,
#     # :layered_afe, :none.
#     weight_mode = :shell,
#     weight_range = 3,
#     shell_weight_1 = 1.0f0,
#     shell_weight_2 = 0.1f0,
#     shell_weight_3 = 0.01f0,
#     shell_weight_beyond = 0.0f0,
#
#     # For custom weights, define a function above cases, for example:
#     # my_weight(cfg; dc) = (dc == (1, 0, 0) ? 1.0f0 : 0.0f0) * cfg.exchange_energy
#     # weight_function = my_weight,
#
#     # Output. Sweep also has SweepCase(..., outdir = "...") per case.
#     show_interfaces = false,
#     show_figures = false,
#     save_outputs = true,
#     save_figures = true,
#     save_excel = true,
#     outdir = raw"D:\Code\data\20260623",
# )
