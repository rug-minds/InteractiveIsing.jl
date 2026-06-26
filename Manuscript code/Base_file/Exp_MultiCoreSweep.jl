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



# Base.@kwdef struct ExperimentConfig
#     nx::Int = 10
#     ny::Int = 10
#     nz::Int = 10

#     energy_scale = 1.0f0u"meV"
#     length_scale = 1.0f0u"nm"
#     elementary_charge = 1.602176634f-19u"C"

#     lattice_x = 1.0f0u"nm"
#     lattice_y = 1.0f0u"nm"
#     lattice_z = 1.0f0u"nm"

#     exchange_energy = 1.0f0u"meV"
#     coulomb_dipole_scale = elementary_charge * 0f0u"nm"
#     coulomb_screening = 0.0001f0u"nm"
#     coulomb_recalc_interval::Int = 1000

#     # Constructor fallbacks. Experiment scripts should override these explicitly.
#     initial_kBT = 0.15f0u"meV"
#     anneal_max_kBT = 10.0f0u"meV"
#     initial_field = 0f0u"meV"
#     pulse_amplitude = 10.0f0u"meV"

#     landau_a::Float32 = -0.3f0
#     landau_b::Float32 = -2.1f0
#     landau_c::Float32 = 1.5f0
#     landau_d::Float32 = 0.0f0
#     landau_e::Float32 = 0.0f0
#     include_landau_8::Bool = false
#     include_landau_10::Bool = false

#     # Add site-to-site Gaussian disorder to Landau coefficients:
#     # coeff_i[site] = landau_i + landau_i_disorder_scale * randn().
#     # These are additive reduced coefficients, not multiplicative scale factors.
#     apply_landau_disorder::Bool = false
#     landau_disorder_seed::Int = 43
#     landau_a_disorder_scale::Float32 = 0.0f0
#     landau_b_disorder_scale::Float32 = 0.0f0
#     landau_c_disorder_scale::Float32 = 0.0f0
#     landau_d_disorder_scale::Float32 = 0.0f0
#     landau_e_disorder_scale::Float32 = 0.0f0

#     linear_defect_count::Int = 0
#     linear_defect_strength = 0.0f0u"meV"
#     linear_defect_disorder_scale = 0.0f0u"meV"
#     # If true, each built-in-field defect randomly chooses + or - sign.
#     linear_defect_random_sign::Bool = false
#     # RNG seed makes the random defect locations/signs/disorder reproducible.
#     linear_defect_rng_seed::Int = 44

#     vacancy_count::Int = 10
#     electron_count::Int = 20
#     vacancy_charge_number::Float32 = 2.0f0
#     electron_charge_number::Float32 = 1.0f0
#     # Larger values make mobile electrons attempt hopping more often.
#     electron_attempt_rate::Float32 = 10.0f0
#     # Mobile vacancies/electrons are updated once every this many dynamics steps.
#     # Smaller values mean more frequent defect motion.
#     defect_step_interval::Int = 1000
#     free_charge_split::Float32 = 0.5f0
#     vacancy_quadratic_shift::Float32 = 0.012f0
#     vacancy_quartic_shift::Float32 = 0.004f0
#     defect_rng_seed::Int = 42

#     steps::Int = 4000
#     time_factor::Int = 1
#     pulse_repeats::Int = 3
#     relax_fraction::Float64 = 0.5

#     algorithm_name::Symbol = :local_langevin
#     langevin_stepsize::Float32 = 0.02f0
#     langevin_adjusted::Bool = true
#     proposer_delta::Float32 = 0.1f0

#     # Weight function options for the Ising J term.
#     # Simple path: set weight_mode and shell weights below, e.g.
#     #   weight_mode = :nearest
#     #   weight_mode = :shell
#     #   weight_mode = :layered_afe
#     #   weight_mode = :none
#     # Custom path: pass weight_function = (cfg; dc) -> ... .
#     # The custom function receives dc = (dx, dy, dz) and must return an energy
#     # with units, for example: 0.2f0 * cfg.exchange_energy.
#     # If weight_function is not nothing, it overrides weight_mode.
#     weight_function = nothing
#     weight_mode::Symbol = :shell
#     # Maximum neighbor offset shell passed to WeightGenerator.
#     weight_range::Int = 3
#     # Dimensionless factors multiplied by exchange_energy in :shell mode.
#     shell_weight_1::Float32 = 1.0f0
#     shell_weight_2::Float32 = 0.1f0
#     shell_weight_3::Float32 = 0.01f0
#     shell_weight_beyond::Float32 = 0.0f0

#     show_interfaces::Bool = true
#     show_figures::Bool = false
#     save_outputs::Bool = true
#     save_figures::Bool = true
#     save_excel::Bool = true
#     outdir::String = raw"D:\Code\data\20260623"
# end
# See ExperimentConfig in Basefile.jl for all override keys.
