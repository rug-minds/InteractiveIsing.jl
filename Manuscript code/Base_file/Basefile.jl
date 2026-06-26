using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
import InteractiveIsing as II

using CairoMakie
using DataFrames
using Dates
using GLMakie
using Random
using Statistics
using Unitful
using XLSX

include(joinpath(@__DIR__, "ExperimentLoggers.jl"))

Base.@kwdef struct ExperimentConfig
    nx::Int = 10
    ny::Int = 10
    nz::Int = 10

    energy_scale = 1.0f0u"meV"
    length_scale = 1.0f0u"nm"
    elementary_charge = 1.602176634f-19u"C"

    lattice_x = 1.0f0u"nm"
    lattice_y = 1.0f0u"nm"
    lattice_z = 1.0f0u"nm"

    exchange_energy = 1.0f0u"meV"
    coulomb_dipole_scale = elementary_charge * 0f0u"nm"
    coulomb_screening = 0.0001f0u"nm"
    coulomb_recalc_interval::Int = 1000

    # Constructor fallbacks. Experiment scripts should override these explicitly.
    initial_kBT = 0.15f0u"meV"
    anneal_max_kBT = 10.0f0u"meV"
    initial_field = 0f0u"meV"
    pulse_amplitude = 10.0f0u"meV"

    landau_a::Float32 = -0.3f0
    landau_b::Float32 = -2.1f0
    landau_c::Float32 = 1.5f0
    landau_d::Float32 = 0.0f0
    landau_e::Float32 = 0.0f0
    include_landau_8::Bool = false
    include_landau_10::Bool = false

    # Add site-to-site Gaussian disorder to Landau coefficients:
    # coeff_i[site] = landau_i + landau_i_disorder_scale * randn().
    # These are additive reduced coefficients, not multiplicative scale factors.
    apply_landau_disorder::Bool = false
    landau_disorder_seed::Int = 43
    landau_a_disorder_scale::Float32 = 0.0f0
    landau_b_disorder_scale::Float32 = 0.0f0
    landau_c_disorder_scale::Float32 = 0.0f0
    landau_d_disorder_scale::Float32 = 0.0f0
    landau_e_disorder_scale::Float32 = 0.0f0

    linear_defect_count::Int = 0
    linear_defect_strength = 0.0f0u"meV"
    linear_defect_disorder_scale = 0.0f0u"meV"
    # If true, each built-in-field defect randomly chooses + or - sign.
    linear_defect_random_sign::Bool = false
    # RNG seed makes the random defect locations/signs/disorder reproducible.
    linear_defect_rng_seed::Int = 44

    vacancy_count::Int = 10
    electron_count::Int = 20
    vacancy_charge_number::Float32 = 2.0f0
    electron_charge_number::Float32 = 1.0f0
    # Larger values make mobile electrons attempt hopping more often.
    electron_attempt_rate::Float32 = 10.0f0
    # Mobile vacancies/electrons are updated once every this many dynamics steps.
    # Smaller values mean more frequent defect motion.
    defect_step_interval::Int = 1000
    free_charge_split::Float32 = 0.5f0
    vacancy_quadratic_shift::Float32 = 0.012f0
    vacancy_quartic_shift::Float32 = 0.004f0
    defect_rng_seed::Int = 42

    steps::Int = 4000
    time_factor::Int = 1
    pulse_repeats::Int = 3
    relax_fraction::Float64 = 0.5

    algorithm_name::Symbol = :local_langevin
    langevin_stepsize::Float32 = 0.02f0
    langevin_adjusted::Bool = true
    proposer_delta::Float32 = 0.1f0

    # Weight function options for the Ising J term.
    # Simple path: set weight_mode and shell weights below, e.g.
    #   weight_mode = :nearest
    #   weight_mode = :shell
    #   weight_mode = :layered_afe
    #   weight_mode = :none
    # Custom path: pass weight_function = (cfg; dc) -> ... .
    # The custom function receives dc = (dx, dy, dz) and must return an energy
    # with units, for example: 0.2f0 * cfg.exchange_energy.
    # If weight_function is not nothing, it overrides weight_mode.
    weight_function = nothing
    weight_mode::Symbol = :shell
    # Maximum neighbor offset shell passed to WeightGenerator.
    weight_range::Int = 3
    # Dimensionless factors multiplied by exchange_energy in :shell mode.
    shell_weight_1::Float32 = 1.0f0
    shell_weight_2::Float32 = 0.1f0
    shell_weight_3::Float32 = 0.01f0
    shell_weight_beyond::Float32 = 0.0f0

    show_interfaces::Bool = true
    show_figures::Bool = false
    save_outputs::Bool = true
    save_figures::Bool = true
    save_excel::Bool = true
    outdir::String = raw"D:\Code\data\20260623"
end

env_bool(name, default) =
    lowercase(get(ENV, name, string(default))) in ("true", "1", "yes", "y", "on")

function override_config(cfg::ExperimentConfig; kwargs...)
    values = (;
        (name => getproperty(cfg, name) for name in fieldnames(ExperimentConfig))...,
    )
    return ExperimentConfig(; values..., kwargs...)
end

function config_from_environment()
    steps = parse(Int, get(ENV, "ISING_STEPS", "4000"))
    show_interfaces = env_bool("ISING_SHOW_INTERFACES", true)
    show_figures = env_bool("ISING_SHOW_FIGURES", true)
    save_outputs = env_bool("ISING_SAVE_OUTPUTS", true)
    save_figures = env_bool("ISING_SAVE_FIGURES", true)
    save_excel = env_bool("ISING_SAVE_EXCEL", true)
    outdir = get(ENV, "ISING_OUTDIR", raw"D:\Code\data\20260623")
    return ExperimentConfig(;
        steps,
        show_interfaces,
        show_figures,
        save_outputs,
        save_figures,
        save_excel,
        outdir,
    )
end

physical_scales(cfg::ExperimentConfig) = PhysicalScales(
    energy = cfg.energy_scale,
    length = cfg.length_scale,
    charge = cfg.elementary_charge,
    dipole = cfg.elementary_charge * cfg.length_scale,
)

internal_energy(value::Unitful.Quantity, cfg::ExperimentConfig) =
    Float32(Unitful.ustrip(Unitful.NoUnits, value / cfg.energy_scale))

internal_energy(value::Number, cfg::ExperimentConfig) = Float32(value)

function kBT_to_kelvin(kBT_internal, cfg::ExperimentConfig)
    energy = kBT_internal .* cfg.energy_scale
    return Unitful.ustrip.(u"K", energy ./ Unitful.k)
end

physical_vacancy_charge(cfg::ExperimentConfig) =
    cfg.vacancy_charge_number * cfg.elementary_charge

physical_electron_charge(cfg::ExperimentConfig) =
    cfg.electron_charge_number * cfg.elementary_charge

accepted_proposal_delta(proposal::II.FlipProposal) = II.accepteddelta(proposal)

function accepted_proposal_delta(proposal::II.MultiSpinProposal)
    total = zero(eltype(proposal))
    @inbounds for i in eachindex(proposal)
        total += II.accepteddelta(proposal, i)
    end
    return total
end

function select_dynamics(cfg::ExperimentConfig)
    cfg.algorithm_name in (:default, :metropolis) && return Metropolis()
    cfg.algorithm_name == :local_langevin &&
        return LocalLangevin(
            stepsize = cfg.langevin_stepsize,
            adjusted = cfg.langevin_adjusted,
        )
    cfg.algorithm_name == :global_langevin &&
        return GlobalLangevin(
            stepsize = cfg.langevin_stepsize,
            adjusted = cfg.langevin_adjusted,
        )
    error("Unsupported algorithm $(cfg.algorithm_name).")
end

mean_polarization_zlayer(model, zidx::Integer) =
    Float32(mean(state(model[1])[:, :, zidx]))

function staggered_z_polarization(model)
    s = state(model[1])
    total = 0.0f0
    for z in axes(s, 3)
        total += (isodd(z) ? 1.0f0 : -1.0f0) * Float32(mean(s[:, :, z]))
    end
    return total / size(s, 3)
end

struct TrianglePulse{T} <: ProcessAlgorithm
    amplitude::T
    repeats::Int
end

function StatefulAlgorithms.init(pulse::TrianglePulse, args)
    steps = num_calls(args)
    waveform = Vector{typeof(float(pulse.amplitude))}(undef, steps)
    denominator = max(steps - 1, 1)
    for i in eachindex(waveform)
        phase = 4 * pulse.repeats * (i - 1) / denominator
        segment = mod(phase, 4)
        waveform[i] = pulse.amplitude * (
            segment < 1 ? segment :
            segment < 2 ? 2 - segment :
            segment < 3 ? -(segment - 2) :
            segment - 4
        )
    end
    return (; waveform, step = 1, pulse_value = first(waveform))
end

function StatefulAlgorithms.step!(::TrianglePulse, context)
    (; waveform, step, hamiltonian) = context
    pulse_value = waveform[step]
    hamiltonian.b[] = pulse_value
    return (; step = step + 1, pulse_value)
end

struct FORCPulse{T} <: ProcessAlgorithm
    maximum_field::T
    preset_edge_points::Int
    test_field_step::T
    zero_hold_points::Int
end

FORCPulse(maximum_field, preset_edge_points::Integer) =
    FORCPulse(maximum_field, preset_edge_points, maximum_field / 100, 0)

FORCPulse(maximum_field, preset_edge_points::Integer, test_field_step) =
    FORCPulse(maximum_field, preset_edge_points, test_field_step, 0)

function FORCPulse(
    maximum_field,
    preset_edge_points::Integer,
    test_field_step,
    zero_hold_points::Integer,
)
    T = promote_type(typeof(maximum_field), typeof(test_field_step))
    return FORCPulse{T}(
        convert(T, maximum_field),
        Int(preset_edge_points),
        convert(T, test_field_step),
        Int(zero_hold_points),
    )
end

function forc_axis_by_step(from, to, step_size)
    step_size > 0 || error("FORC step size must be positive.")
    span = abs(to - from)
    intervals = max(1, round(Int, span / step_size))
    return collect(LinRange(from, to, intervals + 1))
end

function forc_axis_by_count(from, to, npoints::Integer)
    npoints >= 2 || error("A FORC edge needs at least 2 points.")
    return collect(LinRange(from, to, Int(npoints)))
end

function forc_protocol(pulse::FORCPulse)
    maximum_field = pulse.maximum_field
    preset_edge_points = pulse.preset_edge_points
    test_field_step = pulse.test_field_step
    zero_hold_points = pulse.zero_hold_points

    maximum_field > 0 || error("FORC maximum field must be positive.")
    preset_edge_points >= 2 || error("FORC preset edge points must be at least 2.")
    test_field_step > 0 || error("FORC test field step must be positive.")
    zero_hold_points >= 0 || error("FORC zero hold points must be nonnegative.")

    edge_step = maximum_field / (preset_edge_points - 1)
    n_curves = ceil(Int, maximum_field / test_field_step)
    end_fields = [min(i * test_field_step, maximum_field) for i in 1:n_curves]

    fields = typeof(maximum_field)[]
    curve = Int[]
    curve_end_field = typeof(maximum_field)[]

    for (curve_id, end_field) in enumerate(end_fields)
        preset_down = forc_axis_by_count(zero(maximum_field), -maximum_field, preset_edge_points)
        preset_up = forc_axis_by_count(-maximum_field, zero(maximum_field), preset_edge_points)
        test_up = forc_axis_by_step(zero(maximum_field), end_field, edge_step)
        test_down = forc_axis_by_step(end_field, zero(maximum_field), edge_step)
        hold = fill(zero(maximum_field), zero_hold_points)
        segment = vcat(preset_down, preset_up[2:end], test_up[2:end], test_down[2:end], hold)

        append!(fields, segment)
        append!(curve, fill(curve_id, length(segment)))
        append!(curve_end_field, fill(end_field, length(segment)))
    end

    return (; fields, curve, curve_end_field, end_fields, edge_step, test_field_step, zero_hold_points)
end

function StatefulAlgorithms.init(pulse::FORCPulse, args)
    protocol = forc_protocol(pulse)
    fields = protocol.fields
    curve = protocol.curve
    curve_end_field = protocol.curve_end_field
    steps = num_calls(args)

    if steps < length(fields)
        fields = fields[1:steps]
        curve = curve[1:steps]
        curve_end_field = curve_end_field[1:steps]
    elseif steps > length(fields)
        missing = steps - length(fields)
        fields = vcat(fields, fill(last(fields), missing))
        curve = vcat(curve, fill(last(curve), missing))
        curve_end_field = vcat(curve_end_field, fill(last(curve_end_field), missing))
    end

    return (;
        fields,
        curve,
        curve_end_field,
        step = 1,
        field = first(fields),
        forc_curve = first(curve),
        forc_end_field = first(curve_end_field),
    )
end

function StatefulAlgorithms.step!(::FORCPulse, context)
    (; fields, curve, curve_end_field, step, hamiltonian) = context
    field = fields[step]
    hamiltonian.b[] = field
    return (;
        step = step + 1,
        field,
        forc_curve = curve[step],
        forc_end_field = curve_end_field[step],
    )
end

function forc_protocol_dataframe(protocol, cfg::ExperimentConfig)
    field_meV = Float64.(protocol.fields) .* Unitful.ustrip(u"meV", cfg.energy_scale)
    return DataFrame(
        step = collect(eachindex(protocol.fields)),
        field = Float64.(protocol.fields),
        field_meV = field_meV,
        forc_curve = protocol.curve,
        forc_end_field = Float64.(protocol.curve_end_field),
    )
end

struct TemperatureCycle{T} <: ProcessAlgorithm
    maximum::T
    minimum::T
end

function StatefulAlgorithms.init(cycle::TemperatureCycle, args)
    steps = num_calls(args)
    if steps == 1
        return (;
            temperatures = [cycle.maximum],
            step = 1,
            temperature = cycle.maximum,
        )
    end
    first_half = cld(steps, 2)
    second_half = steps - first_half
    temperatures = collect(LinRange(cycle.maximum, cycle.minimum, first_half))
    if second_half > 0
        append!(
            temperatures,
            collect(LinRange(cycle.minimum, cycle.maximum, second_half)),
        )
    end
    return (; temperatures, step = 1, temperature = first(temperatures))
end

function StatefulAlgorithms.step!(::TemperatureCycle, context)
    (; temperatures, step, model) = context
    temperature = temperatures[step]
    temp!(model, temperature)
    return (; step = step + 1, temperature, T = temperature)
end

function shell_weight_factor(cfg::ExperimentConfig, shell::Integer)
    shell == 1 && return cfg.shell_weight_1
    shell == 2 && return cfg.shell_weight_2
    shell == 3 && return cfg.shell_weight_3
    return cfg.shell_weight_beyond
end

function shell_weight(cfg::ExperimentConfig; dc)
    if cfg.weight_function !== nothing
        return cfg.weight_function(cfg; dc)
    end

    dx, dy, dz = dc
    shell = dx * dx + dy * dy + dz * dz

    factor =
        cfg.weight_mode == :shell ? shell_weight_factor(cfg, shell) :
        cfg.weight_mode == :nearest ? (shell == 1 ? cfg.shell_weight_1 : 0.0f0) :
        cfg.weight_mode == :xy_nearest ?
            (shell == 1 && dz == 0 ? cfg.shell_weight_1 : 0.0f0) :
        cfg.weight_mode == :z_nearest ?
            (shell == 1 && dx == 0 && dy == 0 ? cfg.shell_weight_1 : 0.0f0) :
        cfg.weight_mode == :layered_afe ?
            (
                shell == 1 && dz == 0 ? cfg.shell_weight_1 :
                shell == 1 && dx == 0 && dy == 0 ? -abs(cfg.shell_weight_1) :
                shell_weight_factor(cfg, shell)
            ) :
        cfg.weight_mode == :none ? 0.0f0 :
        error("Unknown weight_mode $(cfg.weight_mode).")

    return factor * cfg.exchange_energy
end

const ACTIVE_WEIGHT_CONFIG = Ref{Any}()

function configured_shell_weight(; dc)
    return shell_weight(ACTIVE_WEIGHT_CONFIG[]; dc)
end

function normalize_adj_by_average_col!(adj, target)
    matrix = adj.sp
    average = mean(sum(abs, @view matrix[:, j]) for j in axes(matrix, 2))
    matrix .*= target / average
    return adj
end

function add_gaussian_disorder!(rng, values, scale::Float32)
    iszero(scale) && return values
    values .+= scale .* randn(rng, Float32, length(values))
    return values
end

function apply_landau_disorder!(
    coeff2,
    coeff4,
    coeff6,
    coeff8,
    coeff10,
    cfg::ExperimentConfig,
)
    cfg.apply_landau_disorder || return nothing
    rng = MersenneTwister(cfg.landau_disorder_seed)
    add_gaussian_disorder!(rng, coeff2, cfg.landau_a_disorder_scale)
    add_gaussian_disorder!(rng, coeff4, cfg.landau_b_disorder_scale)
    add_gaussian_disorder!(rng, coeff6, cfg.landau_c_disorder_scale)
    add_gaussian_disorder!(rng, coeff8, cfg.landau_d_disorder_scale)
    add_gaussian_disorder!(rng, coeff10, cfg.landau_e_disorder_scale)
    return nothing
end

function build_linear_defect_coefficients(cfg::ExperimentConfig, nspins::Integer)
    coefficients = zeros(Float32, nspins)
    count = clamp(cfg.linear_defect_count, 0, nspins)
    count == 0 && return (; coefficients, indices = Int[])

    rng = MersenneTwister(cfg.linear_defect_rng_seed)
    indices = randperm(rng, nspins)[1:count]
    center = internal_energy(cfg.linear_defect_strength, cfg)
    disorder = internal_energy(cfg.linear_defect_disorder_scale, cfg)

    for idx in indices
        value = center + disorder * randn(rng, Float32)
        if cfg.linear_defect_random_sign
            value *= rand(rng, Bool) ? 1.0f0 : -1.0f0
        end
        coefficients[idx] = value
    end

    return (; coefficients, indices)
end

function build_physical_model(cfg::ExperimentConfig)
    scales = physical_scales(cfg)
    nspins = cfg.nx * cfg.ny * cfg.nz
    coeff2 = fill(cfg.landau_a, nspins)
    coeff4 = fill(cfg.landau_b, nspins)
    coeff6 = fill(cfg.landau_c, nspins)
    coeff8 = fill(cfg.landau_d, nspins)
    coeff10 = fill(cfg.landau_e, nspins)
    apply_landau_disorder!(coeff2, coeff4, coeff6, coeff8, coeff10, cfg)
    linear_defect = build_linear_defect_coefficients(cfg, nspins)

    q_vacancy = physical_vacancy_charge(cfg)
    q_electron = physical_electron_charge(cfg)
    net_charge = cfg.vacancy_count * q_vacancy -
                 cfg.electron_count * q_electron
    iszero(net_charge) || error(
        "Mobile free charge must be neutral; current net charge is $net_charge.",
    )

    ACTIVE_WEIGHT_CONFIG[] = cfg
    weight_generator = PhysicalWeightGenerator(
        WeightGenerator(configured_shell_weight, cfg.weight_range),
    )

    hamiltonian =
        ExtField(b = cfg.initial_field) +
        Bilinear() +
        CoulombHamiltonian(
            scaling = cfg.coulomb_dipole_scale,
            screening = cfg.coulomb_screening,
            recalc = cfg.coulomb_recalc_interval,
            q_positive = q_vacancy,
            q_negative = q_electron,
            free_charge_split = cfg.free_charge_split,
        ) +
        PolynomialHamiltonian(
            1;
            c = cfg.energy_scale,
            localpotential = linear_defect.coefficients,
        ) +
        Quadratic(c = cfg.energy_scale, localpotential = coeff2) +
        Quartic(c = cfg.energy_scale, localpotential = coeff4) +
        Sextic(c = cfg.energy_scale, localpotential = coeff6)

    if cfg.include_landau_8 || any(x -> !iszero(x), coeff8)
        hamiltonian += Octic(c = cfg.energy_scale, localpotential = coeff8)
    end
    if cfg.include_landau_10 || any(x -> !iszero(x), coeff10)
        hamiltonian += PolynomialHamiltonian(
            10;
            c = cfg.energy_scale,
            localpotential = coeff10,
        )
    end

    graph = IsingGraph(
        cfg.nx,
        cfg.ny,
        cfg.nz,
        Continuous(),
        LocalProposer(cfg.proposer_delta),
        weight_generator,
        LatticeConstants(cfg.lattice_x, cfg.lattice_y, cfg.lattice_z),
        StateSet(-1.5f0, 1.5f0),
        hamiltonian;
        periodic = (:x, :y),
        precision = Float32,
        diag = StateLike(UniformArray),
        physical_scales = scales,
        temperature = cfg.initial_kBT,
    )

    normalize_adj_by_average_col!(
        graph.adj,
        internal_energy(cfg.exchange_energy, cfg),
    )

    vacancy_hamiltonian =
        CoulombChargeCoupling(q_vacancy; split = cfg.free_charge_split) +
        ExternalFieldChargeCoupling() +
        LocalPotentialShiftCoupling(2, cfg.vacancy_quadratic_shift) +
        LocalPotentialShiftCoupling(4, cfg.vacancy_quartic_shift)

    electron_hamiltonian =
        CoulombChargeCoupling(-q_electron; split = cfg.free_charge_split) +
        ExternalFieldChargeCoupling()

    defects = DefectsModel(
        graph;
        vacancies = MobileVacancies(
            cfg.vacancy_count;
            charge = q_vacancy,
            hamiltonian = vacancy_hamiltonian,
        ),
        charges = MobileCharges(
            cfg.electron_count;
            charge = -q_electron,
            hamiltonian = electron_hamiltonian,
        ),
        electron_attempt_rate = cfg.electron_attempt_rate,
        rng = MersenneTwister(cfg.defect_rng_seed),
    )

    if cfg.show_interfaces
        interface(graph; title = "Polarization")
        interface(
            defects;
            title = "Positive vacancies and mobile electrons",
            positive_color = :red,
            negative_color = :cyan,
        )
    end

    return (;
        graph,
        defects,
        scales,
        coeff2,
        coeff4,
        coeff6,
        coeff8,
        coeff10,
        linear_defect_coefficients = linear_defect.coefficients,
        linear_defect_indices = linear_defect.indices,
        net_charge,
    )
end

function experiment_sizes(cfg::ExperimentConfig)
    fullsweep = cfg.nx * cfg.ny * cfg.nz
    point_repeat = fullsweep * cfg.time_factor
    experiment_time = point_repeat * cfg.steps
    relax_time = round(Int, experiment_time * cfg.relax_fraction)
    return (; fullsweep, point_repeat, experiment_time, relax_time)
end

function make_loggers(cfg::ExperimentConfig)
    sizes = experiment_sizes(cfg)
    return (;
        sizes,
        polarization = IntegrateAndLog(Float32, sizes.point_repeat),
        field = ValueLogger(:field),
        temperature = ValueLogger(:temperature),
        depol = DepolLogger(:depol),
        staggered = ValueLogger(:P_AFE_z),
        top = ValueLogger(:P_top),
        middle = ValueLogger(:P_mid),
        bottom = ValueLogger(:P_bot),
    )
end

field_value(hamiltonian) = hamiltonian.b[]

struct LayerMean{Z} end
(::LayerMean{Z})(model) where {Z} = mean_polarization_zlayer(model, Z)

function build_pulse_job(cfg::ExperimentConfig; start::Bool = true)
    model = build_physical_model(cfg)
    graph = model.graph
    charges = model.defects
    reduced_summary = reduced_parameter_summary(cfg, model)

    loggers = make_loggers(cfg)
    point_repeat = loggers.sizes.point_repeat
    pulse_time = loggers.sizes.experiment_time
    relax_time = loggers.sizes.relax_time
    top_mean = LayerMean{cfg.nz}()
    mid_mean = LayerMean{cld(cfg.nz, 2)}()
    bottom_mean = LayerMean{1}()

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
            value = @transform(accepted_proposal_delta, proposal),
        )
        @every point_repeat loggers.field(
            value = @transform(field_value, dynamics.hamiltonian),
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

    algorithm = @Routine begin
        @repeat pulse_time pulse_step()
        @repeat relax_time relax_step()
    end

    process = Process(
        algorithm,
        StatefulAlgorithms.Init(:dynamics; model = graph),
        StatefulAlgorithms.Init(:charge_metro; model = charges),
        Init(loggers.polarization, initialvalue = sum(state(graph)));
        repeats = 1,
    )
    start && run(process)

    return (; cfg, model, graph, process, loggers, reduced_summary)
end

function build_anneal_job(
    cfg::ExperimentConfig;
    start_kBT = cfg.anneal_max_kBT,
    end_kBT = 0.0f0u"meV",
    start::Bool = true,
)
    cfg = override_config(cfg; initial_kBT = start_kBT, anneal_max_kBT = start_kBT)
    model = build_physical_model(cfg)
    graph = model.graph
    charges = model.defects
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
        internal_energy(start_kBT, cfg),
        internal_energy(end_kBT, cfg),
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

    algorithm = @Routine begin
        @repeat anneal_time anneal_step()
    end

    process = Process(
        algorithm,
        StatefulAlgorithms.Init(:dynamics; model = graph),
        StatefulAlgorithms.Init(:charge_metro; model = charges),
        Init(loggers.polarization, initialvalue = sum(state(graph)));
        repeats = 1,
    )
    start && run(process)

    return (; cfg, model, graph, process, loggers, reduced_summary)
end

function build_forc_job(
    cfg::ExperimentConfig;
    max_field,
    preset_edge_points::Integer,
    test_field_step = max_field / 100,
    zero_hold_points::Integer = 0,
    start::Bool = true,
    print_protocol::Bool = true,
)
    model = build_physical_model(cfg)
    graph = model.graph
    charges = model.defects
    reduced_summary = reduced_parameter_summary(cfg, model)

    loggers = make_loggers(cfg)
    point_repeat = loggers.sizes.point_repeat
    forc_pulse = FORCPulse(
        internal_energy(max_field, cfg),
        preset_edge_points,
        internal_energy(test_field_step, cfg),
        zero_hold_points,
    )
    protocol = forc_protocol(forc_pulse)
    forc_time = point_repeat * length(protocol.fields)
    relax_time = loggers.sizes.relax_time
    top_mean = LayerMean{cfg.nz}()
    mid_mean = LayerMean{cld(cfg.nz, 2)}()
    bottom_mean = LayerMean{1}()

    if print_protocol
        println("FORC field samples: ", length(protocol.fields))
        println("FORC curves: ", length(protocol.end_fields))
        println("FORC edge step: ", protocol.edge_step)
        println("FORC test field step: ", protocol.test_field_step)
        println("FORC zero-hold samples per curve: ", protocol.zero_hold_points)
    end

    dynamics = select_dynamics(cfg)
    charge_dynamics = Metropolis()

    forc_monte_carlo = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias charge_metro = charge_dynamics
        @replace dynamics.T => charge_metro.T

        proposal = dynamics()
        @every cfg.defect_step_interval charge_metro()
        @every 1 loggers.polarization(
            value = @transform(accepted_proposal_delta, proposal),
        )
        @every point_repeat loggers.field(
            value = @transform(field_value, dynamics.hamiltonian),
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

    forc_step = @CompositeAlgorithm begin
        @context monte_carlo = forc_monte_carlo()
        @every point_repeat forc_pulse(
            hamiltonian = monte_carlo.dynamics.hamiltonian,
        )
    end

    relax_step = @CompositeAlgorithm begin
        @context monte_carlo = forc_monte_carlo()
    end

    algorithm = @Routine begin
        @repeat forc_time forc_step()
        @repeat relax_time relax_step()
    end

    process = Process(
        algorithm,
        StatefulAlgorithms.Init(:dynamics; model = graph),
        StatefulAlgorithms.Init(:charge_metro; model = charges),
        Init(loggers.polarization, initialvalue = sum(state(graph)));
        repeats = 1,
    )
    start && run(process)

    return (; cfg, model, graph, process, loggers, reduced_summary, protocol)
end

function collect_logged_result(context, loggers; include_temperature = false)
    depol = context[loggers.depol]
    result = (;
        field = context[loggers.field].values,
        polarization = context[loggers.polarization].log,
        depol_mean = depol.means,
        depol_median = depol.medians,
        depol_max = depol.maxima,
        total_energy = depol.total_energy,
        coulomb_energy = depol.depol_energy,
        interaction_energy = depol.interaction_energy,
        field_energy = depol.field_energy,
        polynomial_energy = depol.poly_energy,
        staggered = context[loggers.staggered].values,
        top = context[loggers.top].values,
        middle = context[loggers.middle].values,
        bottom = context[loggers.bottom].values,
    )
    return include_temperature ?
        merge(result, (; temperature = context[loggers.temperature].values)) :
        result
end

function make_series_figure(x, y; xlabel, ylabel, title)
    figure = Figure()
    axis = Axis(figure[1, 1]; xlabel, ylabel, title)
    n = min(length(x), length(y))
    lines!(axis, x[1:n], y[1:n])
    return figure
end

function make_multi_series_figure(x, series; xlabel, ylabel, title)
    figure = Figure()
    axis = Axis(figure[1, 1]; xlabel, ylabel, title)
    for (label, y) in series
        n = min(length(x), length(y))
        lines!(axis, x[1:n], y[1:n], label = label)
    end
    axislegend(axis)
    return figure
end

function landau_energy(P, cfg::ExperimentConfig)
    return cfg.landau_a * P^2 +
           cfg.landau_b * P^4 +
           cfg.landau_c * P^6 +
           cfg.landau_d * P^8 +
           cfg.landau_e * P^10
end

function estimate_landau_barrier(
    cfg::ExperimentConfig;
    Pmin = -1.5,
    Pmax = 1.5,
    ngrid = 20001,
)
    xs = collect(range(Pmin, Pmax, length = ngrid))
    ys = [landau_energy(x, cfg) for x in xs]

    min_idxs = Int[]
    max_idxs = Int[]
    for i in 2:(length(xs) - 1)
        if ys[i] <= ys[i - 1] && ys[i] <= ys[i + 1]
            push!(min_idxs, i)
        end
        if ys[i] >= ys[i - 1] && ys[i] >= ys[i + 1]
            push!(max_idxs, i)
        end
    end

    pos_min_idxs = filter(i -> xs[i] > 0, min_idxs)
    isempty(pos_min_idxs) &&
        error("No positive local minimum found in the scanned Landau window.")

    well_idx = pos_min_idxs[argmin(ys[pos_min_idxs])]
    P0 = xs[well_idx]
    Ewell = ys[well_idx]

    between_max_idxs = filter(i -> 0 <= xs[i] <= P0, max_idxs)
    barrier_idx = isempty(between_max_idxs) ?
        argmin(abs.(xs)) :
        between_max_idxs[argmax(ys[between_max_idxs])]
    Ps = xs[barrier_idx]
    Ebarrier = ys[barrier_idx]

    return (; P0, Ps, DeltaF = Ebarrier - Ewell, Ewell, Ebarrier)
end

function avg_interaction_scale(adj_like)
    matrix = hasproperty(adj_like, :sp) ? adj_like.sp : adj_like
    colsums = Float64[]
    for j in axes(matrix, 2)
        push!(colsums, sum(abs, @view matrix[:, j]))
    end
    return (; SJ = mean(colsums), SJ_min = minimum(colsums), SJ_max = maximum(colsums))
end

function with_saved_coulomb_state(f, graph, coulomb_term)
    saved_state = copy(state(graph))
    saved_tracker = coulomb_term.recalc_tracker[]
    try
        return f()
    finally
        state(graph) .= saved_state
        InteractiveIsing.init!(coulomb_term, graph)
        coulomb_term.recalc_tracker[] = saved_tracker
    end
end

function reference_coulomb_scale(graph, coulomb_term, P0)
    return with_saved_coulomb_state(graph, coulomb_term) do
        state(graph) .= P0
        InteractiveIsing.init!(coulomb_term, graph)
        values = Float64[]
        for i in eachindex(state(graph))
            proposal = SingleSpinProposal(i, state(graph)[i], NoChange(), 1)
            local_field = abs(
                InteractiveIsing.calculate(
                    InteractiveIsing.d_iH(),
                    coulomb_term,
                    graph,
                    proposal,
                ),
            )
            push!(values, 2 * abs(P0) * local_field)
        end
        return (; mean = mean(values), median = median(values), maximum = maximum(values))
    end
end

function reduced_parameter_summary(cfg::ExperimentConfig, model; print_summary = true)
    graph = model.graph
    barrier = estimate_landau_barrier(cfg)
    interaction = avg_interaction_scale(adj(graph))
    coulomb_term = find_first_term(graph.hamiltonian, InteractiveIsing.CoulombHamiltonian)

    P0 = barrier.P0
    DeltaF = barrier.DeltaF
    SJ = interaction.SJ
    field_typ = internal_energy(cfg.pulse_amplitude, cfg)
    defect_typ = max(abs(cfg.vacancy_quadratic_shift), abs(cfg.vacancy_quartic_shift))
    linear_defect_typ = maximum(abs, model.linear_defect_coefficients)
    depol_ref = isnothing(coulomb_term) ? nothing :
        reference_coulomb_scale(graph, coulomb_term, P0)
    depol_current = isnothing(coulomb_term) ? nothing :
        coulomb_local_scale(graph, coulomb_term)

    rows = NamedTuple[]
    addrow!(section, key, value; note = "") =
        push!(rows, (; section, key, value, note))

    addrow!("input", "energy_scale", string(cfg.energy_scale))
    addrow!("input", "exchange_energy", string(cfg.exchange_energy))
    addrow!("input", "coulomb_dipole_scale", string(cfg.coulomb_dipole_scale))
    addrow!("input", "coulomb_screening", string(cfg.coulomb_screening))
    addrow!("input", "pulse_amplitude", string(cfg.pulse_amplitude))
    addrow!("input", "weight_mode", string(cfg.weight_mode))
    addrow!("input", "weight_range", cfg.weight_range)
    addrow!("input", "shell_weight_1", cfg.shell_weight_1)
    addrow!("input", "shell_weight_2", cfg.shell_weight_2)
    addrow!("input", "shell_weight_3", cfg.shell_weight_3)
    addrow!("input", "vacancy_quadratic_shift", cfg.vacancy_quadratic_shift)
    addrow!("input", "vacancy_quartic_shift", cfg.vacancy_quartic_shift)
    addrow!("input", "linear_defect_count", cfg.linear_defect_count)
    addrow!("input", "linear_defect_strength", string(cfg.linear_defect_strength))
    addrow!(
        "input",
        "linear_defect_disorder_scale",
        string(cfg.linear_defect_disorder_scale),
    )
    addrow!("input", "linear_defect_random_sign", cfg.linear_defect_random_sign)
    addrow!("input", "apply_landau_disorder", cfg.apply_landau_disorder)
    addrow!("input", "landau_a_disorder_scale", cfg.landau_a_disorder_scale)
    addrow!("input", "landau_b_disorder_scale", cfg.landau_b_disorder_scale)
    addrow!("input", "landau_c_disorder_scale", cfg.landau_c_disorder_scale)
    addrow!("landau", "P0", P0)
    addrow!("landau", "Ps", barrier.Ps)
    addrow!("landau", "DeltaF_barrier_internal", DeltaF)
    addrow!("landau", "DeltaF_barrier_physical", string(DeltaF * cfg.energy_scale))
    addrow!("landau", "Ewell_internal", barrier.Ewell)
    addrow!("landau", "Ewell_physical", string(barrier.Ewell * cfg.energy_scale))
    addrow!("landau", "Ebarrier_internal", barrier.Ebarrier)
    addrow!("landau", "Ebarrier_physical", string(barrier.Ebarrier * cfg.energy_scale))
    addrow!("interaction", "S_J", SJ)
    addrow!("interaction", "S_J_physical", string(SJ * cfg.energy_scale))
    addrow!("interaction", "S_J_min", interaction.SJ_min)
    addrow!("interaction", "S_J_max", interaction.SJ_max)
    addrow!("reduced", "Lambda_int", (P0^2 * SJ) / DeltaF)
    addrow!("reduced", "Lambda_barrier", DeltaF / (P0^2 * SJ))
    addrow!("reduced", "Lambda_field_pulse", abs(field_typ) / (P0 * SJ))
    addrow!("reduced", "Lambda_defect_local_shift", abs(defect_typ) / (P0 * SJ))
    addrow!("reduced", "Lambda_linear_defect", linear_defect_typ / (P0 * SJ))
    addrow!("reduced", "Theta_field_pulse", abs(P0 * field_typ) / DeltaF)
    addrow!("reduced", "Theta_defect_local_shift", abs(P0 * defect_typ) / DeltaF)
    addrow!("reduced", "Theta_linear_defect", abs(P0 * linear_defect_typ) / DeltaF)

    if !isnothing(depol_ref)
        addrow!("reference_depol", "reference_state", "all dipoles at +P0")
        addrow!(
            "reference_depol",
            "E_dep_ref_mean",
            depol_ref.mean;
            note = "estimated full-flip depolarization work scale",
        )
        addrow!("reference_depol", "E_dep_ref_median", depol_ref.median)
        addrow!("reference_depol", "E_dep_ref_max", depol_ref.maximum)
        addrow!("reference_depol", "Lambda_dep_ref_mean", depol_ref.mean / (P0^2 * SJ))
        addrow!("reference_depol", "Lambda_dep_ref_median", depol_ref.median / (P0^2 * SJ))
        addrow!("reference_depol", "Theta_dep_ref_mean", depol_ref.mean / DeltaF)
        addrow!("reference_depol", "Theta_dep_ref_median", depol_ref.median / DeltaF)
        addrow!(
            "current_depol",
            "dH_dep_current_mean",
            depol_current.mean;
            note = "state-dependent local derivative diagnostic",
        )
        addrow!("current_depol", "dH_dep_current_median", depol_current.median)
        addrow!("current_depol", "dH_dep_current_max", depol_current.maximum)
    end

    dataframe = DataFrame(rows)

    if print_summary
        println()
        println("=== Reduced Parameter Summary ===")
        for row in eachrow(dataframe)
            note = isempty(row.note) ? "" : "  # $(row.note)"
            println("  ", row.key, " = ", row.value, note)
        end
        println("=================================")
        println()
    end

    return dataframe
end

function config_dataframe(cfg::ExperimentConfig)
    rows = Pair{String,Any}[
        "nx" => cfg.nx,
        "ny" => cfg.ny,
        "nz" => cfg.nz,
        "energy_scale" => string(cfg.energy_scale),
        "length_scale" => string(cfg.length_scale),
        "exchange_energy" => string(cfg.exchange_energy),
        "coulomb_dipole_scale" => string(cfg.coulomb_dipole_scale),
        "coulomb_screening" => string(cfg.coulomb_screening),
        "initial_kBT" => string(cfg.initial_kBT),
        "initial_temperature_K" =>
            only(kBT_to_kelvin([internal_energy(cfg.initial_kBT, cfg)], cfg)),
        "anneal_max_kBT" => string(cfg.anneal_max_kBT),
        "pulse_amplitude" => string(cfg.pulse_amplitude),
        "initial_field" => string(cfg.initial_field),
        "landau_a_meV" => cfg.landau_a,
        "landau_b_meV" => cfg.landau_b,
        "landau_c_meV" => cfg.landau_c,
        "landau_d_meV" => cfg.landau_d,
        "landau_e_meV" => cfg.landau_e,
        "include_landau_8" => cfg.include_landau_8,
        "include_landau_10" => cfg.include_landau_10,
        "apply_landau_disorder" => cfg.apply_landau_disorder,
        "landau_disorder_seed" => cfg.landau_disorder_seed,
        "landau_a_disorder_scale" => cfg.landau_a_disorder_scale,
        "landau_b_disorder_scale" => cfg.landau_b_disorder_scale,
        "landau_c_disorder_scale" => cfg.landau_c_disorder_scale,
        "landau_d_disorder_scale" => cfg.landau_d_disorder_scale,
        "landau_e_disorder_scale" => cfg.landau_e_disorder_scale,
        "linear_defect_count" => cfg.linear_defect_count,
        "linear_defect_strength" => string(cfg.linear_defect_strength),
        "linear_defect_disorder_scale" =>
            string(cfg.linear_defect_disorder_scale),
        "linear_defect_random_sign" => cfg.linear_defect_random_sign,
        "linear_defect_rng_seed" => cfg.linear_defect_rng_seed,
        "vacancy_count" => cfg.vacancy_count,
        "electron_count" => cfg.electron_count,
        "vacancy_charge" => string(physical_vacancy_charge(cfg)),
        "electron_charge" => string(-physical_electron_charge(cfg)),
        "defect_step_interval" => cfg.defect_step_interval,
        "steps" => cfg.steps,
        "algorithm" => string(cfg.algorithm_name),
        "proposer_delta" => cfg.proposer_delta,
        "weight_mode" => string(cfg.weight_mode),
        "weight_range" => cfg.weight_range,
        "shell_weight_1" => cfg.shell_weight_1,
        "shell_weight_2" => cfg.shell_weight_2,
        "shell_weight_3" => cfg.shell_weight_3,
        "shell_weight_beyond" => cfg.shell_weight_beyond,
    ]
    return DataFrame(key = first.(rows), value = last.(rows))
end

function coefficient_dataframe(model)
    nspins = length(model.coeff2)
    defect_sites = falses(nspins)
    for idx in model.linear_defect_indices
        defect_sites[idx] = true
    end
    return DataFrame(
        site = collect(1:nspins),
        linear_defect = model.linear_defect_coefficients,
        is_linear_defect = defect_sites,
        landau_a = model.coeff2,
        landau_b = model.coeff4,
        landau_c = model.coeff6,
        landau_d = model.coeff8,
        landau_e = model.coeff10,
    )
end

function result_dataframe(result)
    columns = Pair{Symbol,Any}[]
    for name in propertynames(result)
        values = getproperty(result, name)
        values isa AbstractVector || continue
        push!(columns, name => values)
    end
    isempty(columns) && return DataFrame()

    nrows = maximum(length(last(pair)) for pair in columns)
    table = Dict{Symbol,Vector{Union{Missing,Float64}}}()
    for (name, values) in columns
        column = Vector{Union{Missing,Float64}}(missing, nrows)
        column[1:length(values)] .= Float64.(values)
        table[name] = column
    end
    return DataFrame(table)
end

function write_dataframe_sheet!(sheet, dataframe)
    if ncol(dataframe) == 0
        sheet["A1"] = "empty"
        return nothing
    end
    for (column, name) in enumerate(names(dataframe))
        sheet[XLSX.CellRef(1, column)] = name
    end
    for row in 1:nrow(dataframe), column in 1:ncol(dataframe)
        value = dataframe[row, column]
        sheet[XLSX.CellRef(row + 1, column)] =
            ismissing(value) ? missing :
            value isa Union{Number,String,Bool,Date,DateTime,Time} ? value :
            string(value)
    end
    return nothing
end

function save_experiment(
    experiment_name,
    cfg::ExperimentConfig,
    result,
    figures::Dict{String,<:Any},
    reduced_summary = DataFrame(),
    extra_sheets = Pair{String,DataFrame}[],
)
    cfg.save_outputs || return nothing
    mkpath(cfg.outdir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    basename = "$(experiment_name)_$(stamp)"

    saved_figures = String[]
    if cfg.save_figures
        for (name, figure) in figures
            path = joinpath(cfg.outdir, "$(basename)_$(name).png")
            save(path, figure)
            push!(saved_figures, path)
        end
    end

    xlsx_path = joinpath(cfg.outdir, basename * ".xlsx")
    if cfg.save_excel
        XLSX.openxlsx(xlsx_path, mode = "w") do workbook
            workbook[1].name = "series"
            write_dataframe_sheet!(workbook["series"], result_dataframe(result))
            XLSX.addsheet!(workbook, "parameters")
            write_dataframe_sheet!(
                workbook["parameters"],
                config_dataframe(cfg),
            )
            XLSX.addsheet!(workbook, "reduced_energy")
            write_dataframe_sheet!(workbook["reduced_energy"], reduced_summary)
            for (name, dataframe) in extra_sheets
                XLSX.addsheet!(workbook, name)
                write_dataframe_sheet!(workbook[name], dataframe)
            end
        end
    end

    return (; xlsx_path, saved_figures)
end

function print_physical_summary(cfg::ExperimentConfig, model)
    graph = model.graph
    println("Energy scale: ", physicalscales(graph).energy[])
    println("Length scale: ", physicalscales(graph).length[])
    println("Initial kBT: ", cfg.initial_kBT)
    println(
        "Initial temperature: ",
        only(kBT_to_kelvin([temp(graph)], cfg)),
        " K",
    )
    println("Exchange energy: ", cfg.exchange_energy)
    println("Coulomb dipole scale: ", cfg.coulomb_dipole_scale)
    println("Coulomb screening length: ", cfg.coulomb_screening)
    println("Net mobile charge: ", model.net_charge)
end
