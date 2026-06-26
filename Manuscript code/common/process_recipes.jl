struct AnnealRun
    algorithm
    M_Integrator
    M_Logger
    B_Logger
    T_Logger
    Depol_Logger
    PAFEz_Logger
    PTop_Logger
    PMid_Logger
    PBot_Logger
end

struct PulseRun
    algorithm
    M_Integrator
    M_Logger
    B_Logger
    Depol_Logger
    PAFEz_Logger
    PTop_Logger
    PMid_Logger
    PBot_Logger
    Graph_Logger
    capture_dir::String
end

dsl_count(x) = max(1, round(Int, x))

function diagnostic_loggers(p::ManuscriptParams)
    if !p.log_diagnostics
        return (;
            Depol_Logger = nothing,
            PAFEz_Logger = nothing,
            PTop_Logger = nothing,
            PMid_Logger = nothing,
            PBot_Logger = nothing,
        )
    end
    return (;
        Depol_Logger = DepolLogger(:depol),
        PAFEz_Logger = ValueLogger(:P_AFE_z),
        PTop_Logger = ValueLogger(:P_top),
        PMid_Logger = ValueLogger(:P_mid),
        PBot_Logger = ValueLogger(:P_bot),
    )
end

function anneal_components(g, p::ManuscriptParams)
    d = derived_params(p)
    return (;
        d,
        dynamics = select_dynamics(g, p),
        annealing = LinAnealingB(p.Temp_aneal, 0f0),
        M_Integrator = Integrator(Float32, name = :M_integrator),
        M_Logger = Logger(Float32, name = :M_logger),
        B_Logger = ValueLogger(:b),
        T_Logger = ValueLogger(:T),
        diagnostic_loggers(p)...,
    )
end

function pulse_components(g, p::ManuscriptParams; capture_dir = joinpath(p.outdir, "capture"))
    d = derived_params(p)
    return (;
        d,
        dynamics = select_dynamics(g, p),
        pulse = TrianglePulseA(p.Amp1, p.nrepeats),
        M_Integrator = Integrator(Float32, name = :M_integrator),
        M_Logger = Logger(Float32, name = :M_logger),
        B_Logger = ValueLogger(:b),
        Graph_Logger = p.capture ? ImageCapture(:Graph, p.state_min, p.state_max) : nothing,
        capture_dir,
        diagnostic_loggers(p)...,
    )
end

function anneal_run(algorithm, parts)
    return AnnealRun(
        algorithm,
        parts.M_Integrator,
        parts.M_Logger,
        parts.B_Logger,
        parts.T_Logger,
        parts.Depol_Logger,
        parts.PAFEz_Logger,
        parts.PTop_Logger,
        parts.PMid_Logger,
        parts.PBot_Logger,
    )
end

function pulse_run(algorithm, parts)
    return PulseRun(
        algorithm,
        parts.M_Integrator,
        parts.M_Logger,
        parts.B_Logger,
        parts.Depol_Logger,
        parts.PAFEz_Logger,
        parts.PTop_Logger,
        parts.PMid_Logger,
        parts.PBot_Logger,
        parts.Graph_Logger,
        parts.capture_dir,
    )
end

function build_anneal_process(g, p::ManuscriptParams)
    parts = anneal_components(g, p)
    (; d, dynamics, annealing, M_Integrator, M_Logger, B_Logger, T_Logger,
        Depol_Logger, PAFEz_Logger, PTop_Logger, PMid_Logger, PBot_Logger) = parts
    point_repeat = dsl_count(d.point_repeat)
    anneal_time = dsl_count(d.anneal_time)

    if p.log_diagnostics
        metro_t = @CompositeAlgorithm begin
            @alias dynamics = dynamics
            @alias M_Integrator = M_Integrator
            @alias M_Logger = M_Logger
            @alias B_Logger = B_Logger
            @alias T_Logger = T_Logger
            @alias Depol_Logger = Depol_Logger
            @alias PAFEz_Logger = PAFEz_Logger
            @alias PTop_Logger = PTop_Logger
            @alias PMid_Logger = PMid_Logger
            @alias PBot_Logger = PBot_Logger

            proposal = dynamics()
            total = M_Integrator(
                Δvalue = @transform(accepted_proposal_delta, proposal),
            )
            @every point_repeat M_Logger(value = total)
            @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
            @every point_repeat T_Logger(value = @transform(InteractiveIsing.temp, dynamics.model))
            @every point_repeat Depol_Logger(
                model = dynamics.model,
                hamiltonian = dynamics.hamiltonian,
            )
            @every point_repeat PAFEz_Logger(
                value = @transform(staggered_z_polarization, dynamics.model),
            )
            @every point_repeat PTop_Logger(
                value = @transform(mean_polarization_top, dynamics.model),
            )
            @every point_repeat PMid_Logger(
                value = @transform(mean_polarization_mid, dynamics.model),
            )
            @every point_repeat PBot_Logger(
                value = @transform(mean_polarization_bottom, dynamics.model),
            )
        end
    else
        metro_t = @CompositeAlgorithm begin
            @alias dynamics = dynamics
            @alias M_Integrator = M_Integrator
            @alias M_Logger = M_Logger
            @alias B_Logger = B_Logger
            @alias T_Logger = T_Logger

            proposal = dynamics()
            total = M_Integrator(
                Δvalue = @transform(accepted_proposal_delta, proposal),
            )
            @every point_repeat M_Logger(value = total)
            @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
            @every point_repeat T_Logger(value = @transform(InteractiveIsing.temp, dynamics.model))
        end
    end

    anneal_part = @CompositeAlgorithm begin
        @alias annealing = annealing
        @context metro_t = metro_t()
        @every point_repeat annealing(model = metro_t.dynamics.model)
    end

    algorithm = @Routine begin
        @repeat anneal_time anneal_part()
    end
    return anneal_run(algorithm, parts)
end

function build_pulse_process(g, p::ManuscriptParams; capture_dir = joinpath(p.outdir, "capture"))
    p.Steps_1 % (4 * p.nrepeats) == 0 || throw(ArgumentError(
        "`Steps_1` must be divisible by `4 * nrepeats`; got Steps_1=$(p.Steps_1), nrepeats=$(p.nrepeats).",
    ))

    parts = pulse_components(g, p; capture_dir)
    (; d, dynamics, pulse, M_Integrator, M_Logger, B_Logger,
        Depol_Logger, PAFEz_Logger, PTop_Logger, PMid_Logger, PBot_Logger,
        Graph_Logger) = parts
    point_repeat = dsl_count(d.point_repeat)
    capture_interval1 = dsl_count(d.capture_interval1)
    capture_interval2 = dsl_count(d.capture_interval2)
    pulse_time = dsl_count(d.pulse_time)
    relax_time = dsl_count(d.relax_time)

    if p.log_diagnostics
        metro_pulse = @CompositeAlgorithm begin
            @alias dynamics = dynamics
            @alias M_Integrator = M_Integrator
            @alias M_Logger = M_Logger
            @alias B_Logger = B_Logger
            @alias Depol_Logger = Depol_Logger
            @alias PAFEz_Logger = PAFEz_Logger
            @alias PTop_Logger = PTop_Logger
            @alias PMid_Logger = PMid_Logger
            @alias PBot_Logger = PBot_Logger

            proposal = dynamics()
            total = M_Integrator(
                Δvalue = @transform(accepted_proposal_delta, proposal),
            )
            @every point_repeat M_Logger(value = total)
            @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
            @every point_repeat Depol_Logger(
                model = dynamics.model,
                hamiltonian = dynamics.hamiltonian,
            )
            @every point_repeat PAFEz_Logger(
                value = @transform(staggered_z_polarization, dynamics.model),
            )
            @every point_repeat PTop_Logger(
                value = @transform(mean_polarization_top, dynamics.model),
            )
            @every point_repeat PMid_Logger(
                value = @transform(mean_polarization_mid, dynamics.model),
            )
            @every point_repeat PBot_Logger(
                value = @transform(mean_polarization_bottom, dynamics.model),
            )
        end
    else
        metro_pulse = @CompositeAlgorithm begin
            @alias dynamics = dynamics
            @alias M_Integrator = M_Integrator
            @alias M_Logger = M_Logger
            @alias B_Logger = B_Logger

            proposal = dynamics()
            total = M_Integrator(
                Δvalue = @transform(accepted_proposal_delta, proposal),
            )
            @every point_repeat M_Logger(value = total)
            @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
        end
    end

    if p.capture
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = metro_pulse()
            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
            @every capture_interval1 Graph_Logger(
                array = @transform(graph_array, metro_pulse.dynamics.model),
            )
        end
        relax_part = @CompositeAlgorithm begin
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = metro_pulse()
            @every capture_interval2 Graph_Logger(
                array = @transform(graph_array, metro_pulse.dynamics.model),
            )
        end
    else
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @context metro_pulse = metro_pulse()
            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
        end
        relax_part = @CompositeAlgorithm begin
            @context metro_pulse = metro_pulse()
        end
    end

    algorithm = @Routine begin
        @repeat pulse_time pulse_part()
        @repeat relax_time relax_part()
    end
    return pulse_run(algorithm, parts)
end

function start_anneal!(g, run::AnnealRun; repeats = 1)
    return createProcess(g, run.algorithm, repeats = repeats)
end

function start_pulse!(g, run::PulseRun; repeats = 1)
    return createProcess(g, run.algorithm, basefile_process_inputs(g, run)...; repeats)
end

function basefile_process_inputs(g, run::PulseRun)
    if isnothing(run.Graph_Logger)
        return (Init(run.M_Integrator, initialvalue = sum(graph_array(g))),)
    end
    return (
        Init(run.Graph_Logger, filepath = run.capture_dir),
        Init(run.M_Integrator, initialvalue = sum(graph_array(g))),
    )
end

basefile_process_inputs(g, run::AnnealRun) =
    (Init(run.M_Integrator, initialvalue = sum(graph_array(g))),)

_logger_values(context, logger) = isnothing(logger) ? nothing : context[logger].values

function _diagnostic_result(context, run)
    if isnothing(run.Depol_Logger)
        return (;
            depol_mean = nothing, depol_median = nothing, depol_max = nothing,
            H_total = nothing, H_dep = nothing, H_J = nothing,
            H_field = nothing, H_poly = nothing, H_rest = nothing,
            P_AFE_z = nothing, P_top = nothing, P_mid = nothing, P_bot = nothing,
        )
    end
    depol = context[run.Depol_Logger]
    H_rest = depol.depol_energy .+ depol.interaction_energy .+ depol.poly_energy
    return (;
        depol_mean = depol.means,
        depol_median = depol.medians,
        depol_max = depol.maxima,
        H_total = depol.total_energy,
        H_dep = depol.depol_energy,
        H_J = depol.interaction_energy,
        H_field = depol.field_energy,
        H_poly = depol.poly_energy,
        H_rest,
        P_AFE_z = _logger_values(context, run.PAFEz_Logger),
        P_top = _logger_values(context, run.PTop_Logger),
        P_mid = _logger_values(context, run.PMid_Logger),
        P_bot = _logger_values(context, run.PBot_Logger),
    )
end

function read_run_context(context, run::AnnealRun)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Logger].log,
        Temp = context[run.T_Logger].values,
        _diagnostic_result(context, run)...,
    )
end

function read_run_context(context, run::PulseRun)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Logger].log,
        _diagnostic_result(context, run)...,
    )
end

fetch_run(process, run::Union{AnnealRun,PulseRun}) = read_run_context(fetch(process), run)
