struct AnnealRun
    algorithm
    M_Integrator
    M_Logger
    B_Logger
    T_Logger
end

struct PulseRun
    algorithm
    M_Integrator
    M_Logger
    B_Logger
    Graph_Logger
    capture_dir::String
end

dsl_count(x) = max(1, round(Int, x))

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
        Graph_Logger = ImageCapture(:Graph, -1.5, 1.5),
        capture_dir,
    )
end

function anneal_run(algorithm, parts)
    return AnnealRun(algorithm, parts.M_Integrator, parts.M_Logger, parts.B_Logger, parts.T_Logger)
end

function pulse_run(algorithm, parts)
    return PulseRun(algorithm, parts.M_Integrator, parts.M_Logger, parts.B_Logger, parts.Graph_Logger, parts.capture_dir)
end

function build_anneal_process(g, p::ManuscriptParams)
    parts = anneal_components(g, p)
    (; d, dynamics, annealing, M_Integrator, M_Logger, B_Logger, T_Logger) = parts
    point_repeat = dsl_count(d.point_repeat)
    anneal_time = dsl_count(d.anneal_time)

    Metro_T = @CompositeAlgorithm begin
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
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
        @every point_repeat T_Logger(
            value = @transform(InteractiveIsing.temp, dynamics.model),
        )
    end

    anneal_part = @CompositeAlgorithm begin
        @alias annealing = annealing
        @context metro_t = Metro_T()

        @every point_repeat annealing(model = metro_t.dynamics.model)
    end

    algorithm = @Routine begin
        @repeat anneal_time anneal_part()
    end

    return anneal_run(algorithm, parts)
end

function build_pulse_process(g, p::ManuscriptParams; capture_dir = joinpath(p.outdir, "capture"))
    parts = pulse_components(g, p; capture_dir)
    (; d, dynamics, pulse, M_Integrator, M_Logger, B_Logger, Graph_Logger) = parts
    point_repeat = dsl_count(d.point_repeat)
    capture_interval1 = dsl_count(d.capture_interval1)
    capture_interval2 = dsl_count(d.capture_interval2)
    pulse_time = dsl_count(d.pulse_time)
    relax_time = dsl_count(d.relax_time)

    Metro_Pulse = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias M_Integrator = M_Integrator
        @alias M_Logger = M_Logger
        @alias B_Logger = B_Logger

        proposal = dynamics()
        total = M_Integrator(
            Δvalue = @transform(accepted_proposal_delta, proposal),
        )
        @every point_repeat M_Logger(value = total)
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
    end

    pulse_part = @CompositeAlgorithm begin
        @alias pulse = pulse
        @alias Graph_Logger = Graph_Logger
        @context metro_pulse = Metro_Pulse()

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
        @context metro_pulse = Metro_Pulse()

        @every capture_interval2 Graph_Logger(
            array = @transform(graph_array, metro_pulse.dynamics.model),
        )
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
    inputs = isnothing(run.Graph_Logger) ?
        (Input(run.M_Integrator, initialvalue = sum(graph_array(g))),) :
        (
            Input(run.Graph_Logger, filepath = run.capture_dir),
            Input(run.M_Integrator, initialvalue = sum(graph_array(g))),
        )

    return createProcess(
        g,
        run.algorithm,
        inputs...;
        repeats = repeats,
    )
end

function fetch_run(process, run::AnnealRun)
    context = fetch(process)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Logger].log,
        Temp = context[run.T_Logger].values,
    )
end

function fetch_run(process, run::PulseRun)
    context = fetch(process)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Logger].log,
    )
end
