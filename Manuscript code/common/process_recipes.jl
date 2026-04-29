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

function build_anneal_process(g, p::ManuscriptParams)
    d = derived_params(p)
    metropolis = g.default_algorithm
    anealing = LinAnealingB(p.Temp_aneal, 0f0)

    M_Integrator = Integrator(Float32, name = :M_integrator)
    M_Logger = Logger(Float32, name = :M_logger)
    B_Logger = ValueLogger(:b)
    T_Logger = ValueLogger(:T)

    metro_t = CompositeAlgorithm(
        metropolis, M_Integrator, M_Logger, B_Logger, T_Logger,
        (1, 1, d.point_repeat, d.point_repeat, d.point_repeat),
        Route(metropolis => M_Integrator, :proposal => :Δvalue,
            transform = proposal -> accepteddelta(proposal)),
        Route(M_Integrator => M_Logger, :total => :value),
        Route(metropolis => B_Logger, :hamiltonian => :value, transform = x -> x.b[]),
        Route(metropolis => T_Logger, :model => :value, transform = temp),
    )

    anneal_part = CompositeAlgorithm(
        metro_t, anealing,
        (1, d.point_repeat),
        Route(metropolis => anealing, :model),
    )

    algorithm = Routine(anneal_part, (d.anneal_time,))
    return AnnealRun(algorithm, M_Integrator, M_Logger, B_Logger, T_Logger)
end

function build_pulse_process(g, p::ManuscriptParams; capture_dir = joinpath(p.outdir, "capture"))
    d = derived_params(p)
    metropolis = g.default_algorithm
    pulse = TrianglePulseA(p.Amp1, p.nrepeats)

    M_Integrator = Integrator(Float32, name = :M_integrator)
    M_Logger = Logger(Float32, name = :M_logger)
    B_Logger = ValueLogger(:b)
    Graph_Logger = ImageCapture(:Graph, -1.5, 1.5)

    metro_pulse = CompositeAlgorithm(
        metropolis, M_Integrator, M_Logger, B_Logger,
        (1, 1, d.point_repeat, d.point_repeat),
        Route(metropolis => M_Integrator, :proposal => :Δvalue,
            transform = proposal -> accepteddelta(proposal)),
        Route(M_Integrator => M_Logger, :total => :value),
        Route(metropolis => B_Logger, :hamiltonian => :value,
            transform = x -> x.b[]),
    )

    pulse_part = CompositeAlgorithm(
        metro_pulse, pulse, Graph_Logger,
        (1, d.point_repeat, d.capture_interval1),
        Route(metropolis => Graph_Logger, :model => :array, transform = graph_array),
    )
    relax_part = CompositeAlgorithm(
        metro_pulse, Graph_Logger,
        (1, d.capture_interval2),
        Route(metropolis => Graph_Logger, :model => :array, transform = graph_array),
    )
    algorithm = Routine(
        pulse_part, relax_part,
        (d.pulse_time, d.relax_time),
        Route(metropolis => pulse, :hamiltonian, :M),
    )

    return PulseRun(algorithm, M_Integrator, M_Logger, B_Logger, Graph_Logger, capture_dir)
end

function start_anneal!(g, run::AnnealRun; repeats = 1)
    return createProcess(g, run.algorithm, repeats = repeats)
end

function start_pulse!(g, run::PulseRun; repeats = 1)
    return createProcess(
        g,
        run.algorithm,
        Input(run.Graph_Logger, filepath = run.capture_dir),
        Input(run.M_Integrator, initialvalue = sum(graph_array(g)));
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
