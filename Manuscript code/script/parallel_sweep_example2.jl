# DSL-first sweep example.
#
# This file keeps route design in the experiment script. The `common/` folder
# only supplies graph construction, small algorithms, saving, and simple start/fetch
# helpers.

include(joinpath(@__DIR__, "common", "ManuscriptTools.jl"))
import InteractiveIsing
using InteractiveIsing.Processes

MT = ManuscriptTools

dsl_count(x) = max(1, round(Int, x))

###############################################################################
# Parameters
###############################################################################

base = MT.ManuscriptParams(;
    outdir = raw"D:\Code\data\Manuscript\Demo1_parallel_dsl",
    xL = 40,
    yL = 40,
    zL = 10,
    JIsing = 0.5,
    Scale = 1.0,
    Screening = 0.01,
    Temp = 0.5f0,
    Temp_aneal = 3f0,
    time_fctr = 1.0,
    Steps_1 = 6000,
    Amp1 = 10.0,
    nrepeats = 2,
    # nothing uses the old global proposer. A number uses LocalProposer(delta).
    proposal_delta = 0.1,
    algorithm_name = :metropolis,
    algorithm_kwargs = (;),
    a1 = -2.0,
    b1 = nothing,
    c1 = 10.0,
    landau_mode = :independent,
    landau_coeffs = nothing,
)

###############################################################################
# Small reusable DSL component: Integrator + Logger
###############################################################################

function IntegrateAndLog(type = Float64, loginterval = 1)
    integrator = Integrator(type, name = :integrate_and_log)
    logger = Logger(type, name = :integrate_and_log)

    c = @CompositeAlgorithm begin
        @alias integrator = integrator
        @alias logger = logger

        total = integrator()
        @every loginterval logger(value = @transform(x -> x[], total))
    end

    return package(c)
end

struct PackagedPulseRun
    algorithm
    M_Integrate_and_Logger
    B_Logger
    Graph_Logger
    capture_dir::String
end

struct PackagedAnnealRun
    algorithm
    M_Integrate_and_Logger
    B_Logger
    T_Logger
end

###############################################################################
# Example 1: pulse route using your collaborator's packaged DSL style
###############################################################################

function make_packaged_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"), capture = false)
    d = MT.derived_params(p)
    point_repeat = dsl_count(d.point_repeat)
    capture_interval1 = dsl_count(d.capture_interval1)
    capture_interval2 = dsl_count(d.capture_interval2)
    pulse_time = dsl_count(d.pulse_time)
    relax_time = dsl_count(d.relax_time)
    dynamics = MT.select_dynamics(g, p)
    pulse1 = MT.TrianglePulseA(p.Amp1, p.nrepeats)
    M_Integrate_and_Logger = IntegrateAndLog(Float32, point_repeat)
    B_Logger = MT.ValueLogger(:b)
    Graph_Logger = capture ? MT.ImageCapture(:Graph, -1.5, 1.5) : nothing

    Metro_Pulse = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias M_Integrate_and_Logger = M_Integrate_and_Logger
        @alias B_Logger = B_Logger

        proposal = dynamics()
        M_Integrate_and_Logger(
            Δvalue = @transform(MT.accepted_proposal_delta, proposal),
        )
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
    end

    if capture
        pulse_part = @CompositeAlgorithm begin
            @alias pulse1 = pulse1
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = Metro_Pulse()

            @every point_repeat pulse1(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
            @every capture_interval1 Graph_Logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end

        relax_part = @CompositeAlgorithm begin
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = Metro_Pulse()

            @every capture_interval2 Graph_Logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end
    else
        pulse_part = @CompositeAlgorithm begin
            @alias pulse1 = pulse1
            @context metro_pulse = Metro_Pulse()

            @every point_repeat pulse1(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
        end

        relax_part = @CompositeAlgorithm begin
            @context metro_pulse = Metro_Pulse()
        end
    end

    algorithm = @Routine begin
        @repeat pulse_time pulse_part()
        @repeat relax_time relax_part()
    end

    return PackagedPulseRun(algorithm, M_Integrate_and_Logger, B_Logger, Graph_Logger, capture_dir)
end

function start_packaged_pulse!(g, run::PackagedPulseRun; repeats = 1)
    inputs = isnothing(run.Graph_Logger) ?
        (Init(run.M_Integrate_and_Logger, initialvalue = sum(MT.graph_array(g))),) :
        (
            Init(run.Graph_Logger, filepath = run.capture_dir),
            Init(run.M_Integrate_and_Logger, initialvalue = sum(MT.graph_array(g))),
        )

    return InteractiveIsing.createProcess(
        g,
        run.algorithm,
        inputs...;
        repeats,
    )
end

function fetch_packaged_run(process, run::PackagedPulseRun)
    context = fetch(process)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Integrate_and_Logger].log,
    )
end

###############################################################################
# Example 2: anneal route using the same packaged IntegrateAndLog style
###############################################################################

function make_packaged_anneal_run(g, p)
    d = MT.derived_params(p)
    point_repeat = dsl_count(d.point_repeat)
    anneal_time = dsl_count(d.anneal_time)
    dynamics = MT.select_dynamics(g, p)
    annealing = MT.LinAnealingB(p.Temp_aneal, 0f0)
    M_Integrate_and_Logger = IntegrateAndLog(Float32, point_repeat)
    B_Logger = MT.ValueLogger(:b)
    T_Logger = MT.ValueLogger(:T)

    Metro_T = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias M_Integrate_and_Logger = M_Integrate_and_Logger
        @alias B_Logger = B_Logger
        @alias T_Logger = T_Logger

        proposal = dynamics()
        M_Integrate_and_Logger(
            Δvalue = @transform(MT.accepted_proposal_delta, proposal),
        )
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

    return PackagedAnnealRun(algorithm, M_Integrate_and_Logger, B_Logger, T_Logger)
end

function start_packaged_anneal!(g, run::PackagedAnnealRun; repeats = 1)
    return InteractiveIsing.createProcess(
        g,
        run.algorithm,
        Init(run.M_Integrate_and_Logger, initialvalue = sum(MT.graph_array(g)));
        repeats,
    )
end

function fetch_packaged_run(process, run::PackagedAnnealRun)
    context = fetch(process)
    return (;
        context,
        voltage = context[run.B_Logger].values,
        Pr = context[run.M_Integrate_and_Logger].log,
        Temp = context[run.T_Logger].values,
    )
end

###############################################################################
# Example 3: split Integrator + Logger DSL fallback
###############################################################################

function make_split_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"), capture = false)
    parts = MT.pulse_components(g, p; capture_dir)
    (; d, dynamics, pulse, M_Integrator, M_Logger, B_Logger, Graph_Logger) = parts
    Graph_Logger = capture ? Graph_Logger : nothing
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
            Δvalue = @transform(MT.accepted_proposal_delta, proposal),
        )
        @every point_repeat M_Logger(value = total)
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
    end

    if capture
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = Metro_Pulse()

            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
            @every capture_interval1 Graph_Logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end

        relax_part = @CompositeAlgorithm begin
            @alias Graph_Logger = Graph_Logger
            @context metro_pulse = Metro_Pulse()

            @every capture_interval2 Graph_Logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end
    else
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @context metro_pulse = Metro_Pulse()

            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
        end

        relax_part = @CompositeAlgorithm begin
            @context metro_pulse = Metro_Pulse()
        end
    end

    algorithm = @Routine begin
        @repeat pulse_time pulse_part()
        @repeat relax_time relax_part()
    end

    return MT.PulseRun(algorithm, parts.M_Integrator, parts.M_Logger, parts.B_Logger, Graph_Logger, parts.capture_dir)
end

###############################################################################
# Running one experiment
###############################################################################

function run_one_packaged_pulse(p)
    g = MT.build_graph(p)
    run = make_packaged_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"))
    process = start_packaged_pulse!(g, run; repeats = 1)
    result = fetch_packaged_run(process, run)
    paths = MT.save_run_outputs(g, p; pulse = result)
    return (; graph = g, run, result, paths)
end

function run_one_split_pulse(p)
    g = MT.build_graph(p)
    run = make_split_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"))
    process = MT.start_pulse!(g, run; repeats = 1)
    result = MT.fetch_run(process, run)
    paths = MT.save_run_outputs(g, p; pulse = result)
    return (; graph = g, run, result, paths)
end

###############################################################################
# Parallel sweep patterns
###############################################################################

function screening_paramsets(base)
    screenings = (0.005, 0.01, 0.05, 0.1)
    return [
        MT.update_params(
            base;
            Screening = screening,
            outdir = joinpath(base.outdir, "screening=$(screening)"),
        )
        for screening in screenings
    ]
end

function proposal_delta_paramsets(base)
    deltas = (0.05, 0.1, 0.2, 0.5, nothing)
    return [
        MT.update_params(
            base;
            proposal_delta = delta,
            outdir = joinpath(base.outdir, "proposal_delta=$(isnothing(delta) ? "global" : delta)"),
        )
        for delta in deltas
    ]
end

function algorithm_paramsets(base)
    configs = [
        (name = "metropolis", algorithm_name = :metropolis, algorithm_kwargs = (;)),
        (name = "local_langevin", algorithm_name = :local_langevin, algorithm_kwargs = (; stepsize = 0.05f0, adjusted = true)),
        (name = "global_langevin", algorithm_name = :global_langevin, algorithm_kwargs = (; stepsize = 0.01f0, adjusted = false)),
        (name = "block_langevin", algorithm_name = :block_langevin, algorithm_kwargs = (; stepsize = 0.02f0, block_size = 128, adjusted = false)),
    ]

    return [
        MT.update_params(
            base;
            algorithm_name = item.algorithm_name,
            algorithm_kwargs = item.algorithm_kwargs,
            outdir = joinpath(base.outdir, item.name),
        )
        for item in configs
    ]
end

function start_all_packaged_pulse_sweep(paramsets; capture = false)
    runs = Any[]
    for p in paramsets
        g = MT.build_graph(p)
        run = make_packaged_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"), capture)
        process = start_packaged_pulse!(g, run; repeats = 1)
        push!(runs, (; params = p, graph = g, run, process))
    end
    return runs
end

function fetch_packaged_pulse_sweep(runs)
    return map(runs) do item
        result = fetch_packaged_run(item.process, item.run)
        paths = MT.save_run_outputs(item.graph, item.params; pulse = result)
        (; item.params, item.graph, item.run, item.process, result, paths)
    end
end

function run_packaged_pulse_sweep_batched(paramsets; max_inflight = Threads.nthreads(), capture = false)
    finished = Any[]

    for first in 1:max_inflight:length(paramsets)
        last = min(first + max_inflight - 1, length(paramsets))
        batch = paramsets[first:last]
        running = start_all_packaged_pulse_sweep(batch; capture)
        append!(finished, fetch_packaged_pulse_sweep(running))
    end

    return finished
end

###############################################################################
# Choose one entry point
###############################################################################

if abspath(PROGRAM_FILE) == @__FILE__
    # One run:
    # run_one_packaged_pulse(base)
    # run_one_split_pulse(base)

    # Start every process immediately. If there are more runs than CPU cores,
    # the OS will time-slice them; Julia/Processes will not automatically cap
    # concurrency for you.
    # paramsets = screening_paramsets(base)
    # paramsets = proposal_delta_paramsets(base)
    # paramsets = algorithm_paramsets(base)
    # runs = start_all_packaged_pulse_sweep(paramsets)
    # fetch_packaged_pulse_sweep(runs)

    # Manual queue. This is usually safer for large graphs because memory and
    # FFT plans can be the real bottleneck before CPU core count.
    paramsets = screening_paramsets(base)
    run_packaged_pulse_sweep_batched(paramsets; max_inflight = 4)
end
