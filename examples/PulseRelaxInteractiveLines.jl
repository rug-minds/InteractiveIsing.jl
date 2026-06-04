using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using InteractiveIsing.Windows
using Statistics

function weightfunc_shell(ax, ay, az, csr, lambda1, lambda2; dc)
    dx, dy, dz = dc
    r2 = (ax * dx)^2 + (ay * dy)^2 + (az * dz)^2
    r2 == 0 && return 0.0f0

    shell = dx * dx + dy * dy + dz * dz
    k1 = 1.0f0
    k2 = Float32(lambda1) * k1
    k3 = Float32(lambda2) * k2
    short_range = shell == 1 ? k1 : shell == 2 ? k2 : shell == 3 ? k3 : 0.0f0
    return Float32(csr) * short_range
end

struct TrianglePulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end

function StatefulAlgorithms.init(tp::TrianglePulseA, args)
    n_calls = num_calls(args)
    samples = max(1, round(Int, n_calls / (4 * tp.numpulses)))
    pulse = repeat(vcat(
        LinRange(0, tp.amp, samples),
        LinRange(tp.amp, 0, samples),
        LinRange(0, -tp.amp, samples),
        LinRange(-tp.amp, 0, samples),
    ), tp.numpulses)
    length(pulse) < n_calls && append!(pulse, zeros(eltype(pulse), n_calls - length(pulse)))
    return (; pulse, step = 1, pulseval = pulse[1])
end

function StatefulAlgorithms.step!(::TrianglePulseA, context)
    (; pulse, step, hamiltonian) = context
    pulseval = pulse[min(step, length(pulse))]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()

function StatefulAlgorithms.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    return (; values)
end

function StatefulAlgorithms.step!(::ValueLogger, context)
    (; values, value) = context
    push!(values, Float32(value))
    return (;)
end

struct StepLogger{Name} <: ProcessAlgorithm end
StepLogger(name) = StepLogger{Symbol(name)}()

function StatefulAlgorithms.init(::StepLogger, args)
    steps = Int[]
    processsizehint!(steps, args)
    return (; steps)
end

function StatefulAlgorithms.step!(::StepLogger, context)
    (; steps) = context
    push!(steps, length(steps) + 1)
    return (;)
end

struct YieldControl <: ProcessAlgorithm
    seconds::Float64
end

StatefulAlgorithms.init(::YieldControl, args) = (;)

function StatefulAlgorithms.step!(yield_control::YieldControl, context)
    sleep(yield_control.seconds)
    return (;)
end

function integrate_and_log(type = Float32, log_interval = 1)
    integrator = Integrator(type, name = :integrate_and_log)
    logger = Logger(type, name = :integrate_and_log)
    return package(@CompositeAlgorithm begin
        @alias integrator = integrator
        @alias logger = logger

        total = integrator()
        @every log_interval logger(value = @transform(x -> x[], total))
    end)
end

function normalize_adj_by_average_col!(g, scaling = 1.0f0)
    A = adj(g).sp
    avg_col_sum = mean(sum(abs, @view A[:, j]) for j in axes(A, 2))
    avg_col_sum == 0 && return g
    A .*= scaling / avg_col_sum
    return g
end

xL, yL, zL = 16, 16, 6
scale = 1.0f0
screening = 5.0f0
temp0 = 0.1f0

a1, c1 = -2.0f0, 10.0f0
b1 = -(a1 + 3c1) / 2

wg = @WG (; dc) -> weightfunc_shell(1, 1, 1, 1, 0.1, 0.1; dc) NN = 3

g = IsingGraph(
    xL, yL, zL,
    Continuous(),
    wg,
    LatticeConstants(1.0f0, 1.0f0, 1.0f0),
    Ising(b = UniformArray(0)) +
        CoulombHamiltonian(scaling = scale, screening = screening, recalc = 1000) +
        Quartic(c = b1 / a1) +
        Sextic(c = c1 / a1),
    StateSet(-1.5f0, 1.5f0);
    periodic = (:x, :y),
    diag = StateLike(UniformArray),
    precision = Float32,
)
normalize_adj_by_average_col!(g, 1.0f0)
adj(g)[1, 1] = a1
temp!(g, temp0)

simulation_host = interface(g)

amp = 1.1f0
nrepeats = 2
steps_per_sweep = nstates(g)
log_interval = steps_per_sweep
ui_interval = max(1, steps_per_sweep ÷ 8)
pulse_time = 40 * steps_per_sweep
relax_time = 20 * steps_per_sweep

pulse = TrianglePulseA(amp, nrepeats)
dynamics = LocalLangevin(stepsize = 0.05f0, adjusted = true)
polarization_logger = integrate_and_log(Float32, log_interval)
voltage_logger = ValueLogger(:voltage)
step_logger = StepLogger(:step)
ui_yield = YieldControl(0.001)

metro_pulse = @CompositeAlgorithm begin
    @alias dynamics = dynamics

    proposal = @every 1 dynamics()
    @every 1 polarization_logger(Δvalue = @transform(proposal -> accepteddelta(proposal), proposal))
    @every log_interval voltage_logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
    @every log_interval step_logger()
    @every ui_interval ui_yield()
end

pulse_part = @CompositeAlgorithm begin
    @context metro = metro_pulse()
    @every log_interval pulse(hamiltonian = metro.dynamics.hamiltonian)
end

relax_part = @CompositeAlgorithm begin
    @context metro = metro_pulse()
end

pulse_and_relax = @Routine begin
    @repeat pulse_time pulse_part()
    @repeat relax_time relax_part()
end

StatefulAlgorithms.close(g)
process_func = deepcopy(pulse_and_relax)
process_inputs = (
    InteractiveIsing._mc_model_inits(process_func, g)...,
    Init(polarization_logger, initialvalue = sum(state(g))),
)
process = Process(
    process_func,
    process_inputs...;
    repeats = 1,
)
push!(processes(g), process)

trace_host = window(title = "Pulse Trace", size = (1100, 800), fps = 30)
panel!(
    trace_host,
    ContextLinesPanel(
        process,
        voltage_logger => :values,
        polarization_logger => :log;
        xlabel = "Voltage",
        ylabel = "Polarization",
        title = "Polarization vs voltage",
        line_kwargs = (; color = :dodgerblue, linewidth = 2),
    ),
    (1, 1),
)
panel!(
    trace_host,
    ContextLinesPanel(
        process,
        step_logger => :steps,
        polarization_logger => :log;
        xlabel = "Logged step",
        ylabel = "Polarization",
        title = "Polarization trace",
        line_kwargs = (; color = :darkorange, linewidth = 2),
    ),
    (2, 1),
)

run(process)
