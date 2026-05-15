using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "simple_2_4_1_curriculum.jl"))

using IsingLearning.InteractiveIsing.Processes:
    @Routine, @CompositeAlgorithm, @repeat, @state, @alias, @context

const Processes = II.Processes

"""
    AnnealedLocalLangevin(inner, start_T, stop_T, nsteps, power)

Experiment-local sampler wrapper. Each call writes a power-law temperature
schedule to `context.model`, then delegates one step to `inner`.

The counter cycles every `nsteps`, so repeated worker reuse starts a fresh
nudged anneal for each plus/minus branch without needing toolbox changes.
"""
struct AnnealedLocalLangevin{Side,L,T<:Real} <: II.IsingMCAlgorithm
    inner::L
    start_T::T
    stop_T::T
    nsteps::Int
    power::T
end

AnnealedLocalLangevin(side::Symbol, inner, start_T::T, stop_T::T, nsteps::Integer, power::T) where {T<:Real} =
    AnnealedLocalLangevin{side,typeof(inner),T}(inner, start_T, stop_T, Int(nsteps), power)

"""
    annealed_temperature(step, total, start_T, stop_T, power)

Return the scheduled temperature for a zero-based relaxation `step`.
"""
function annealed_temperature(step::Integer, total::Integer, start_T, stop_T, power)
    total = max(total, 1)
    progress = total == 1 ? one(float(start_T)) : float(step) / float(total - 1)
    progress = clamp(progress, zero(progress), one(progress))
    return stop_T + (start_T - stop_T) * (one(progress) - progress)^power
end

"""
    II.Processes.init(annealed, context)

Initialize the wrapped `LocalLangevin` context and add an annealing counter.
"""
function II.Processes.init(annealed::AnnealedLocalLangevin, context)
    base = II.Processes.init(annealed.inner, context)
    return merge(base, (; anneal_step = Ref(0)))
end

"""
    II.Processes.step!(annealed, context)

Set the current scheduled temperature, advance the cyclic annealing counter,
and perform one wrapped `LocalLangevin` step.
"""
function II.Processes.step!(annealed::AnnealedLocalLangevin, context)
    phase_step = context.anneal_step[] % max(annealed.nsteps, 1)
    Tcur = annealed_temperature(phase_step, annealed.nsteps, annealed.start_T, annealed.stop_T, annealed.power)
    II.temp!(context.model, Tcur)
    context.anneal_step[] += 1
    result = II.Processes.step!(annealed.inner, context)
    return merge(result, (; scheduled_T = Tcur))
end

"""
    anneal_start_temp(config)

Read the nudged-phase start temperature for this experiment.
"""
anneal_start_temp(config::Analytic241Config) =
    parse(FT, get(ENV, "ISING_241_NUDGED_START_TEMP", string(FT(1.5) * config.temp)))

"""
    anneal_stop_temp(config)

Read the nudged-phase stop temperature for this experiment.
"""
anneal_stop_temp(config::Analytic241Config) =
    parse(FT, get(ENV, "ISING_241_NUDGED_STOP_TEMP", string(config.temp)))

"""
    anneal_power()

Read the power-law curvature for nudged-phase annealing.
"""
anneal_power() = parse(FT, get(ENV, "ISING_241_NUDGED_ANNEAL_POWER", "1.0"))

function layer_241(graph, config::Analytic241Config)
    dynamics = II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    return LayeredIsingGraphLayer(
        () -> graph_241(config);
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_relaxation,
        free_relaxation_steps = config.free_relaxation,
        nudged_relaxation_steps = config.nudged_relaxation,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""
    annealed_nudged_dynamics(layer, config)

Build plus/minus nudged branches that use the same `dynamics` context as the
free phase. This matters because the Langevin context owns Hamiltonian caches;
using separate nudged contexts after `setgraph!` can leave those caches out of
sync with the restored free state.
"""
function annealed_nudged_dynamics(layer, config::Analytic241Config)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    warm_steps = clamp(
        round(Int, parse(FT, get(ENV, "ISING_241_NUDGED_WARM_FRACTION", "0.2")) * relaxation_steps),
        1,
        max(relaxation_steps - 1, 1),
    )
    cool_steps = relaxation_steps - warm_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    start_T = anneal_start_temp(config)
    stop_T = anneal_stop_temp(config)

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, beta)
        II.temp!(dynamics.model, start_T)
        model = @repeat warm_steps dynamics()
        II.temp!(dynamics.model, stop_T)
        model = @repeat cool_steps dynamics()
        II.temp!(dynamics.model, config.temp)
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, -beta)
        II.temp!(dynamics.model, start_T)
        model = @repeat warm_steps dynamics()
        II.temp!(dynamics.model, stop_T)
        model = @repeat cool_steps dynamics()
        II.temp!(dynamics.model, config.temp)
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture)
end

"""
    forward_and_annealed_nudged(layer, config)

Compose the standard free phase with annealed plus/minus nudged branches and
the usual `(plus - minus) / (2β)` contrastive gradient.
"""
function forward_and_annealed_nudged(layer, config::Analytic241Config)
    forward = IsingLearning.ForwardDynamics(layer).algorithm
    nudged = annealed_nudged_dynamics(layer, config)
    beta = layer.β
    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = nudged.algorithm()
        IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(
            c1.dynamics.model,
            c2.plus_capture.captured,
            c2.minus_capture.captured,
            beta,
            buffers = buffers,
        )
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture)
end

"""
    annealed_worker_process(layer, worker_graph, config)

Create one training worker for the annealed nudged curriculum.
"""
function annealed_worker_process(layer, worker_graph, config::Analytic241Config)
    algo = II.Processes.resolve(forward_and_annealed_nudged(layer, config).algorithm)
    xdim = length(layer.input_layer)
    ydim = length(layer.output_layer)
    buffers = IsingLearning.gradient_buffer(worker_graph)
    return II.Processes.Process(
        algo,
        II.Processes.Init(:_state;
            x = zeros(eltype(worker_graph), xdim),
            y = zeros(eltype(worker_graph), ydim),
            buffers = buffers,
            equilibrium_state = copy(II.state(worker_graph)),
        ),
        II.Processes.Init(:dynamics, model = worker_graph),
        II.Processes.Init(:plus_capture, state = worker_graph),
        II.Processes.Init(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

"""
    trainer_241(config)

Build a normal `MNISTThreadedTrainer`, but use the annealed nudged worker
process for training.
"""
function trainer_241(config::Analytic241Config)
    graph = graph_241(config)
    config.init === :analytic && apply_corner_detector_solution!(graph, config)
    layer = layer_241(graph, config)
    optimiser = Optimisers.Adam(config.lr)
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)

    worker_template_graph = IsingLearning._worker_graph(graph, params)
    worker_template = annealed_worker_process(layer, worker_template_graph, config)
    workers = [worker_template]
    worker_graphs = [Processes.context(worker).dynamics.model for worker in workers]

    validation_template_graph = IsingLearning._worker_graph(graph, params)
    validation_worker = IsingLearning._validation_process(layer, validation_template_graph)
    validation_graph = Processes.context(validation_worker).dynamics.model

    return IsingLearning.MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        worker_graphs,
        workers,
        validation_graph,
        validation_worker,
        optimiser,
    )
end

"""
    run_annealed_curriculum()

Run the scalar curriculum with mild nudged-phase annealing and write an
experiment README that records the annealing settings.
"""
function run_annealed_curriculum()
    result = run_curriculum()
    open(joinpath(result.outdir, "README.md"), "a") do io
        println(io)
        println(io, "## Nudged-Phase Annealing")
        println(io)
        println(io, "- start temperature = `$(get(ENV, "ISING_241_NUDGED_START_TEMP", "1.5*T"))`")
        println(io, "- stop temperature = `$(get(ENV, "ISING_241_NUDGED_STOP_TEMP", "T"))`")
        println(io, "- power = `$(get(ENV, "ISING_241_NUDGED_ANNEAL_POWER", "1.0"))`")
        println(io)
        println(io, "Only the plus/minus nudged branches use this schedule. The free and validation")
        println(io, "branches use the base `T` from the run config.")
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_annealed_curriculum()
end
