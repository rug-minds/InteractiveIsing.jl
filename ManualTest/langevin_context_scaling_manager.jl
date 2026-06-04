include("langevin_context_scaling.jl")

"""
    RawLangevinBenchmarkStep

One manager-owned `ProcessAlgorithm` job for the raw Langevin scaling benchmark.
The process context owns the raw Langevin process context created from the same
helpers as `langevin_context_scaling.jl`; `step!` runs the no-merge hot loop.
"""
struct RawLangevinBenchmarkStep{A,T<:Integer,F<:Real,I<:Integer} <: StatefulAlgorithms.ProcessAlgorithm
    algorithm::A
    side::T
    fullsweeps::T
    temperature::F
    seed::I
end

"""
    StatefulAlgorithms.init(step::RawLangevinBenchmarkStep, context)

Build the manager-owned raw Langevin context for one worker process.
"""
function StatefulAlgorithms.init(step::RawLangevinBenchmarkStep, context)
    template = make_context_template(step.algorithm, step.side; temperature = step.temperature, seed = step.seed)
    nsteps = fullsweep_steps(template.context, template.instance, step.fullsweeps)
    return (; instance = template.instance, raw_context = template.context, nsteps)
end

"""
    StatefulAlgorithms.step!(step::RawLangevinBenchmarkStep, context)

Run one raw no-merge Langevin workload inside a manager-owned process.
"""
function StatefulAlgorithms.step!(step::RawLangevinBenchmarkStep, context)
    result = run_raw_langevin!(context.instance, context.raw_context, context.nsteps)
    return (; result)
end

"""
    benchmark_step(; kwargs...)

Construct the local raw Langevin `ProcessAlgorithm` used by manager workers.
"""
function benchmark_step(;
    fullsweeps::S,
    side::D,
    stepsize::E,
    temperature::F,
    seed::I,
) where {S<:Integer,D<:Integer,E<:Real,F<:Real,I<:Integer}
    algorithm = LocalLangevin(
        stepsize = Float32(stepsize),
        max_drift_fraction = 0.15f0,
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    return RawLangevinBenchmarkStep(algorithm, side, fullsweeps, Float32(temperature), seed)
end

"""
    build_manager(step, runs) -> (manager, results)

Build a `ProcessManager` whose worker contexts are created by manager-owned
`Process` construction.
"""
function build_manager(step::S, runs::R) where {S<:RawLangevinBenchmarkStep,R<:Integer}
    results = Vector{Any}(undef, Int(runs))
    recipe = (;
        makeworker = (idx, manager) -> Process(
            RawLangevinBenchmarkStep(
                step.algorithm,
                step.side,
                step.fullsweeps,
                step.temperature,
                step.seed + idx,
            );
            repeats = 1,
        ),
        prepare! = (slot, job, manager) -> resetworker!(slot),
        consume! = (slot, job, manager) -> begin
            instance = only(StatefulAlgorithms.getalgos(StatefulAlgorithms.getalgo(slot.worker)))
            results[Int(job)] = view(StatefulAlgorithms.context(slot.worker), instance).result
            return nothing
        end,
    )

    manager = ProcessManager(
        recipe;
        nworkers = Int(runs),
        flush_policy = NoFlush(),
        poll_interval = 0.0,
        job_type = Int,
    )
    return (; manager, results)
end

"""
    run_manager!(manager, runs) -> ProcessManager

Run one job per manager-owned worker slot.
"""
function run_manager!(manager::M, runs::R) where {M<:ProcessManager,R<:Integer}
    run!(manager, 1:Int(runs))
    return manager
end

"""
    print_manager_summary(; kwargs...) -> NamedTuple

Run the same raw Langevin scaling benchmark as `print_summary`, but compare the
plain `Threads.@spawn` path with a `ProcessManager`-scheduled path.
"""
function print_manager_summary(;
    runs::R = env_int("ISING_MANUAL_RUNS", 8),
    fullsweeps::S = env_int("ISING_MANUAL_FULLSWEEPS", 500),
    side::D = env_int("ISING_MANUAL_SIDE", 32),
    stepsize::E = env_float32("ISING_MANUAL_STEPSIZE", 0.02f0),
    temperature::F = env_float32("ISING_MANUAL_TEMP", 1.5f0),
    seed::I = env_int("ISING_MANUAL_SEED", 1),
) where {R<:Integer,S<:Integer,D<:Integer,E<:Real,F<:Real,I<:Integer}
    step = benchmark_step(; fullsweeps, side, stepsize, temperature, seed)
    algorithm = step.algorithm
    template = make_context_template(algorithm, side; temperature, seed)
    instance = template.instance
    base_context = template.context
    nsteps = fullsweep_steps(base_context, instance, fullsweeps)
    active_spins = length(view(base_context, instance).active_spins)

    println("ProcessManager raw Langevin step scaling")
    println("Julia threads:          ", Threads.nthreads())
    println("contexts/runs:          ", runs)
    println("graph side:             ", side, "x", side)
    println("active spins/context:   ", active_spins)
    println("full sweeps/context:    ", fullsweeps)
    println("raw steps/context:      ", nsteps)
    println("Langevin stepsize/temp: ", Float32(stepsize), " / ", Float32(temperature))
    println()

    # Pay compilation and first-cycle setup costs before timing both schedulers.
    run_raw_langevin!(instance, deepcopy(base_context), active_spins)
    run_threaded!(instance, [deepcopy(base_context)], active_spins)
    warmup_manager = build_manager(step, 1)
    run_manager!(warmup_manager.manager, 1)
    close(warmup_manager.manager)

    single_context = deepcopy(base_context)
    single_seconds, single_result = elapsed_seconds() do
        run_raw_langevin!(instance, single_context, nsteps)
    end

    serial_contexts = [deepcopy(base_context) for _ in 1:runs]
    serial_seconds, serial_results = elapsed_seconds() do
        run_serial!(instance, serial_contexts, nsteps)
    end

    threaded_contexts = [deepcopy(base_context) for _ in 1:runs]
    threaded_seconds, threaded_results = elapsed_seconds() do
        run_threaded!(instance, threaded_contexts, nsteps)
    end

    manager_setup = build_manager(step, runs)
    manager_seconds, _ = elapsed_seconds() do
        run_manager!(manager_setup.manager, runs)
    end
    manager_results = manager_setup.results
    close(manager_setup.manager)

    println("single context seconds: ", round(single_seconds, digits = 4))
    println("serial ", runs, " contexts sec: ", round(serial_seconds, digits = 4))
    println("threaded ", runs, " contexts:   ", round(threaded_seconds, digits = 4))
    println("manager ", runs, " contexts:    ", round(manager_seconds, digits = 4))
    println()
    println("serial / threaded:      ", round(serial_seconds / threaded_seconds, digits = 3), "x")
    println("serial / manager:       ", round(serial_seconds / manager_seconds, digits = 3), "x")
    println("manager / threaded:     ", round(manager_seconds / threaded_seconds, digits = 3), "x wall-time ratio")
    println("ideal threaded/single:  1.0x wall time, ", runs, "x throughput")
    println()
    println("single acceptance:      ", round(single_result.acceptance_rate, digits = 4))
    println("threaded acceptances:   ", round.(getproperty.(threaded_results, :acceptance_rate), digits = 4))
    println("manager acceptances:    ", round.(getproperty.(manager_results, :acceptance_rate), digits = 4))

    return (;
        single_seconds,
        serial_seconds,
        threaded_seconds,
        manager_seconds,
        single_result,
        serial_results,
        threaded_results,
        manager_results,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_manager_summary()
end
