include("langevin_context_scaling_manager.jl")

"""
    run_threads_manager!(manager, runs; schedule = :dynamic) -> Vector

Run manager-owned worker processes with `Threads.@threads` instead of the
manager's dynamic task/poll scheduler. This is a fixed-wave diagnostic: job `i`
is assigned to slot `i`, so it is only appropriate for homogeneous batches with
at most one job per worker slot.
"""
function run_threads_manager!(
    manager::M,
    runs::R;
    schedule::Symbol = :dynamic,
) where {M<:ProcessManager,R<:Integer}
    selected_slots = collect(Iterators.take(slots(manager), Int(runs)))
    results = Vector{Any}(undef, length(selected_slots))

    if schedule === :static
        Threads.@threads :static for idx in eachindex(selected_slots)
            slot = selected_slots[idx]
            resetworker!(slot)
            runprocessinline!(slot.worker)
            instance = only(StatefulAlgorithms.getalgos(StatefulAlgorithms.getalgo(slot.worker)))
            results[idx] = view(StatefulAlgorithms.context(slot.worker), instance).result
        end
    elseif schedule === :dynamic
        Threads.@threads for idx in eachindex(selected_slots)
            slot = selected_slots[idx]
            resetworker!(slot)
            runprocessinline!(slot.worker)
            instance = only(StatefulAlgorithms.getalgos(StatefulAlgorithms.getalgo(slot.worker)))
            results[idx] = view(StatefulAlgorithms.context(slot.worker), instance).result
        end
    else
        throw(ArgumentError("schedule must be :dynamic or :static, got $(schedule)"))
    end

    return results
end

"""
    print_manager_threads_summary(; kwargs...) -> NamedTuple

Compare the normal `ProcessManager` scheduler against a fixed-wave
`Threads.@threads` runner over manager-owned process contexts.
"""
function print_manager_threads_summary(;
    runs::R = env_int("ISING_MANUAL_RUNS", 8),
    fullsweeps::S = env_int("ISING_MANUAL_FULLSWEEPS", 500),
    side::D = env_int("ISING_MANUAL_SIDE", 32),
    stepsize::E = env_float32("ISING_MANUAL_STEPSIZE", 0.02f0),
    temperature::F = env_float32("ISING_MANUAL_TEMP", 1.5f0),
    seed::I = env_int("ISING_MANUAL_SEED", 1),
) where {R<:Integer,S<:Integer,D<:Integer,E<:Real,F<:Real,I<:Integer}
    step = benchmark_step(; fullsweeps, side, stepsize, temperature, seed)
    template = make_context_template(step.algorithm, side; temperature, seed)
    instance = template.instance
    base_context = template.context
    nsteps = fullsweep_steps(base_context, instance, fullsweeps)
    active_spins = length(view(base_context, instance).active_spins)

    println("ProcessManager Threads.@threads Langevin scaling")
    println("Julia threads:          ", Threads.nthreads())
    println("contexts/runs:          ", runs)
    println("graph side:             ", side, "x", side)
    println("active spins/context:   ", active_spins)
    println("full sweeps/context:    ", fullsweeps)
    println("raw steps/context:      ", nsteps)
    println("Langevin stepsize/temp: ", Float32(stepsize), " / ", Float32(temperature))
    println()

    # Warm all execution paths before timing.
    run_raw_langevin!(instance, deepcopy(base_context), active_spins)
    warmup_manager = build_manager(step, 1)
    run_manager!(warmup_manager.manager, 1)
    run_threads_manager!(warmup_manager.manager, 1; schedule = :dynamic)
    run_threads_manager!(warmup_manager.manager, 1; schedule = :static)
    close(warmup_manager.manager)

    serial_contexts = [deepcopy(base_context) for _ in 1:runs]
    serial_seconds, _ = elapsed_seconds() do
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

    dynamic_setup = build_manager(step, runs)
    dynamic_seconds, dynamic_results = elapsed_seconds() do
        run_threads_manager!(dynamic_setup.manager, runs; schedule = :dynamic)
    end
    close(dynamic_setup.manager)

    static_setup = build_manager(step, runs)
    static_seconds, static_results = elapsed_seconds() do
        run_threads_manager!(static_setup.manager, runs; schedule = :static)
    end
    close(static_setup.manager)

    println("serial ", runs, " contexts sec:      ", round(serial_seconds, digits = 4))
    println("plain threaded contexts:    ", round(threaded_seconds, digits = 4))
    println("normal ProcessManager:      ", round(manager_seconds, digits = 4))
    println("manager Threads.@threads:   ", round(dynamic_seconds, digits = 4))
    println("manager @threads :static:   ", round(static_seconds, digits = 4))
    println()
    println("serial / threaded:          ", round(serial_seconds / threaded_seconds, digits = 3), "x")
    println("serial / manager:           ", round(serial_seconds / manager_seconds, digits = 3), "x")
    println("serial / manager @threads:  ", round(serial_seconds / dynamic_seconds, digits = 3), "x")
    println("serial / manager static:    ", round(serial_seconds / static_seconds, digits = 3), "x")
    println("manager/threaded ratio:     ", round(manager_seconds / threaded_seconds, digits = 3), "x")
    println("@threads/manager ratio:     ", round(dynamic_seconds / manager_seconds, digits = 3), "x")
    println("static/manager ratio:       ", round(static_seconds / manager_seconds, digits = 3), "x")
    println()
    println("threaded acceptances:       ", round.(getproperty.(threaded_results, :acceptance_rate), digits = 4))
    println("manager acceptances:        ", round.(getproperty.(manager_results, :acceptance_rate), digits = 4))
    println("@threads acceptances:       ", round.(getproperty.(dynamic_results, :acceptance_rate), digits = 4))
    println("static acceptances:         ", round.(getproperty.(static_results, :acceptance_rate), digits = 4))

    return (;
        serial_seconds,
        threaded_seconds,
        manager_seconds,
        dynamic_seconds,
        static_seconds,
        threaded_results,
        manager_results,
        dynamic_results,
        static_results,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_manager_threads_summary()
end
