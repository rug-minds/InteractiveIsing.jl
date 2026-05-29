include("langevin_context_scaling_manager.jl")

"""
    run_scheduled_manager!(setup, runs, schedule) -> Vector

Run manager-owned `Process` contexts with the new public threaded manager API.
The schedule is one of `Dynamic()`, `Static()`, or `Greedy()`.
"""
function run_scheduled_manager!(setup::Setup, runs::R, schedule::S) where {Setup,R<:Integer,S}
    runthreaded!(setup.manager, 1:Int(runs), schedule)
    return setup.results[1:Int(runs)]
end

"""
    print_manager_runthreaded_summary(; kwargs...) -> NamedTuple

Compare normal `ProcessManager` scheduling against the native threaded manager
runner added by Processes.jl.
"""
function print_manager_runthreaded_summary(;
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

    println("ProcessManager native runthreaded! Langevin scaling")
    println("Julia threads:          ", Threads.nthreads())
    println("contexts/runs:          ", runs)
    println("graph side:             ", side, "x", side)
    println("active spins/context:   ", active_spins)
    println("full sweeps/context:    ", fullsweeps)
    println("raw steps/context:      ", nsteps)
    println("Langevin stepsize/temp: ", Float32(stepsize), " / ", Float32(temperature))
    println()

    # Warm core hot loops and each manager mode before timing.
    run_raw_langevin!(instance, deepcopy(base_context), active_spins)
    warmup_normal = build_manager(step, 1)
    run_manager!(warmup_normal.manager, 1)
    close(warmup_normal.manager)
    warmup_dynamic = build_manager(step, max(Threads.maxthreadid(), 1))
    run_scheduled_manager!(warmup_dynamic, 1, Processes.Dynamic())
    close(warmup_dynamic.manager)
    warmup_static = build_manager(step, max(Threads.maxthreadid(), 1))
    run_scheduled_manager!(warmup_static, 1, Processes.Static())
    close(warmup_static.manager)
    warmup_greedy = build_manager(step, max(Threads.maxthreadid(), 1))
    run_scheduled_manager!(warmup_greedy, 1, Processes.Greedy())
    close(warmup_greedy.manager)

    serial_contexts = [deepcopy(base_context) for _ in 1:runs]
    serial_seconds, _ = elapsed_seconds() do
        run_serial!(instance, serial_contexts, nsteps)
    end

    plain_contexts = [deepcopy(base_context) for _ in 1:runs]
    plain_seconds, plain_results = elapsed_seconds() do
        run_threaded!(instance, plain_contexts, nsteps)
    end

    normal_setup = build_manager(step, runs)
    normal_seconds, _ = elapsed_seconds() do
        run_manager!(normal_setup.manager, runs)
    end
    normal_results = normal_setup.results
    close(normal_setup.manager)

    dynamic_setup = build_manager(step, runs)
    dynamic_seconds, dynamic_results = elapsed_seconds() do
        run_scheduled_manager!(dynamic_setup, runs, Processes.Dynamic())
    end
    close(dynamic_setup.manager)

    static_setup = build_manager(step, max(Int(runs), Threads.maxthreadid()))
    static_seconds, static_results = elapsed_seconds() do
        run_scheduled_manager!(static_setup, runs, Processes.Static())
    end
    close(static_setup.manager)

    greedy_setup = build_manager(step, runs)
    greedy_seconds, greedy_results = elapsed_seconds() do
        run_scheduled_manager!(greedy_setup, runs, Processes.Greedy())
    end
    close(greedy_setup.manager)

    println("serial ", runs, " contexts sec: ", round(serial_seconds, digits = 4))
    println("plain Threads.@spawn:   ", round(plain_seconds, digits = 4))
    println("normal ProcessManager:  ", round(normal_seconds, digits = 4))
    println("runthreaded Dynamic():  ", round(dynamic_seconds, digits = 4))
    println("runthreaded Static():   ", round(static_seconds, digits = 4))
    println("runthreaded Greedy():   ", round(greedy_seconds, digits = 4))
    println()
    println("serial / plain:         ", round(serial_seconds / plain_seconds, digits = 3), "x")
    println("serial / normal:        ", round(serial_seconds / normal_seconds, digits = 3), "x")
    println("serial / dynamic:       ", round(serial_seconds / dynamic_seconds, digits = 3), "x")
    println("serial / static:        ", round(serial_seconds / static_seconds, digits = 3), "x")
    println("serial / greedy:        ", round(serial_seconds / greedy_seconds, digits = 3), "x")
    println("dynamic / normal:       ", round(dynamic_seconds / normal_seconds, digits = 3), "x wall-time ratio")
    println("static / normal:        ", round(static_seconds / normal_seconds, digits = 3), "x wall-time ratio")
    println("greedy / normal:        ", round(greedy_seconds / normal_seconds, digits = 3), "x wall-time ratio")
    println()
    println("plain acceptances:      ", round.(getproperty.(plain_results, :acceptance_rate), digits = 4))
    println("normal acceptances:     ", round.(getproperty.(normal_results, :acceptance_rate), digits = 4))
    println("dynamic acceptances:    ", round.(getproperty.(dynamic_results, :acceptance_rate), digits = 4))
    println("static acceptances:     ", round.(getproperty.(static_results, :acceptance_rate), digits = 4))
    println("greedy acceptances:     ", round.(getproperty.(greedy_results, :acceptance_rate), digits = 4))

    return (;
        serial_seconds,
        plain_seconds,
        normal_seconds,
        dynamic_seconds,
        static_seconds,
        greedy_seconds,
        plain_results,
        normal_results,
        dynamic_results,
        static_results,
        greedy_results,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_manager_runthreaded_summary()
end
