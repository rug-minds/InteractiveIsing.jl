using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Random
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_MANUAL_CONTEXT_WORKERS", "16,32"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_SWEEPS", "500.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_STEPSIZE", "0.5"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_MANUAL_CONTEXT_WEIGHT_SCALE", "0.005"))
const OUTDIR = get(ENV, "ISING_MNIST_MANUAL_CONTEXT_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_manual_contexts")))

mkpath(OUTDIR)

"""
    append_csv_row!(path, row)

Append a named-tuple row to a CSV file, creating the header on first use.
"""
function append_csv_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    active_units(graph)

Return hidden plus output units. The input layer is fixed/off in the manual
MNIST context test, matching the manager's relaxation active set.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    build_prototype()

Build the 10x-hidden MNIST graph/layer and one normalized MNIST input vector.
Only the layer is used for data normalization; no `Process` or manager is built.
"""
function build_prototype()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(70_000),
    )
    temp!(graph, TEMP)
    layer = MNISTLayer(graph = graph)
    x, _ = load_mnist_arrays(layer; split = :train, limit = 1)
    relaxation_steps = max(1, round(Int, SWEEPS * active_units(graph)))
    return (; graph, x = copy(view(x, :, 1)), relaxation_steps)
end

"""
    build_contexts(prototype, langevin, ncontexts)

Create `ncontexts` independent graph copies and initialize one raw
`LocalLangevin` context per graph. This deliberately avoids `Process` and
`ProcessManager`; the only runtime call in the measured loop is
`Processes.step!(langevin, context)`.
"""
function build_contexts(prototype::P, langevin::L, ncontexts::Integer) where {P,L}
    n = Int(ncontexts)
    build_start = time_ns()
    first_graph = deepcopy(prototype.graph)
    temp!(first_graph, TEMP)
    IsingLearning.apply_input(first_graph, prototype.x)
    first_context = Processes.init(langevin, (; model = first_graph))
    contexts = Vector{typeof(first_context)}(undef, n)
    contexts[1] = first_context
    for idx in 2:n
        graph = deepcopy(prototype.graph)
        temp!(graph, TEMP)
        IsingLearning.apply_input(graph, prototype.x)
        contexts[idx] = Processes.init(langevin, (; model = graph))
    end
    build_seconds = (time_ns() - build_start) / 1.0e9
    return contexts, build_seconds
end

"""
    run_steps!(langevin, context, nsteps)

Run the raw Langevin step loop for one initialized context. `step!` mutates the
context-owned graph and cached arrays; its small diagnostics return value is not
the context itself.
"""
function run_steps!(langevin::L, context::C, nsteps::Integer) where {L,C}
    last_result = nothing
    for _ in 1:Int(nsteps)
        last_result = Processes.step!(langevin, context)
    end
    return last_result
end

"""
    timed_run_steps!(langevin, context, nsteps)

Measure one raw Langevin step loop in a regular function. Keeping `@elapsed`
outside the spawned expression lets the caller use dollar interpolation directly
with `Threads.@spawn`.
"""
function timed_run_steps!(langevin::L, context::C, nsteps::Integer) where {L,C}
    result = Ref{Any}(nothing)
    seconds = @elapsed result[] = run_steps!(langevin, context, nsteps)
    return (; seconds, result = result[])
end

"""
    timed_spawn_run!(langevin, context, nsteps, idx)

Run one timed raw-context loop and attach the original context index to the
result returned by the spawned task.
"""
function timed_spawn_run!(langevin::L, context::C, nsteps::Integer, idx::Integer) where {L,C}
    timed = timed_run_steps!(langevin, context, nsteps)
    return (; idx = Int(idx), timed.seconds, timed.result)
end

"""
    run_serial_reference!(langevin, context, nsteps)

Measure one context on the current task as the single-relaxation baseline.
"""
function run_serial_reference!(langevin::L, context::C, nsteps::Integer) where {L,C}
    return timed_run_steps!(langevin, context, nsteps)
end

"""
    run_spawned_contexts!(langevin, contexts, nsteps)

Spawn one task per context and run only raw `step!` loops inside those tasks.
"""
function run_spawned_contexts!(langevin::L, contexts::C, nsteps::Integer) where {L,C<:AbstractVector}
    task_seconds = zeros(Float64, length(contexts))
    total_seconds = @elapsed begin
        tasks = map(eachindex(contexts)) do idx
            Threads.@spawn timed_spawn_run!($langevin, $(contexts[idx]), $nsteps, $idx)
        end
        for task in tasks
            output = fetch(task)
            task_seconds[output.idx] = output.seconds
        end
    end
    return (; total_seconds, task_seconds)
end

"""
    run_config(ncontexts)

Run one manual raw-context scheduling probe for `ncontexts`.
"""
function run_config(ncontexts::Integer)
    prototype = build_prototype()
    langevin = LocalLangevin(stepsize = STEPSIZE, adjusted = false)

    serial_contexts, serial_build_seconds = build_contexts(prototype, langevin, 1)
    run_steps!(langevin, only(serial_contexts), 1)
    serial = run_serial_reference!(langevin, only(serial_contexts), prototype.relaxation_steps)

    contexts, context_build_seconds = build_contexts(prototype, langevin, ncontexts)
    for context in contexts
        run_steps!(langevin, context, 1)
    end
    spawned = run_spawned_contexts!(langevin, contexts, prototype.relaxation_steps)

    ideal_one_wave = serial.seconds
    serial_all = serial.seconds * Int(ncontexts)
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        contexts = Int(ncontexts),
        threads = Threads.nthreads(),
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        sweeps = SWEEPS,
        relaxation_steps = prototype.relaxation_steps,
        serial_build_seconds,
        context_build_seconds,
        single_context_seconds = serial.seconds,
        spawned_total_seconds = spawned.total_seconds,
        min_task_seconds = minimum(spawned.task_seconds),
        mean_task_seconds = mean(spawned.task_seconds),
        max_task_seconds = maximum(spawned.task_seconds),
        ideal_one_wave_seconds = ideal_one_wave,
        serial_all_seconds = serial_all,
        spawned_over_ideal_one_wave = spawned.total_seconds / ideal_one_wave,
        speedup_vs_serial_all = serial_all / spawned.total_seconds,
    )
    append_csv_row!(joinpath(OUTDIR, "manual_langevin_contexts.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Manual raw-context scheduling benchmark for the MNIST 10x-hidden Langevin step.
"""
function main()
    println(
        "Manual Langevin context scheduling workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " sweeps=", SWEEPS,
    )
    for ncontexts in WORKERS
        run_config(ncontexts)
    end
    println("Saved outputs in ", OUTDIR)
end

main()
