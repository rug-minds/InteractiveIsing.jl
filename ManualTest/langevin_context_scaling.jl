using InteractiveIsing
using InteractiveIsing.Processes
using Random

# Run from the repository root with:
#     julia --project=. -t 8 ManualTest/langevin_context_scaling.jl
#
# Optional knobs:
#     ISING_MANUAL_RUNS=8
#     ISING_MANUAL_FULLSWEEPS=500
#     ISING_MANUAL_SIDE=32
#     ISING_MANUAL_STEPSIZE=0.02
#     ISING_MANUAL_TEMP=1.5

const II = InteractiveIsing
const NN_WEIGHTS = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
const LOCAL_LANGEVIN_OUTPUT_NAMES = (
    :proposal,
    :ΔE,
    :accepted,
    :attempted,
    :acceptance_rate,
    :T,
    :η,
    :σ,
    :group_steps,
    :gradient_max,
    :gradient_rms,
    :reflected_fraction,
)

"""
    env_int(name, default) -> Int

Read an integer benchmark option from `ENV`, falling back to `default`.
"""
function env_int(name::T, default::I) where {T<:AbstractString,I<:Integer}
    return parse(Int, get(ENV, name, string(default)))
end

"""
    env_float32(name, default) -> Float32

Read a `Float32` benchmark option from `ENV`, falling back to `default`.
"""
function env_float32(name::T, default::F) where {T<:AbstractString,F<:Real}
    return parse(Float32, get(ENV, name, string(default)))
end

"""
    make_langevin_graph(side; temperature) -> IsingGraph

Construct the continuous nearest-neighbor Ising graph used by the raw Langevin
scaling benchmark.
"""
function make_langevin_graph(side::T; temperature::F = 1.5f0) where {T<:Integer,F<:Real}
    graph = IsingGraph(
        side,
        side,
        Continuous(),
        NN_WEIGHTS,
        StateSet(-1f0, 1f0),
        Ising(c = ConstVal(1f0), b = 0f0);
        periodic = (:x, :y),
    )
    temp!(graph, Float32(temperature))
    return graph
end

"""
    make_context_template(algorithm, side; temperature, seed) -> NamedTuple

Initialize one reusable process-style Langevin context. Benchmark workers are
`deepcopy`s of this context, so every timed run has private graph, RNG, and
preallocation. The hot loop steps a `SubContextView`, matching the generated
process-loop path. The returned `instance` is the registered `IdentifiableAlgo`;
using it avoids dynamic registry lookup in `view`.
"""
function make_context_template(
    algorithm::A,
    side::T;
    temperature::F = 1.5f0,
    seed::I = 1,
) where {A<:II.LocalLangevin,T<:Integer,F<:Real,I<:Integer}
    graph = make_langevin_graph(side; temperature)
    process = Process(algorithm, Init(algorithm, model = graph); repeats = 1)
    context = Processes.context(process)
    instance = only(Processes.getalgos(Processes.getalgo(process)))
    Random.seed!(view(context, instance).rng, seed)
    return (; context, instance)
end

"""
    fullsweep_steps(context, instance, fullsweeps) -> Int

Convert full sweeps to raw `Processes.step!` calls. `LocalLangevin` attempts
one active spin update per step, so one full sweep is one pass over all active
spins.
"""
function fullsweep_steps(context::C, instance::I, fullsweeps::T) where {C<:Processes.ProcessContext,I<:Processes.AbstractIdentifiableAlgo,T<:Integer}
    return Int(fullsweeps) * length(view(context, instance).active_spins)
end

"""
    checked_langevin_step!(instance, contextview) -> NamedTuple

Run one raw Langevin step and assert the nonempty active-set output shape. This
removes the `NamedTuple()` branch from the benchmark hot loop.
"""
@inline function checked_langevin_step!(instance::I, context::C) where {I<:Processes.AbstractIdentifiableAlgo,C}
    return Processes.step!(Processes.getalgo(instance), context)::NamedTuple{LOCAL_LANGEVIN_OUTPUT_NAMES}
end

"""
    run_raw_langevin!(instance, context, nsteps) -> NamedTuple

Run `nsteps` single-spin Langevin updates without merging scalar diagnostics
back into the full context. This keeps one `SubContextView` because the full
context shape never changes in the no-merge path.
"""
function run_raw_langevin!(instance::I, context::C, nsteps::T) where {I<:Processes.AbstractIdentifiableAlgo,C<:Processes.ProcessContext,T<:Integer}
    contextview = view(context, instance)
    algorithm = Processes.getalgo(instance)
    accepted = 0
    attempted = 0
    last_gradient_max = zero(eltype(contextview.model))

    # Do not `merge(context, out)` here. For LocalLangevin, the state needed by
    # future raw steps is held in mutable context fields; merging allocates one
    # fresh NamedTuple per step and mostly updates diagnostics.
    for _ in 1:nsteps
        out = Processes.step!(algorithm, contextview)
        accepted += out.accepted
        attempted += out.attempted
        last_gradient_max = out.gradient_max
    end

    return (;
        accepted,
        attempted,
        acceptance_rate = attempted == 0 ? NaN : accepted / attempted,
        final_state_sum = sum(II.state(contextview.model)),
        last_gradient_max,
    )
end

"""
    stabilize_langevin_context(instance, context) -> ProcessContext

Run the initial shape-changing merge outside the timed stable loop. The process
engine does this with an unstable warmup step before stable generated steps.
"""
function stabilize_langevin_context(instance::I, context::C) where {I<:Processes.AbstractIdentifiableAlgo,C<:Processes.ProcessContext}
    contextview = view(context, instance)
    out = checked_langevin_step!(instance, contextview)
    return Processes.unstablemerge(contextview, out)
end

"""
    run_raw_langevin_merged_inner!(instance, context, nsteps, accepted, attempted) -> NamedTuple

Run the stable-merge loop after the context has its post-warmup type.
"""
function run_raw_langevin_merged_inner!(
    instance::I,
    context::C,
    nsteps::T,
    accepted::N,
    attempted::N,
) where {I<:Processes.AbstractIdentifiableAlgo,C<:Processes.ProcessContext,T<:Integer,N<:Integer}
    for _ in 1:nsteps
        contextview = view(context, instance)
        out = checked_langevin_step!(instance, contextview)
        context = Processes.stablemerge(contextview, out)
        accepted += out.accepted
        attempted += out.attempted
    end

    final_view = view(context, instance)
    return (;
        accepted,
        attempted,
        acceptance_rate = attempted == 0 ? NaN : accepted / attempted,
        final_state_sum = sum(II.state(final_view.model)),
        last_gradient_max = final_view.gradient_max,
    )
end

"""
    run_raw_langevin_with_merge!(instance, context, nsteps) -> NamedTuple

Run the same raw dynamics while stable-merging each returned diagnostic tuple
into the context. Pass a context that has already gone through
`stabilize_langevin_context`.
"""
function run_raw_langevin_with_merge!(instance::I, context::C, nsteps::T) where {I<:Processes.AbstractIdentifiableAlgo,C<:Processes.ProcessContext,T<:Integer}
    contextview = view(context, instance)
    nsteps <= 0 && return (;
        accepted = 0,
        attempted = 0,
        acceptance_rate = NaN,
        final_state_sum = sum(II.state(contextview.model)),
        last_gradient_max = contextview.gradient_max,
    )

    return run_raw_langevin_merged_inner!(instance, context, nsteps, 0, 0)
end

"""
    elapsed_seconds(f) -> (seconds, value)

Time one benchmark closure with `time_ns`, returning elapsed wall time and the
closure result.
"""
function elapsed_seconds(f::F) where {F}
    GC.gc()
    start = time_ns()
    value = f()
    return (time_ns() - start) / 1e9, value
end

"""
    run_serial!(algorithm, contexts, nsteps) -> Vector

Run all raw contexts one after another on the current Julia thread.
"""
function run_serial!(instance::I, contexts::C, nsteps::T) where {I<:Processes.AbstractIdentifiableAlgo,C<:AbstractVector,T<:Integer}
    return [run_raw_langevin!(instance, context, nsteps) for context in contexts]
end

"""
    run_threaded!(algorithm, contexts, nsteps) -> Vector

Run all raw contexts concurrently using Julia thread tasks.
"""
function run_threaded!(instance::I, contexts::C, nsteps::T) where {I<:Processes.AbstractIdentifiableAlgo,C<:AbstractVector,T<:Integer}
    tasks = map(eachindex(contexts)) do idx
        context = contexts[idx]
        Threads.@spawn run_raw_langevin!($instance, $context, $nsteps)
    end
    return fetch.(tasks)
end

"""
    run_static_threaded!(algorithm, contexts, nsteps) -> Vector

Run all raw contexts concurrently using Julia's static thread scheduler.
"""
function run_static_threaded!(instance::I, contexts::C, nsteps::T) where {I<:Processes.AbstractIdentifiableAlgo,C<:AbstractVector,T<:Integer}
    results = Vector{Any}(undef, length(contexts))
    Threads.@threads :static for idx in eachindex(contexts)
        results[idx] = run_raw_langevin!(instance, contexts[idx], nsteps)
    end
    return results
end

"""
    ns_per_step(seconds, nsteps, runs) -> Float64

Convert elapsed seconds to nanoseconds per raw single-spin Langevin step.
"""
function ns_per_step(seconds::S, nsteps::T, runs::R) where {S<:Real,T<:Integer,R<:Integer}
    return 1e9 * seconds / (Int(nsteps) * Int(runs))
end

"""
    print_summary(; kwargs...) -> NamedTuple

Build deep-copied raw contexts, compare the no-merge and merge raw loops, then
time one context, serial multi-context execution, and threaded multi-context
execution.
"""
function print_summary(;
    runs::R = env_int("ISING_MANUAL_RUNS", 8),
    fullsweeps::S = env_int("ISING_MANUAL_FULLSWEEPS", 500),
    side::D = env_int("ISING_MANUAL_SIDE", 32),
    stepsize::E = env_float32("ISING_MANUAL_STEPSIZE", 0.02f0),
    temperature::F = env_float32("ISING_MANUAL_TEMP", 1.5f0),
    seed::I = env_int("ISING_MANUAL_SEED", 1),
) where {R<:Integer,S<:Integer,D<:Integer,E<:Real,F<:Real,I<:Integer}
    algorithm = LocalLangevin(
        stepsize = Float32(stepsize),
        max_drift_fraction = 0.15f0,
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    template = make_context_template(algorithm, side; temperature, seed)
    instance = template.instance
    base_context = template.context
    nsteps = fullsweep_steps(base_context, instance, fullsweeps)
    active_spins = length(view(base_context, instance).active_spins)

    println("Manual raw Langevin step scaling")
    println("Julia threads:          ", Threads.nthreads())
    println("contexts/runs:          ", runs)
    println("graph side:             ", side, "x", side)
    println("active spins/context:   ", active_spins)
    println("full sweeps/context:    ", fullsweeps)
    println("raw steps/context:      ", nsteps)
    println("Langevin stepsize/temp: ", Float32(stepsize), " / ", Float32(temperature))
    println()

    # Pay compilation and first-cycle setup costs before timing.
    run_raw_langevin!(instance, deepcopy(base_context), active_spins)
    run_threaded!(instance, [deepcopy(base_context)], active_spins)
    run_static_threaded!(instance, [deepcopy(base_context)], active_spins)
    stable_warmup = stabilize_langevin_context(instance, deepcopy(base_context))
    run_raw_langevin_with_merge!(instance, stable_warmup, 1)

    no_merge_context = deepcopy(base_context)
    no_merge_seconds, no_merge_result = elapsed_seconds() do
        run_raw_langevin!(instance, no_merge_context, nsteps)
    end

    merge_context = stabilize_langevin_context(instance, deepcopy(base_context))
    merge_allocated = @allocated run_raw_langevin_with_merge!(instance, merge_context, min(nsteps, active_spins))
    merge_context = stabilize_langevin_context(instance, deepcopy(base_context))
    merge_seconds, merge_result = elapsed_seconds() do
        run_raw_langevin_with_merge!(instance, merge_context, nsteps)
    end

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

    static_contexts = [deepcopy(base_context) for _ in 1:runs]
    static_seconds, static_results = elapsed_seconds() do
        run_static_threaded!(instance, static_contexts, nsteps)
    end

    println("no-merge seconds:       ", round(no_merge_seconds, digits = 4),
        " (", round(ns_per_step(no_merge_seconds, nsteps, 1), digits = 2), " ns/step)")
    println("merge seconds:          ", round(merge_seconds, digits = 4),
        " (", round(ns_per_step(merge_seconds, nsteps, 1), digits = 2), " ns/step)")
    println("merge alloc/sample:     ", merge_allocated, " bytes for ", active_spins, " steps")
    println()
    println("single context seconds: ", round(single_seconds, digits = 4))
    println("serial ", runs, " contexts sec: ", round(serial_seconds, digits = 4))
    println("threaded ", runs, " contexts:   ", round(threaded_seconds, digits = 4))
    println("static ", runs, " contexts:     ", round(static_seconds, digits = 4))
    println()
    println("serial / single:        ", round(serial_seconds / single_seconds, digits = 3), "x")
    println("threaded / single:      ", round(threaded_seconds / single_seconds, digits = 3), "x")
    println("static / single:        ", round(static_seconds / single_seconds, digits = 3), "x")
    println("serial / threaded:      ", round(serial_seconds / threaded_seconds, digits = 3), "x")
    println("serial / static:        ", round(serial_seconds / static_seconds, digits = 3), "x")
    println("static / threaded:      ", round(static_seconds / threaded_seconds, digits = 3), "x wall-time ratio")
    println("ideal threaded/single:  1.0x wall time, ", runs, "x throughput")
    println()
    println("single acceptance:      ", round(single_result.acceptance_rate, digits = 4))
    println("threaded acceptances:   ", round.(getproperty.(threaded_results, :acceptance_rate), digits = 4))
    println("static acceptances:     ", round.(getproperty.(static_results, :acceptance_rate), digits = 4))

    return (;
        no_merge_seconds,
        merge_seconds,
        merge_allocated,
        no_merge_result,
        merge_result,
        single_seconds,
        serial_seconds,
        threaded_seconds,
        static_seconds,
        single_result,
        serial_results,
        threaded_results,
        static_results,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_summary()
end
