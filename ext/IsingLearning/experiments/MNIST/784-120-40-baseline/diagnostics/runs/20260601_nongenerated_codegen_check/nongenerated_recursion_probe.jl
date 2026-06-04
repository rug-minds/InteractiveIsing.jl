using Dates

const RUN_DIR = @__DIR__
const HELPER_PATH = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER_PATH)

"""Print a compact timestamped marker for long-running recursion probes."""
function trace(label)
    println(now(), " ", label)
    flush(stdout)
    return nothing
end

"""Build a one-sample MNIST contrastive worker for reduced or default sweep probes."""
function build_probe(mode::AbstractString)
    config = if mode == "default"
        updated_config(langevin_learning_config(); batchsize = 1, workers = 1)
    else
        updated_config(langevin_learning_config(); batchsize = 1, workers = 1, sweeps = 0.001f0)
    end
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    setup = build_layer(config)
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    StatefulAlgorithms.reset!(worker)
    context = StatefulAlgorithms.context(worker)
    runtime_context = StatefulAlgorithms._merge_runtime_inputs(context, (; phase_beta = config.β))
    plan = StatefulAlgorithms.getplan(algorithm)
    wiring = StatefulAlgorithms.getwiring(plan)
    return (; config, setup, algorithm, worker, context, runtime_context, plan, wiring)
end

"""Run one labelled probe, avoiding huge context prints on success."""
function timed(label::AbstractString, f)
    trace("BEGIN " * label)
    wall = @elapsed result = f()
    trace("END " * label * " wall=$(wall) type=$(typeof(result))")
    return result
end

"""Run the `Repeat` NonGenerated loop body without the package `loop` signature."""
function manual_repeat_loop_untyped(process, algo, context, r, inputs)
    println("[manual untyped] enter"); flush(stdout)
    println("[manual untyped] before isresolved"); flush(stdout)
    @assert StatefulAlgorithms.isresolved(algo)
    println("[manual untyped] after isresolved"); flush(stdout)
    println("[manual untyped] before before_while"); flush(stdout)
    StatefulAlgorithms.before_while(process)
    println("[manual untyped] after before_while"); flush(stdout)

    println("[manual untyped] before getplan"); flush(stdout)
    step_plan = StatefulAlgorithms.getplan(algo)
    println("[manual untyped] after getplan"); flush(stdout)
    println("[manual untyped] before getwiring"); flush(stdout)
    step_wiring = StatefulAlgorithms.getwiring(step_plan)
    println("[manual untyped] after getwiring"); flush(stdout)

    println("[manual untyped] before _merge_runtime_inputs"); flush(stdout)
    runtime_context = StatefulAlgorithms._merge_runtime_inputs(context, inputs)
    println("[manual untyped] after _merge_runtime_inputs"); flush(stdout)
    println("[manual untyped] before initial _step!"); flush(stdout)
    stablecontext = StatefulAlgorithms._step!(
        step_plan,
        runtime_context,
        step_wiring,
        StatefulAlgorithms.Namespace{nothing}(),
        process,
        r,
        StatefulAlgorithms.Stable(),
    )
    println("[manual untyped] after initial _step!"); flush(stdout)
    println("[manual untyped] before tick!"); flush(stdout)
    StatefulAlgorithms.tick!(process)
    println("[manual untyped] after tick!"); flush(stdout)
    println("[manual untyped] before inc!"); flush(stdout)
    StatefulAlgorithms.inc!(process)
    println("[manual untyped] after inc!"); flush(stdout)

    println("[manual untyped] before loopidx"); flush(stdout)
    start_idx = StatefulAlgorithms.loopidx(process)
    println("[manual untyped] after loopidx"); flush(stdout)
    println("[manual untyped] before repeats"); flush(stdout)
    end_idx = StatefulAlgorithms.repeats(r)
    println("[manual untyped] after repeats start=$(start_idx) end=$(end_idx)"); flush(stdout)

    println("[manual untyped] before for"); flush(stdout)
    for _ in start_idx:end_idx
        println("[manual untyped] for before _step!"); flush(stdout)
        nextcontext = StatefulAlgorithms._step!(
            step_plan,
            stablecontext,
            step_wiring,
            StatefulAlgorithms.Namespace{nothing}(),
            process,
            r,
            StatefulAlgorithms.Stable(),
        )
        println("[manual untyped] for after _step!"); flush(stdout)
        stablecontext = nextcontext
        println("[manual untyped] for before tick!"); flush(stdout)
        StatefulAlgorithms.tick!(process)
        println("[manual untyped] for after tick!"); flush(stdout)
        println("[manual untyped] for before inc!"); flush(stdout)
        StatefulAlgorithms.inc!(process)
        println("[manual untyped] for after inc!"); flush(stdout)
        println("[manual untyped] for before breakcondition"); flush(stdout)
        if StatefulAlgorithms.breakcondition(r, process, stablecontext)
            println("[manual untyped] for break true"); flush(stdout)
            break
        end
        println("[manual untyped] for break false"); flush(stdout)
    end
    println("[manual untyped] after for"); flush(stdout)

    println("[manual untyped] before after_while"); flush(stdout)
    final_result = StatefulAlgorithms.after_while(process, algo, stablecontext, context)
    println("[manual untyped] after after_while"); flush(stdout)
    return final_result
end

"""Run the same body with the package-like type selectors but without `@inline`/`@constprop`."""
function manual_repeat_loop_typed(
    process::P,
    algo::F,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:StatefulAlgorithms.AbstractProcess,F<:StatefulAlgorithms.AbstractLoopAlgorithm,C,R<:StatefulAlgorithms.RepeatLifetime}
    println("[manual typed] enter"); flush(stdout)
    @assert StatefulAlgorithms.isresolved(algo)
    StatefulAlgorithms.before_while(process)
    step_plan = StatefulAlgorithms.getplan(algo)
    step_wiring = StatefulAlgorithms.getwiring(step_plan)
    runtime_context = StatefulAlgorithms._merge_runtime_inputs(context, inputs)
    stablecontext = StatefulAlgorithms._step!(step_plan, runtime_context, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
    StatefulAlgorithms.tick!(process)
    StatefulAlgorithms.inc!(process)
    start_idx = StatefulAlgorithms.loopidx(process)
    end_idx = StatefulAlgorithms.repeats(r)
    for _ in start_idx:end_idx
        nextcontext = StatefulAlgorithms._step!(step_plan, stablecontext, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        stablecontext = nextcontext
        StatefulAlgorithms.tick!(process)
        StatefulAlgorithms.inc!(process)
        StatefulAlgorithms.breakcondition(r, process, stablecontext) && break
    end
    println("[manual typed] before after_while"); flush(stdout)
    final_result = StatefulAlgorithms.after_while(process, algo, stablecontext, context)
    println("[manual typed] after after_while"); flush(stdout)
    return final_result
end

"""Run the same body with the package loop dispatch shape but no method annotations."""
function manual_repeat_loop_dispatchlike(
    process::P,
    algo::F,
    context::C,
    r::R,
    inputs::NamedTuple,
    ::StatefulAlgorithms.Resuming{isresuming},
    ::StatefulAlgorithms.NonGenerated,
) where {P<:StatefulAlgorithms.AbstractProcess,F<:StatefulAlgorithms.AbstractLoopAlgorithm,C,R<:StatefulAlgorithms.RepeatLifetime,isresuming}
    println("[manual dispatchlike] enter isresuming=$(isresuming)"); flush(stdout)
    @assert StatefulAlgorithms.isresolved(algo)
    StatefulAlgorithms.before_while(process)
    step_plan = StatefulAlgorithms.getplan(algo)
    step_wiring = StatefulAlgorithms.getwiring(step_plan)
    runtime_context = StatefulAlgorithms._merge_runtime_inputs(context, inputs)
    stablecontext = if isresuming
        @atomic process.paused = false
        runtime_context
    else
        stepped_context = StatefulAlgorithms._step!(step_plan, runtime_context, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        StatefulAlgorithms.tick!(process)
        StatefulAlgorithms.inc!(process)
        stepped_context
    end
    start_idx = StatefulAlgorithms.loopidx(process)
    end_idx = StatefulAlgorithms.repeats(r)
    for _ in start_idx:end_idx
        nextcontext = StatefulAlgorithms._step!(step_plan, stablecontext, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        stablecontext = nextcontext
        StatefulAlgorithms.tick!(process)
        StatefulAlgorithms.inc!(process)
        StatefulAlgorithms.breakcondition(r, process, stablecontext) && break
    end
    println("[manual dispatchlike] before after_while"); flush(stdout)
    final_result = StatefulAlgorithms.after_while(process, algo, stablecontext, context)
    println("[manual dispatchlike] after after_while"); flush(stdout)
    return final_result
end

function main()
    stage = isempty(ARGS) ? "all" : ARGS[1]
    mode = length(ARGS) >= 2 ? ARGS[2] : "reduced"
    probe = build_probe(mode)
    trace("stage=$(stage)")
    trace("mode=$(mode) free_steps=$(probe.setup.layer.free_relaxation_steps) nudged_steps=$(probe.setup.layer.nudged_relaxation_steps)")
    trace("plan_type=$(typeof(probe.plan))")
    trace("worker_algo_type=$(typeof(StatefulAlgorithms.getalgo(probe.worker)))")
    trace("worker_plan_type=$(typeof(StatefulAlgorithms.getplan(StatefulAlgorithms.getalgo(probe.worker))))")
    trace("worker_lifetime=$(StatefulAlgorithms.lifetime(probe.worker))")
    trace("context_type=$(typeof(probe.runtime_context))")

    if stage == "inc" || stage == "all"
        timed("parent inc!", () -> begin
            StatefulAlgorithms.inc!(probe.plan)
            return StatefulAlgorithms.inc(probe.plan)
        end)
    end

    if stage == "top_step" || stage == "all"
        timed("top composite _step!", () -> StatefulAlgorithms._step!(
            probe.plan,
            probe.runtime_context,
            probe.wiring,
            StatefulAlgorithms.Namespace{nothing}(),
            probe.worker,
            StatefulAlgorithms.Repeat(1),
            StatefulAlgorithms.Stable(),
        ))
    end

    if stage == "worker_top_step" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        worker_plan = StatefulAlgorithms.getplan(worker_algo)
        worker_context = StatefulAlgorithms._merge_runtime_inputs(StatefulAlgorithms.context(probe.worker), (; phase_beta = probe.config.β))
        timed("worker top composite _step!", () -> StatefulAlgorithms._step!(
            worker_plan,
            worker_context,
            StatefulAlgorithms.getwiring(worker_plan),
            StatefulAlgorithms.Namespace{nothing}(),
            probe.worker,
            StatefulAlgorithms.lifetime(probe.worker),
            StatefulAlgorithms.Stable(),
        ))
    end

    if stage == "loop_manual" || stage == "all"
        timed("before_while", () -> StatefulAlgorithms.before_while(probe.worker))
        stepped = timed("loop first top _step!", () -> StatefulAlgorithms._step!(
            probe.plan,
            probe.runtime_context,
            probe.wiring,
            StatefulAlgorithms.Namespace{nothing}(),
            probe.worker,
            StatefulAlgorithms.Repeat(1),
            StatefulAlgorithms.Stable(),
        ))
        timed("process tick/inc", () -> begin
            StatefulAlgorithms.tick!(probe.worker)
            StatefulAlgorithms.inc!(probe.worker)
            return StatefulAlgorithms.loopidx(probe.worker)
        end)
        trace("loopidx=$(StatefulAlgorithms.loopidx(probe.worker)) repeat_end=$(StatefulAlgorithms.repeats(StatefulAlgorithms.Repeat(1)))")
        timed("after_while", () -> StatefulAlgorithms.after_while(probe.worker, probe.algorithm, stepped, probe.context))
    end

    if stage == "runinline" || stage == "all"
        timed("runprocessinline! NonGenerated", () -> StatefulAlgorithms.runprocessinline!(
            probe.worker;
            phase_beta = probe.config.β,
            looptype = StatefulAlgorithms.NonGenerated(),
        ))
    end

    if stage == "worker_loop" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("loop(worker, worker_algo, NonGenerated)", () -> StatefulAlgorithms.loop(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
            StatefulAlgorithms.Resuming{false}(),
            StatefulAlgorithms.NonGenerated(),
        ))
    end

    if stage == "worker_loop_repeat2" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        StatefulAlgorithms.reset!(probe.worker)
        timed("loop(worker, worker_algo, Repeat(2), NonGenerated)", () -> StatefulAlgorithms.loop(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.Repeat(2),
            (; phase_beta = probe.config.β),
            StatefulAlgorithms.Resuming{false}(),
            StatefulAlgorithms.NonGenerated(),
        ))
    end

    if stage == "worker_loop_indefinite" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("loop(worker, worker_algo, Indefinite, NonGenerated)", () -> StatefulAlgorithms.loop(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.Indefinite(),
            (; phase_beta = probe.config.β),
            StatefulAlgorithms.Resuming{false}(),
            StatefulAlgorithms.NonGenerated(),
        ))
    end

    if stage == "manual_untyped" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("manual_repeat_loop_untyped", () -> manual_repeat_loop_untyped(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    if stage == "manual_typed" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("manual_repeat_loop_typed", () -> manual_repeat_loop_typed(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    if stage == "manual_dispatchlike" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("manual_repeat_loop_dispatchlike", () -> manual_repeat_loop_dispatchlike(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
            StatefulAlgorithms.Resuming{false}(),
            StatefulAlgorithms.NonGenerated(),
        ))
    end

    if stage == "process_helper_direct" || stage == "all"
        worker_algo = StatefulAlgorithms.getalgo(probe.worker)
        timed("StatefulAlgorithms._nongenerated_repeat_loop_impl direct", () -> StatefulAlgorithms._nongenerated_repeat_loop_impl(
            probe.worker,
            worker_algo,
            StatefulAlgorithms.context(probe.worker),
            StatefulAlgorithms.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    close(probe.worker)
    trace("done")
    return nothing
end

main()
