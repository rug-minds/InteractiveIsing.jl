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
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    context = Processes.context(worker)
    runtime_context = Processes._merge_runtime_inputs(context, (; phase_beta = config.β))
    plan = Processes.getplan(algorithm)
    wiring = Processes.getwiring(plan)
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
    @assert Processes.isresolved(algo)
    println("[manual untyped] after isresolved"); flush(stdout)
    println("[manual untyped] before before_while"); flush(stdout)
    Processes.before_while(process)
    println("[manual untyped] after before_while"); flush(stdout)

    println("[manual untyped] before getplan"); flush(stdout)
    step_plan = Processes.getplan(algo)
    println("[manual untyped] after getplan"); flush(stdout)
    println("[manual untyped] before getwiring"); flush(stdout)
    step_wiring = Processes.getwiring(step_plan)
    println("[manual untyped] after getwiring"); flush(stdout)

    println("[manual untyped] before _merge_runtime_inputs"); flush(stdout)
    runtime_context = Processes._merge_runtime_inputs(context, inputs)
    println("[manual untyped] after _merge_runtime_inputs"); flush(stdout)
    println("[manual untyped] before initial _step!"); flush(stdout)
    stablecontext = Processes._step!(
        step_plan,
        runtime_context,
        step_wiring,
        Processes.Namespace{nothing}(),
        process,
        r,
        Processes.Stable(),
    )
    println("[manual untyped] after initial _step!"); flush(stdout)
    println("[manual untyped] before tick!"); flush(stdout)
    Processes.tick!(process)
    println("[manual untyped] after tick!"); flush(stdout)
    println("[manual untyped] before inc!"); flush(stdout)
    Processes.inc!(process)
    println("[manual untyped] after inc!"); flush(stdout)

    println("[manual untyped] before loopidx"); flush(stdout)
    start_idx = Processes.loopidx(process)
    println("[manual untyped] after loopidx"); flush(stdout)
    println("[manual untyped] before repeats"); flush(stdout)
    end_idx = Processes.repeats(r)
    println("[manual untyped] after repeats start=$(start_idx) end=$(end_idx)"); flush(stdout)

    println("[manual untyped] before for"); flush(stdout)
    for _ in start_idx:end_idx
        println("[manual untyped] for before _step!"); flush(stdout)
        nextcontext = Processes._step!(
            step_plan,
            stablecontext,
            step_wiring,
            Processes.Namespace{nothing}(),
            process,
            r,
            Processes.Stable(),
        )
        println("[manual untyped] for after _step!"); flush(stdout)
        stablecontext = nextcontext
        println("[manual untyped] for before tick!"); flush(stdout)
        Processes.tick!(process)
        println("[manual untyped] for after tick!"); flush(stdout)
        println("[manual untyped] for before inc!"); flush(stdout)
        Processes.inc!(process)
        println("[manual untyped] for after inc!"); flush(stdout)
        println("[manual untyped] for before breakcondition"); flush(stdout)
        if Processes.breakcondition(r, process, stablecontext)
            println("[manual untyped] for break true"); flush(stdout)
            break
        end
        println("[manual untyped] for break false"); flush(stdout)
    end
    println("[manual untyped] after for"); flush(stdout)

    println("[manual untyped] before after_while"); flush(stdout)
    final_result = Processes.after_while(process, algo, stablecontext, context)
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
) where {P<:Processes.AbstractProcess,F<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime}
    println("[manual typed] enter"); flush(stdout)
    @assert Processes.isresolved(algo)
    Processes.before_while(process)
    step_plan = Processes.getplan(algo)
    step_wiring = Processes.getwiring(step_plan)
    runtime_context = Processes._merge_runtime_inputs(context, inputs)
    stablecontext = Processes._step!(step_plan, runtime_context, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
    Processes.tick!(process)
    Processes.inc!(process)
    start_idx = Processes.loopidx(process)
    end_idx = Processes.repeats(r)
    for _ in start_idx:end_idx
        nextcontext = Processes._step!(step_plan, stablecontext, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        stablecontext = nextcontext
        Processes.tick!(process)
        Processes.inc!(process)
        Processes.breakcondition(r, process, stablecontext) && break
    end
    println("[manual typed] before after_while"); flush(stdout)
    final_result = Processes.after_while(process, algo, stablecontext, context)
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
    ::Processes.Resuming{isresuming},
    ::Processes.NonGenerated,
) where {P<:Processes.AbstractProcess,F<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime,isresuming}
    println("[manual dispatchlike] enter isresuming=$(isresuming)"); flush(stdout)
    @assert Processes.isresolved(algo)
    Processes.before_while(process)
    step_plan = Processes.getplan(algo)
    step_wiring = Processes.getwiring(step_plan)
    runtime_context = Processes._merge_runtime_inputs(context, inputs)
    stablecontext = if isresuming
        @atomic process.paused = false
        runtime_context
    else
        stepped_context = Processes._step!(step_plan, runtime_context, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        Processes.tick!(process)
        Processes.inc!(process)
        stepped_context
    end
    start_idx = Processes.loopidx(process)
    end_idx = Processes.repeats(r)
    for _ in start_idx:end_idx
        nextcontext = Processes._step!(step_plan, stablecontext, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        stablecontext = nextcontext
        Processes.tick!(process)
        Processes.inc!(process)
        Processes.breakcondition(r, process, stablecontext) && break
    end
    println("[manual dispatchlike] before after_while"); flush(stdout)
    final_result = Processes.after_while(process, algo, stablecontext, context)
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
    trace("worker_algo_type=$(typeof(Processes.getalgo(probe.worker)))")
    trace("worker_plan_type=$(typeof(Processes.getplan(Processes.getalgo(probe.worker))))")
    trace("worker_lifetime=$(Processes.lifetime(probe.worker))")
    trace("context_type=$(typeof(probe.runtime_context))")

    if stage == "inc" || stage == "all"
        timed("parent inc!", () -> begin
            Processes.inc!(probe.plan)
            return Processes.inc(probe.plan)
        end)
    end

    if stage == "top_step" || stage == "all"
        timed("top composite _step!", () -> Processes._step!(
            probe.plan,
            probe.runtime_context,
            probe.wiring,
            Processes.Namespace{nothing}(),
            probe.worker,
            Processes.Repeat(1),
            Processes.Stable(),
        ))
    end

    if stage == "worker_top_step" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        worker_plan = Processes.getplan(worker_algo)
        worker_context = Processes._merge_runtime_inputs(Processes.context(probe.worker), (; phase_beta = probe.config.β))
        timed("worker top composite _step!", () -> Processes._step!(
            worker_plan,
            worker_context,
            Processes.getwiring(worker_plan),
            Processes.Namespace{nothing}(),
            probe.worker,
            Processes.lifetime(probe.worker),
            Processes.Stable(),
        ))
    end

    if stage == "loop_manual" || stage == "all"
        timed("before_while", () -> Processes.before_while(probe.worker))
        stepped = timed("loop first top _step!", () -> Processes._step!(
            probe.plan,
            probe.runtime_context,
            probe.wiring,
            Processes.Namespace{nothing}(),
            probe.worker,
            Processes.Repeat(1),
            Processes.Stable(),
        ))
        timed("process tick/inc", () -> begin
            Processes.tick!(probe.worker)
            Processes.inc!(probe.worker)
            return Processes.loopidx(probe.worker)
        end)
        trace("loopidx=$(Processes.loopidx(probe.worker)) repeat_end=$(Processes.repeats(Processes.Repeat(1)))")
        timed("after_while", () -> Processes.after_while(probe.worker, probe.algorithm, stepped, probe.context))
    end

    if stage == "runinline" || stage == "all"
        timed("runprocessinline! NonGenerated", () -> Processes.runprocessinline!(
            probe.worker;
            phase_beta = probe.config.β,
            looptype = Processes.NonGenerated(),
        ))
    end

    if stage == "worker_loop" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("loop(worker, worker_algo, NonGenerated)", () -> Processes.loop(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
            Processes.Resuming{false}(),
            Processes.NonGenerated(),
        ))
    end

    if stage == "worker_loop_repeat2" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        Processes.reset!(probe.worker)
        timed("loop(worker, worker_algo, Repeat(2), NonGenerated)", () -> Processes.loop(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.Repeat(2),
            (; phase_beta = probe.config.β),
            Processes.Resuming{false}(),
            Processes.NonGenerated(),
        ))
    end

    if stage == "worker_loop_indefinite" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("loop(worker, worker_algo, Indefinite, NonGenerated)", () -> Processes.loop(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.Indefinite(),
            (; phase_beta = probe.config.β),
            Processes.Resuming{false}(),
            Processes.NonGenerated(),
        ))
    end

    if stage == "manual_untyped" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("manual_repeat_loop_untyped", () -> manual_repeat_loop_untyped(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    if stage == "manual_typed" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("manual_repeat_loop_typed", () -> manual_repeat_loop_typed(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    if stage == "manual_dispatchlike" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("manual_repeat_loop_dispatchlike", () -> manual_repeat_loop_dispatchlike(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
            Processes.Resuming{false}(),
            Processes.NonGenerated(),
        ))
    end

    if stage == "process_helper_direct" || stage == "all"
        worker_algo = Processes.getalgo(probe.worker)
        timed("Processes._nongenerated_repeat_loop_impl direct", () -> Processes._nongenerated_repeat_loop_impl(
            probe.worker,
            worker_algo,
            Processes.context(probe.worker),
            Processes.lifetime(probe.worker),
            (; phase_beta = probe.config.β),
        ))
    end

    close(probe.worker)
    trace("done")
    return nothing
end

main()
