using Dates

const RUN_DIR = @__DIR__
const HELPER_PATH = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(HELPER_PATH)

"""Print a timestamped trace marker and flush immediately."""
function trace(label)
    println(now(), " ", label)
    flush(stdout)
    return nothing
end

"""Return a current reduced-step worker context and loop internals."""
function build_non_generated_probe()
    config = updated_config(langevin_learning_config(); batchsize = 1, workers = 1, sweeps = 0.001f0)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    setup = build_layer(config)
    trace("free_steps=$(setup.layer.free_relaxation_steps) nudged_steps=$(setup.layer.nudged_relaxation_steps)")
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    StatefulAlgorithms.reset!(worker)
    context = StatefulAlgorithms.context(worker)
    context = StatefulAlgorithms._merge_runtime_inputs(context, (; phase_beta = config.β))
    plan = StatefulAlgorithms.getplan(algorithm)
    wiring = StatefulAlgorithms.getwiring(plan)
    return (; config, setup, algorithm, worker, context, plan, wiring)
end

"""Run one labelled step and report whether it returned before the process exits."""
function traced_step(label::AbstractString, f)
    trace("BEGIN " * label)
    wall = @elapsed result = f()
    trace("END " * label * " wall=$(wall) result_type=$(typeof(result))")
    return result
end

function main()
    probe = build_non_generated_probe()
    plan_algos = StatefulAlgorithms.getalgos(probe.plan)
    trace("plan_type=$(typeof(probe.plan))")
    trace("context_type=$(typeof(probe.context))")

    free_plan = getfield(plan_algos, 1)
    nudged_plan = getfield(plan_algos, 2)
    finish_algo = getfield(plan_algos, 3)
    trace("free_plan_type=$(typeof(free_plan))")
    trace("nudged_plan_type=$(typeof(nudged_plan))")
    trace("finish_algo_type=$(typeof(finish_algo))")

    child_wiring = StatefulAlgorithms.child_wiring(probe.wiring)
    child_namespaces = getfield(probe.plan, :namespaces)
    free_wiring = getfield(child_wiring, 1)
    nudged_wiring = getfield(child_wiring, 2)
    finish_wiring = getfield(child_wiring, 3)
    free_namespace = getfield(child_namespaces, 1)
    nudged_namespace = getfield(child_namespaces, 2)
    finish_namespace = getfield(child_namespaces, 3)

    trace("free_wiring_type=$(typeof(free_wiring))")
    trace("nudged_wiring_type=$(typeof(nudged_wiring))")
    trace("finish_wiring_type=$(typeof(finish_wiring))")
    trace("free_namespace=$(free_namespace)")
    trace("nudged_namespace=$(nudged_namespace)")
    trace("finish_namespace=$(finish_namespace)")

    context = probe.context
    initial_context = context
    context = traced_step("free routine _step!", () -> StatefulAlgorithms._step!(
        free_plan,
        context,
        free_wiring,
        free_namespace,
        probe.worker,
        StatefulAlgorithms.Repeat(1),
        StatefulAlgorithms.Stable(),
    ))

    context = traced_step("nudged routine _step!", () -> StatefulAlgorithms._step!(
        nudged_plan,
        context,
        nudged_wiring,
        nudged_namespace,
        probe.worker,
        StatefulAlgorithms.Repeat(1),
        StatefulAlgorithms.Stable(),
    ))

    context = traced_step("finish algo _step!", () -> StatefulAlgorithms._step!(
        finish_algo,
        context,
        finish_wiring,
        finish_namespace,
        probe.worker,
        StatefulAlgorithms.Repeat(1),
        StatefulAlgorithms.Stable(),
    ))

    traced_step("after_while", () -> StatefulAlgorithms.after_while(probe.worker, probe.algorithm, context, initial_context))

    trace("done final_context_type=$(typeof(context))")
    close(probe.worker)
    return nothing
end

main()
