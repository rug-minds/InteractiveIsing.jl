using Dates
using Random

const RUN_DIR = @__DIR__
const BASELINE_PATH = normpath(joinpath(RUN_DIR, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(BASELINE_PATH)

"""Print a timestamped marker and flush so compile stalls are visible in redirected logs."""
function marker(label::S) where {S<:AbstractString}
    println(now(), " ", label)
    flush(stdout)
    return nothing
end

"""Build one reduced single-sample MNIST Process worker without loading MNIST data."""
function build_repro_worker()
    config = InputFieldMNISTConfig(;
        workers = 1,
        epochs = 1,
        batchsize = 1,
        train_per_class = 1,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 0,
        sweeps = 0.001f0,
        outdir = RUN_DIR,
    )
    setup = build_layer(config)
    algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))

    rng = Random.MersenneTwister(config.seed)
    x = rand(rng, FT, INPUT_DIM, 1)
    y = fill(-one(FT), NCLASSES * config.output_replicas, 1)
    y[1:config.output_replicas, 1] .= one(FT)
    load_sample_into_worker!(worker_context(worker), x, y, 1)
    StatefulAlgorithms.reset!(worker)
    return (; config, setup, algorithm = StatefulAlgorithms.getalgo(worker), worker)
end

"""Run the old local-variable loop shape with call-site `@inline _step!`."""
function bad_local_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:StatefulAlgorithms.AbstractProcess,A<:StatefulAlgorithms.AbstractLoopAlgorithm,C,R<:StatefulAlgorithms.RepeatLifetime}
    marker("bad-local enter")
    @assert StatefulAlgorithms.isresolved(algo)
    @inline StatefulAlgorithms.before_while(process)
    step_plan = @inline StatefulAlgorithms.getplan(algo)
    step_wiring = @inline StatefulAlgorithms.getwiring(step_plan)
    runtime_context = @inline StatefulAlgorithms._merge_runtime_inputs(context, inputs)

    marker("bad-local first _step begin")
    stablecontext = @inline StatefulAlgorithms._step!(step_plan, runtime_context, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
    marker("bad-local first _step end")
    @inline StatefulAlgorithms.tick!(process)
    @inline StatefulAlgorithms.inc!(process)

    start_idx = @inline StatefulAlgorithms.loopidx(process)
    end_idx = @inline StatefulAlgorithms.repeats(r)
    marker("bad-local loop range start=$(start_idx) end=$(end_idx)")
    for idx in start_idx:end_idx
        marker("bad-local for $(idx) _step begin")
        nextcontext = @inline StatefulAlgorithms._step!(step_plan, stablecontext, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        marker("bad-local for $(idx) _step end")
        stablecontext = nextcontext
        @inline StatefulAlgorithms.tick!(process)
        @inline StatefulAlgorithms.inc!(process)
        (@inline StatefulAlgorithms.breakcondition(r, process, stablecontext)) && break
    end
    marker("bad-local after loop")
    return @inline StatefulAlgorithms.after_while(process, algo, stablecontext, context)
end

"""Run the same loop shape but remove call-site `@inline` from `_step!` calls."""
function no_inline_local_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:StatefulAlgorithms.AbstractProcess,A<:StatefulAlgorithms.AbstractLoopAlgorithm,C,R<:StatefulAlgorithms.RepeatLifetime}
    marker("no-inline-local enter")
    @assert StatefulAlgorithms.isresolved(algo)
    StatefulAlgorithms.before_while(process)
    step_plan = StatefulAlgorithms.getplan(algo)
    step_wiring = StatefulAlgorithms.getwiring(step_plan)
    runtime_context = StatefulAlgorithms._merge_runtime_inputs(context, inputs)

    marker("no-inline-local first _step begin")
    stablecontext = StatefulAlgorithms._step!(step_plan, runtime_context, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
    marker("no-inline-local first _step end")
    StatefulAlgorithms.tick!(process)
    StatefulAlgorithms.inc!(process)

    start_idx = StatefulAlgorithms.loopidx(process)
    end_idx = StatefulAlgorithms.repeats(r)
    marker("no-inline-local loop range start=$(start_idx) end=$(end_idx)")
    for idx in start_idx:end_idx
        marker("no-inline-local for $(idx) _step begin")
        nextcontext = StatefulAlgorithms._step!(step_plan, stablecontext, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        marker("no-inline-local for $(idx) _step end")
        stablecontext = nextcontext
        StatefulAlgorithms.tick!(process)
        StatefulAlgorithms.inc!(process)
        StatefulAlgorithms.breakcondition(r, process, stablecontext) && break
    end
    marker("no-inline-local after loop")
    return StatefulAlgorithms.after_while(process, algo, stablecontext, context)
end

"""Run the workaround loop shape with a concretely typed `RefValue` context cell."""
function typed_ref_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:StatefulAlgorithms.AbstractProcess,A<:StatefulAlgorithms.AbstractLoopAlgorithm,C,R<:StatefulAlgorithms.RepeatLifetime}
    marker("typed-ref enter")
    @assert StatefulAlgorithms.isresolved(algo)
    @inline StatefulAlgorithms.before_while(process)
    step_plan = @inline StatefulAlgorithms.getplan(algo)
    step_wiring = @inline StatefulAlgorithms.getwiring(step_plan)
    runtime_context = @inline StatefulAlgorithms._merge_runtime_inputs(context, inputs)

    marker("typed-ref first _step begin")
    initial_context = @inline StatefulAlgorithms._step!(step_plan, runtime_context, step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
    marker("typed-ref first _step end")
    @inline StatefulAlgorithms.tick!(process)
    @inline StatefulAlgorithms.inc!(process)

    start_idx = @inline StatefulAlgorithms.loopidx(process)
    end_idx = @inline StatefulAlgorithms.repeats(r)
    marker("typed-ref loop range start=$(start_idx) end=$(end_idx)")
    refcontext = Base.RefValue{typeof(initial_context)}(initial_context)
    marker("typed-ref ref_type=$(typeof(refcontext))")
    for idx in start_idx:end_idx
        marker("typed-ref for $(idx) _step begin")
        refcontext[] = @inline StatefulAlgorithms._step!(step_plan, refcontext[], step_wiring, StatefulAlgorithms.Namespace{nothing}(), process, r, StatefulAlgorithms.Stable())
        marker("typed-ref for $(idx) _step end")
        @inline StatefulAlgorithms.tick!(process)
        @inline StatefulAlgorithms.inc!(process)
        (@inline StatefulAlgorithms.breakcondition(r, process, refcontext[])) && break
    end
    marker("typed-ref after loop")
    return @inline StatefulAlgorithms.after_while(process, algo, refcontext[], context)
end

"""Run one selected loop-shape repro stage."""
function run_stage(stage::S, repeat_count::I) where {S<:AbstractString,I<:Integer}
    probe = build_repro_worker()
    lifetime = StatefulAlgorithms.Repeat(Int(repeat_count))
    inputs = (; phase_beta = probe.config.β)
    context = StatefulAlgorithms.context(probe.worker)
    marker("stage=$(stage) repeat_count=$(repeat_count) context_type=$(nameof(typeof(context)))")
    wall = @elapsed result = if stage == "bad-local"
        bad_local_loop!(probe.worker, probe.algorithm, context, lifetime, inputs)
    elseif stage == "no-inline-local"
        no_inline_local_loop!(probe.worker, probe.algorithm, context, lifetime, inputs)
    elseif stage == "typed-ref"
        typed_ref_loop!(probe.worker, probe.algorithm, context, lifetime, inputs)
    else
        error("stage must be one of: bad-local, no-inline-local, typed-ref")
    end
    marker("done stage=$(stage) wall=$(wall) result_type=$(nameof(typeof(result)))")
    return result
end

function main()
    stage = isempty(ARGS) ? "typed-ref" : ARGS[1]
    repeat_count = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    return run_stage(stage, repeat_count)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
