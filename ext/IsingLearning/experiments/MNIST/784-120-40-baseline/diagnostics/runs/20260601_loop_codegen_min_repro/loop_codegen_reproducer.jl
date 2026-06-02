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
    algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
    worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(setup.graph), Ref(copy(setup.input_hidden_w)))

    rng = Random.MersenneTwister(config.seed)
    x = rand(rng, FT, INPUT_DIM, 1)
    y = fill(-one(FT), NCLASSES * config.output_replicas, 1)
    y[1:config.output_replicas, 1] .= one(FT)
    load_sample_into_worker!(worker_context(worker), x, y, 1)
    Processes.reset!(worker)
    return (; config, setup, algorithm = Processes.getalgo(worker), worker)
end

"""Run the old local-variable loop shape with call-site `@inline _step!`."""
function bad_local_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:Processes.AbstractProcess,A<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime}
    marker("bad-local enter")
    @assert Processes.isresolved(algo)
    @inline Processes.before_while(process)
    step_plan = @inline Processes.getplan(algo)
    step_wiring = @inline Processes.getwiring(step_plan)
    runtime_context = @inline Processes._merge_runtime_inputs(context, inputs)

    marker("bad-local first _step begin")
    stablecontext = @inline Processes._step!(step_plan, runtime_context, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
    marker("bad-local first _step end")
    @inline Processes.tick!(process)
    @inline Processes.inc!(process)

    start_idx = @inline Processes.loopidx(process)
    end_idx = @inline Processes.repeats(r)
    marker("bad-local loop range start=$(start_idx) end=$(end_idx)")
    for idx in start_idx:end_idx
        marker("bad-local for $(idx) _step begin")
        nextcontext = @inline Processes._step!(step_plan, stablecontext, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        marker("bad-local for $(idx) _step end")
        stablecontext = nextcontext
        @inline Processes.tick!(process)
        @inline Processes.inc!(process)
        (@inline Processes.breakcondition(r, process, stablecontext)) && break
    end
    marker("bad-local after loop")
    return @inline Processes.after_while(process, algo, stablecontext, context)
end

"""Run the same loop shape but remove call-site `@inline` from `_step!` calls."""
function no_inline_local_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:Processes.AbstractProcess,A<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime}
    marker("no-inline-local enter")
    @assert Processes.isresolved(algo)
    Processes.before_while(process)
    step_plan = Processes.getplan(algo)
    step_wiring = Processes.getwiring(step_plan)
    runtime_context = Processes._merge_runtime_inputs(context, inputs)

    marker("no-inline-local first _step begin")
    stablecontext = Processes._step!(step_plan, runtime_context, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
    marker("no-inline-local first _step end")
    Processes.tick!(process)
    Processes.inc!(process)

    start_idx = Processes.loopidx(process)
    end_idx = Processes.repeats(r)
    marker("no-inline-local loop range start=$(start_idx) end=$(end_idx)")
    for idx in start_idx:end_idx
        marker("no-inline-local for $(idx) _step begin")
        nextcontext = Processes._step!(step_plan, stablecontext, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        marker("no-inline-local for $(idx) _step end")
        stablecontext = nextcontext
        Processes.tick!(process)
        Processes.inc!(process)
        Processes.breakcondition(r, process, stablecontext) && break
    end
    marker("no-inline-local after loop")
    return Processes.after_while(process, algo, stablecontext, context)
end

"""Run the workaround loop shape with a concretely typed `RefValue` context cell."""
function typed_ref_loop!(
    process::P,
    algo::A,
    context::C,
    r::R,
    inputs::NamedTuple,
) where {P<:Processes.AbstractProcess,A<:Processes.AbstractLoopAlgorithm,C,R<:Processes.RepeatLifetime}
    marker("typed-ref enter")
    @assert Processes.isresolved(algo)
    @inline Processes.before_while(process)
    step_plan = @inline Processes.getplan(algo)
    step_wiring = @inline Processes.getwiring(step_plan)
    runtime_context = @inline Processes._merge_runtime_inputs(context, inputs)

    marker("typed-ref first _step begin")
    initial_context = @inline Processes._step!(step_plan, runtime_context, step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
    marker("typed-ref first _step end")
    @inline Processes.tick!(process)
    @inline Processes.inc!(process)

    start_idx = @inline Processes.loopidx(process)
    end_idx = @inline Processes.repeats(r)
    marker("typed-ref loop range start=$(start_idx) end=$(end_idx)")
    refcontext = Base.RefValue{typeof(initial_context)}(initial_context)
    marker("typed-ref ref_type=$(typeof(refcontext))")
    for idx in start_idx:end_idx
        marker("typed-ref for $(idx) _step begin")
        refcontext[] = @inline Processes._step!(step_plan, refcontext[], step_wiring, Processes.Namespace{nothing}(), process, r, Processes.Stable())
        marker("typed-ref for $(idx) _step end")
        @inline Processes.tick!(process)
        @inline Processes.inc!(process)
        (@inline Processes.breakcondition(r, process, refcontext[])) && break
    end
    marker("typed-ref after loop")
    return @inline Processes.after_while(process, algo, refcontext[], context)
end

"""Run one selected loop-shape repro stage."""
function run_stage(stage::S, repeat_count::I) where {S<:AbstractString,I<:Integer}
    probe = build_repro_worker()
    lifetime = Processes.Repeat(Int(repeat_count))
    inputs = (; phase_beta = probe.config.β)
    context = Processes.context(probe.worker)
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
