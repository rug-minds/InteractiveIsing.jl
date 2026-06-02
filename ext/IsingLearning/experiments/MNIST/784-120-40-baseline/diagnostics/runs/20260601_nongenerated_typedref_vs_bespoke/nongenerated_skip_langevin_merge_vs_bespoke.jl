using Dates
using Statistics

const RUN_DIR = @__DIR__
include(joinpath(RUN_DIR, "nongenerated_typedref_vs_bespoke.jl"))

"""
Run `LocalLangevin` as an in-place dynamics kernel without merging diagnostics.

This diagnostic specialization preserves graph/state mutation, but intentionally
does not persist per-step diagnostics such as `proposal`, `ΔE`, and
`gradient_max` into the immutable process context.
"""
@inline function Processes._step!(
    algo::LL,
    context::C,
    wiring::W,
    namespace::Processes.Namespace{Name},
    process::P,
    lifetime::LT,
    stability::S = Processes.Stable(),
) where {LL<:LocalLangevin,C<:Processes.AbstractContext,W<:Processes.Wiring{Tuple{},Tuple{}},Name,P<:Processes.AbstractProcess,LT<:Processes.Lifetime,S<:Processes.Stability}
    contextview = @inline view(context, algo, namespace)
    @inline Processes.step!(algo, contextview)
    return context
end

"""
Run `LocalLangevin` as an in-place dynamics kernel for routed/shared views.

This mirrors the normal view construction while skipping only the diagnostic
return merge.
"""
@inline function Processes._step!(
    algo::LL,
    context::C,
    wiring::W,
    namespace::Processes.Namespace{Name},
    process::P,
    lifetime::LT,
    stability::S = Processes.Stable(),
) where {LL<:LocalLangevin,C<:Processes.AbstractContext,W<:Processes.Wiring,Name,P<:Processes.AbstractProcess,LT<:Processes.Lifetime,S<:Processes.Stability}
    contextview = @inline view(
        context,
        algo,
        namespace;
        sharedcontexts = (@inline Processes.shares(wiring)),
        sharedvars = (@inline Processes.routes(wiring)),
    )
    @inline Processes.step!(algo, contextview)
    return context
end

"""Benchmark the diagnostic no-diagnostics-merge `LocalLangevin` specialization."""
function main()
    mkpath(RUN_DIR)
    csv_path = joinpath(RUN_DIR, "nongenerated_skip_langevin_merge_vs_bespoke.csv")
    rm(csv_path; force = true)

    config = langevin_learning_config()
    repeats = parse(Int, get(ENV, "ISING_MNIST_SKIP_MERGE_REPEATS", "1"))
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    looptype = Processes.NonGenerated()

    println(now(), " begin skip-merge NonGenerated vs bespoke threads=$(Threads.nthreads()) batchsize=$(config.batchsize) sweeps=$(config.sweeps) looptype=$(looptype)")
    flush(stdout)

    rows = NamedTuple[]
    for repeat_idx in 1:repeats
        direct_setup = build_layer(config)
        process_setup = build_layer(config)

        direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
        process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config, looptype)
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            repeat = repeat_idx,
            threads = Threads.nthreads(),
            batchsize = config.batchsize,
            sweeps = config.sweeps,
            looptype = string(looptype),
            direct_seconds = direct.wall,
            direct_seconds_per_example = direct.seconds_per_example,
            process_warmup_seconds = process.warmup_wall,
            process_seconds = process.wall,
            process_seconds_per_example = process.seconds_per_example,
            process_over_bespoke = process.wall / direct.wall,
        )
        append_row!(csv_path, row)
        push!(rows, row)
        println(now(), " rep=$(repeat_idx) direct_spe=$(direct.seconds_per_example) process_spe=$(process.seconds_per_example) over=$(process.wall / direct.wall)")
        flush(stdout)
    end

    process_over = map(row -> row.process_over_bespoke, rows)
    direct_spe = map(row -> row.direct_seconds_per_example, rows)
    process_spe = map(row -> row.process_seconds_per_example, rows)
    summary = (;
        repeats,
        direct_median_spe = median(direct_spe),
        process_median_spe = median(process_spe),
        process_over_bespoke_median = median(process_over),
        process_over_bespoke_mean = mean(process_over),
    )
    append_row!(joinpath(RUN_DIR, "nongenerated_skip_langevin_merge_vs_bespoke_summary.csv"), summary)
    println(now(), " summary=$(summary)")
    println(now(), " csv=$(csv_path)")
    flush(stdout)
    return (; rows, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
