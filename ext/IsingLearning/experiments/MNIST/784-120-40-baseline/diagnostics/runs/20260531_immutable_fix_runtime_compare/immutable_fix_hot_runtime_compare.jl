using Dates
using Statistics

const RUN_DIR = @__DIR__
const HELPER = normpath(joinpath(
    RUN_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
const REPO_ROOT = normpath(joinpath(RUN_DIR, "..", "..", "..", "..", "..", "..", "..", ".."))
include(HELPER)

"""Print a timestamped progress marker and flush it immediately for live polling."""
function logline(message::M) where {M<:AbstractString}
    println(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), " ", message)
    flush(stdout)
    return nothing
end

"""Append one durable comparison row to this diagnostic run directory."""
function append_runtime_compare_row!(row::R) where {R<:NamedTuple}
    path = joinpath(RUN_DIR, "immutable_fix_hot_runtime_compare.csv")
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run the hot full-learning comparison for bespoke direct code and serial Processes code."""
function main()
    mkpath(RUN_DIR)
    batchsize = parse(Int, get(ENV, "ISING_RUNTIME_COMPARE_BATCHSIZE", "1"))
    repeats = parse(Int, get(ENV, "ISING_RUNTIME_COMPARE_REPEATS", "1"))
    config = updated_config(langevin_learning_config(); batchsize = batchsize)

    logline("begin runtime compare")
    logline("threads=$(Threads.nthreads()) batchsize=$(config.batchsize) repeats=$repeats sweeps=$(config.sweeps)")
    logline("helper=$HELPER")
    logline("loading mnist data")
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    rows = NamedTuple[]

    for repeat_idx in 1:repeats
        logline("repeat=$repeat_idx build direct setup")
        direct_setup = build_layer(config)
        logline("repeat=$repeat_idx time direct bespoke full learning minibatch")
        direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
        logline("repeat=$repeat_idx direct wall=$(direct.wall) seconds_per_example=$(direct.seconds_per_example)")

        logline("repeat=$repeat_idx build serial process setup")
        process_setup = build_layer(config)
        logline("repeat=$repeat_idx time serial Processes full learning minibatch")
        serial_process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config)
        logline("repeat=$repeat_idx process wall=$(serial_process.wall) seconds_per_example=$(serial_process.seconds_per_example)")

        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            processes_head = readchomp(`git -C $(joinpath(REPO_ROOT, "deps", "Processes")) rev-parse --short HEAD`),
            threads = Threads.nthreads(),
            batchsize = config.batchsize,
            sweeps = config.sweeps,
            beta = config.β,
            temp = config.temp,
            stepsize = config.stepsize,
            relaxation_steps = direct_setup.relaxation_steps,
            repeat = repeat_idx,
            direct_bespoke_seconds = direct.wall,
            direct_bespoke_seconds_per_example = direct.seconds_per_example,
            serial_process_seconds = serial_process.wall,
            serial_process_seconds_per_example = serial_process.seconds_per_example,
            process_over_bespoke = serial_process.wall / direct.wall,
            process_speedup_vs_bespoke = direct.wall / serial_process.wall,
        )
        append_runtime_compare_row!(row)
        push!(rows, row)
        logline("repeat=$repeat_idx row=$row")
    end

    process_over = map(row -> row.process_over_bespoke, rows)
    speedup = map(row -> row.process_speedup_vs_bespoke, rows)
    logline("summary process_over_bespoke_mean=$(mean(process_over)) process_over_bespoke_median=$(median(process_over))")
    logline("summary process_speedup_vs_bespoke_mean=$(mean(speedup)) process_speedup_vs_bespoke_median=$(median(speedup))")
    csv_path = joinpath(RUN_DIR, "immutable_fix_hot_runtime_compare.csv")
    logline("csv=$csv_path")
    return rows
end

main()
