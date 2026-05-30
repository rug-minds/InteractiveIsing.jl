using Dates
using Statistics

const WARMED_BENCH_DIR = @__DIR__
const WARMED_BENCH_SOURCE = normpath(joinpath(
    @__DIR__,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))
include(WARMED_BENCH_SOURCE)

"""Return the output CSV path for the warmed full contrastive learning benchmark."""
function warmed_bench_csv_path()
    return joinpath(WARMED_BENCH_DIR, "warmed_contrastive_learning_process_vs_bespoke.csv")
end

"""Append one benchmark row to the warmed comparison CSV."""
function append_warmed_bench_row!(row::R) where {R<:NamedTuple}
    path = warmed_bench_csv_path()
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Run one explicit warmup and one measured full-learning minibatch for each serial path."""
function run_warmed_benchmark!()
    mkpath(WARMED_BENCH_DIR)
    rm(warmed_bench_csv_path(); force = true)

    batchsize = parse(Int, get(ENV, "ISING_MNIST_WARMED_BENCH_BATCHSIZE", "1"))
    config = updated_config(langevin_learning_config(); batchsize)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    # Build once per path, then use each timing function's internal warmup before measurement.
    direct_setup = build_layer(config)
    process_setup = build_layer(config)

    println("starting_direct")
    direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
    println("finished_direct seconds_per_example=$(direct.seconds_per_example)")

    println("starting_serial_process")
    serial_process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config)
    println("finished_serial_process seconds_per_example=$(serial_process.seconds_per_example)")

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        architecture = "120-40_field_input_from_784",
        benchmark = "warmed_full_contrastive_learning_minibatch",
        threads = Threads.nthreads(),
        batchsize = config.batchsize,
        sweeps = config.sweeps,
        beta = config.β,
        temp = config.temp,
        stepsize = config.stepsize,
        relaxation_steps = direct_setup.relaxation_steps,
        direct_bespoke_seconds = direct.wall,
        direct_bespoke_seconds_per_example = direct.seconds_per_example,
        serial_process_seconds = serial_process.wall,
        serial_process_seconds_per_example = serial_process.seconds_per_example,
        process_over_bespoke = serial_process.wall / direct.wall,
    )
    append_warmed_bench_row!(row)

    summary_path = joinpath(WARMED_BENCH_DIR, "summary.md")
    open(summary_path, "w") do io
        println(io, "# Warmed Full Contrastive Learning Benchmark")
        println(io)
        println(io, "- Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "- Benchmark: full contrastive learning minibatch, not dynamics-only")
        println(io, "- Paths: direct bespoke vs serial `Processes.Process`")
        println(io, "- Manager path: not included")
        println(io, "- Direct seconds/example: `", direct.seconds_per_example, "`")
        println(io, "- Serial Process seconds/example: `", serial_process.seconds_per_example, "`")
        println(io, "- Process/direct ratio: `", serial_process.wall / direct.wall, "`")
    end

    println(row)
    println("csv=" * warmed_bench_csv_path())
    println("summary=" * summary_path)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_warmed_benchmark!()
end
