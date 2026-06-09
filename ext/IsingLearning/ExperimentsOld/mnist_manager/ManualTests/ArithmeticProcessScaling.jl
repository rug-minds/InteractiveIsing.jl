using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Random
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_ARITH_WORKERS", "1,16,32"), ","))
const LENGTHS = parse.(Int, split(get(ENV, "ISING_ARITH_LENGTHS", "4096"), ","))
const ROUNDS = parse.(Int, split(get(ENV, "ISING_ARITH_ROUNDS", "20000"), ","))
const INNER_OPS = parse.(Int, split(get(ENV, "ISING_ARITH_INNER_OPS", "1,32,128"), ","))
const OUTDIR = get(ENV, "ISING_ARITH_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_arithmetic_process_scaling")))

mkpath(OUTDIR)

"""
    ArithmeticStateStep(len, rounds, inner_ops, seed)

Manager-owned `ProcessAlgorithm` that mutates only worker-local floating-point
state. `inner_ops` controls arithmetic intensity: low values are memory-write
heavy, high values are compute-heavy.
"""
struct ArithmeticStateStep{I<:Integer} <: StatefulAlgorithms.ProcessAlgorithm
    len::I
    rounds::I
    inner_ops::I
    seed::I
end

"""
    StatefulAlgorithms.init(step, context)

Allocate one private vector per worker and initialize deterministic scalar
coefficients for the arithmetic loop.
"""
function StatefulAlgorithms.init(step::S, context) where {S<:ArithmeticStateStep}
    rng = Random.MersenneTwister(step.seed + get(context, :worker_idx, 0))
    data = rand(rng, Float64, Int(step.len))
    return (;
        data,
        checksum = Ref(sum(data)),
        a = Ref(1.0000001192092896),
        b = Ref(0.0000002384185791),
        c = Ref(0.9999998807907104),
    )
end

"""
    StatefulAlgorithms.step!(step, context)

Run the worker-local arithmetic workload and store a checksum so the loop cannot
be removed as dead work.
"""
function StatefulAlgorithms.step!(step::S, context::C) where {S<:ArithmeticStateStep,C}
    data = context.data
    a = context.a[]
    b = context.b[]
    c = context.c[]
    total = 0.0

    @inbounds for _ in 1:Int(step.rounds)
        for idx in eachindex(data)
            x = data[idx]
            for _ in 1:Int(step.inner_ops)
                x = muladd(x, a, b)
                x = x - floor(x * 0.000244140625) * c
            end
            data[idx] = x
            total += x
        end
    end

    context.checksum[] = total
    return (; checksum = total)
end

"""
    append_csv_row!(path, row)

Append a named-tuple row to a CSV file, creating the header when needed.
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
    build_manager(step, nworkers)

Create a `ProcessManager` whose workers each own an independent arithmetic
process and private state vector.
"""
function build_manager(step::S, nworkers::I) where {S<:ArithmeticStateStep,I<:Integer}
    recipe = (;
        makeworker = (idx, manager) -> Process(
            ArithmeticStateStep(step.len, step.rounds, step.inner_ops, step.seed + idx);
            repeats = 1,
        ),
        loadjob! = (slot, job, manager) -> resetworker!(slot),
    )
    return ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        sync_policy = NoSync(),
        poll_interval = 0.0,
        job_type = Int,
        result_type = Any,
    )
end

"""
    run_manager_once!(manager, nworkers)

Run one job per worker using the manager's native threaded scheduler.
"""
function run_manager_once!(manager::M, nworkers::I) where {M<:ProcessManager,I<:Integer}
    runthreaded!(manager, 1:Int(nworkers), Dynamic())
    return manager
end

"""
    run_config(len, rounds, inner_ops, nworkers)

Benchmark one arithmetic intensity and worker count.
"""
function run_config(len::L, rounds::R, inner_ops::O, nworkers::N) where {L<:Integer,R<:Integer,O<:Integer,N<:Integer}
    step = ArithmeticStateStep(Int(len), Int(rounds), Int(inner_ops), 210_000)

    warmup = build_manager(step, 1)
    run_manager_once!(warmup, 1)
    close(warmup)

    manager = build_manager(step, nworkers)
    total_seconds = @elapsed run_manager_once!(manager, nworkers)
    checksums = Float64[]
    for slot in slots(manager)
        algo = only(StatefulAlgorithms.getalgos(StatefulAlgorithms.getalgo(slot.worker)))
        push!(checksums, StatefulAlgorithms.context(slot.worker)[algo].checksum[])
    end
    close(manager)

    scalar_updates = Int(len) * Int(rounds) * Int(nworkers)
    nominal_ops = scalar_updates * Int(inner_ops) * 4
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        workers = Int(nworkers),
        threads = Threads.nthreads(),
        len = Int(len),
        rounds = Int(rounds),
        inner_ops = Int(inner_ops),
        scalar_updates,
        nominal_ops,
        total_seconds,
        updates_per_second = scalar_updates / total_seconds,
        nominal_gops_per_second = nominal_ops / total_seconds / 1.0e9,
        checksum = sum(checksums),
    )
    append_csv_row!(joinpath(OUTDIR, "arithmetic_process_scaling.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Run worker-local arithmetic scaling tests through `ProcessManager`.
"""
function main()
    println(
        "Arithmetic ProcessAlgorithm scaling workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " lengths=", LENGTHS,
        " rounds=", ROUNDS,
        " inner_ops=", INNER_OPS,
    )
    for len in LENGTHS
        for rounds in ROUNDS
            for inner_ops in INNER_OPS
                for nworkers in WORKERS
                    run_config(len, rounds, inner_ops, nworkers)
                end
            end
        end
    end
    println("Saved outputs in ", OUTDIR)
end

main()
