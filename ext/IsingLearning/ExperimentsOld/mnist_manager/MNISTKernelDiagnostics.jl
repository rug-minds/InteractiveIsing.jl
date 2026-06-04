using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using Random
using SparseArrays
using Statistics

const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_DIAG_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_DIAG_OUTPUT_REPLICAS", "4"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_DIAG_WEIGHT_SCALE", "0.005"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_DIAG_TEMP", "0.001"))
const SAMPLE_COLUMNS = parse(Int, get(ENV, "ISING_MNIST_DIAG_SAMPLE_COLUMNS", "0"))
const RUN_STEPS = parse(Int, get(ENV, "ISING_MNIST_DIAG_RUN_STEPS", "0"))

"""
    process_memory_mb()

Return current process working set and private memory in MiB on Windows.
"""
function process_memory_mb()
    if Sys.iswindows()
        cmd = `powershell -NoProfile -Command "\$p=Get-Process -Id $(getpid()); \"\$([math]::Round(\$p.WorkingSet64/1MB,1)),\$([math]::Round(\$p.PrivateMemorySize64/1MB,1))\""`
        fields = split(strip(read(cmd, String)), ",")
        return (; working_set = parse(Float64, fields[1]), private = parse(Float64, fields[2]))
    end
    return (; working_set = NaN, private = NaN)
end

"""
    checked_column_contraction(i, v, sp)

Run the sparse neighbor sum with normal Julia indexing so malformed row or
column pointers throw a bounds error instead of producing an unchecked load.
"""
function checked_column_contraction(i::I, v::V, sp::S) where {I<:Integer,V<:AbstractVector,S<:SparseMatrixCSC}
    checkbounds(axes(sp, 2), Int(i))
    total = zero(eltype(sp))
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    for ptr in nzrange(sp, Int(i))
        row = rowval[ptr]
        checkbounds(v, row)
        total += nzval[ptr] * v[row]
    end
    return total
end

"""
    validate_sparse_csc(sp, state_length)

Check the structural invariants that `@turbo` assumes while reading CSC storage.
"""
function validate_sparse_csc(sp::S, state_length::T) where {S<:SparseMatrixCSC,T<:Integer}
    nrows, ncols = size(sp)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    colptr = SparseArrays.getcolptr(sp)

    length(nzval) == length(rowval) || error("nzval length $(length(nzval)) != rowval length $(length(rowval))")
    nrows == Int(state_length) || error("sparse rows $nrows != state length $state_length")
    ncols == Int(state_length) || error("sparse cols $ncols != state length $state_length")
    first(colptr) == 1 || error("first colptr is $(first(colptr)), expected 1")
    last(colptr) == length(rowval) + 1 || error("last colptr is $(last(colptr)), expected $(length(rowval) + 1)")

    bad_col = findfirst(i -> colptr[i] > colptr[i + 1], 1:(length(colptr) - 1))
    isnothing(bad_col) || error("colptr decreases at column $bad_col")

    min_row = minimum(rowval)
    max_row = maximum(rowval)
    1 <= min_row || error("rowval minimum $min_row is below 1")
    max_row <= nrows || error("rowval maximum $max_row is above row count $nrows")

    return (; nrows, ncols, nnz = length(nzval), min_row, max_row)
end

"""
    validate_active_indices(graph)

Check active sampling indices against graph state and sparse matrix dimensions.
"""
function validate_active_indices(graph::G) where {G}
    active = collect(InteractiveIsing.sampling_indices(graph))
    n = InteractiveIsing.nstates(graph)
    isempty(active) && error("active index set is empty")
    min_active = minimum(active)
    max_active = maximum(active)
    1 <= min_active || error("active index minimum $min_active is below 1")
    max_active <= n || error("active index maximum $max_active is above nstates $n")
    return (; count = length(active), min_active, max_active)
end

"""
    validate_checked_contractions(graph; sample_columns)

Run checked sparse contractions over either all active columns or a random
sample of active columns.
"""
function validate_checked_contractions(graph::G; sample_columns::T = 0) where {G,T<:Integer}
    active = collect(InteractiveIsing.sampling_indices(graph))
    if sample_columns > 0 && sample_columns < length(active)
        rng = Random.MersenneTwister(1234)
        shuffle!(rng, active)
        resize!(active, Int(sample_columns))
    end

    sp = InteractiveIsing.adj(graph).sp
    v = InteractiveIsing.state(graph)
    total = zero(eltype(graph))
    for idx in active
        total += checked_column_contraction(idx, v, sp)
    end
    return (; checked_columns = length(active), checksum = total)
end

"""
    run_langevin_steps!(graph, nsteps)

Optionally run a small single-process LocalLangevin loop after structural
checks, to compare pure single-thread dynamics against manager runs.
"""
function run_langevin_steps!(graph::G, nsteps::T) where {G,T<:Integer}
    nsteps <= 0 && return (; steps = 0, seconds = 0.0)
    algorithm = LocalLangevin(stepsize = 0.5f0, adjusted = false)
    context = StatefulAlgorithms.init(algorithm, (; model = graph))
    seconds = @elapsed begin
        for _ in 1:Int(nsteps)
            StatefulAlgorithms.step!(algorithm, context)
        end
    end
    return (; steps = Int(nsteps), seconds)
end

"""
    main()

Run MNIST graph/kernel diagnostics for the large sparse neighbor-sum workload.
"""
function main()
    println(
        "MNIST kernel diagnostics hidden=", HIDDEN,
        " output_replicas=", OUTPUT_REPLICAS,
        " sample_columns=", SAMPLE_COLUMNS,
        " run_steps=", RUN_STEPS,
        " threads=", Threads.nthreads(),
    )
    println("memory_before=", process_memory_mb())
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(70_000),
    )
    temp!(graph, TEMP)
    GC.gc()
    println("memory_after_graph=", process_memory_mb())
    println("graph_states=", InteractiveIsing.nstates(graph))
    println("sparse=", validate_sparse_csc(InteractiveIsing.adj(graph).sp, InteractiveIsing.nstates(graph)))
    println("active=", validate_active_indices(graph))
    println("checked_contractions=", validate_checked_contractions(graph; sample_columns = SAMPLE_COLUMNS))
    println("langevin=", run_langevin_steps!(graph, RUN_STEPS))
    GC.gc()
    println("memory_done=", process_memory_mb())
end

main()
