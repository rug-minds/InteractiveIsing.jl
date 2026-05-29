using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Random
using SparseArrays

const NGRAPHS = parse(Int, get(ENV, "ISING_MNIST_CONSTRUCT_GRAPHS", "64"))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_CONSTRUCT_HIDDEN", "7840"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_CONSTRUCT_OUTPUT_REPLICAS", "4"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_CONSTRUCT_WEIGHT_SCALE", "0.005"))
const KEEP_GRAPHS = get(ENV, "ISING_MNIST_CONSTRUCT_KEEP", "false") == "true"
const OUTDIR = get(ENV, "ISING_MNIST_CONSTRUCT_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_construction_stability")))

mkpath(OUTDIR)

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
    append_csv_row!(path, row)

Append one named-tuple row to a CSV file, writing the header on first use.
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
    validate_graph_topology(graph)

Check sparse CSC topology immediately after graph construction. This verifies
that row indices and column pointers are valid before any dynamics can mutate
the graph state.
"""
function validate_graph_topology(graph::G) where {G}
    sp = InteractiveIsing.adj(graph).sp
    n = InteractiveIsing.nstates(graph)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    colptr = SparseArrays.getcolptr(sp)

    length(rowval) == length(nzval) || error("rowval length $(length(rowval)) != nzval length $(length(nzval))")
    size(sp, 1) == n || error("sparse row count $(size(sp, 1)) != nstates $n")
    size(sp, 2) == n || error("sparse column count $(size(sp, 2)) != nstates $n")
    first(colptr) == 1 || error("first colptr is $(first(colptr)), expected 1")
    last(colptr) == length(rowval) + 1 || error("last colptr is $(last(colptr)), expected $(length(rowval) + 1)")

    bad_col = findfirst(idx -> colptr[idx] > colptr[idx + 1], 1:(length(colptr) - 1))
    isnothing(bad_col) || error("colptr decreases at column $bad_col")

    min_row = minimum(rowval)
    max_row = maximum(rowval)
    1 <= min_row || error("rowval minimum $min_row is below 1")
    max_row <= n || error("rowval maximum $max_row is above nstates $n")

    return (; nstates = n, nnz = length(nzval), min_row, max_row)
end

"""
    build_graph(seed)

Construct one MNIST graph with the same 10x-hidden architecture as the manager
stability runs.
"""
function build_graph(seed::T) where {T<:Integer}
    return MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(Int(seed)),
    )
end

"""
    main()

Repeatedly build MNIST graphs and validate sparse topology before any dynamics
or process-manager execution.
"""
function main()
    csv_path = joinpath(OUTDIR, "construction_stability.csv")
    kept = Any[]
    println(
        "MNIST construction stability graphs=", NGRAPHS,
        " hidden=", HIDDEN,
        " output_replicas=", OUTPUT_REPLICAS,
        " keep_graphs=", KEEP_GRAPHS,
        " threads=", Threads.nthreads(),
    )
    println("memory_start=", process_memory_mb())

    for idx in 1:NGRAPHS
        graph = nothing
        topo = nothing
        seconds = @elapsed begin
            graph = build_graph(200_000 + idx)
            topo = validate_graph_topology(graph)
            KEEP_GRAPHS ? push!(kept, graph) : (graph = nothing)
        end
        memory = process_memory_mb()
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            idx,
            hidden = HIDDEN,
            output_replicas = OUTPUT_REPLICAS,
            seconds,
            nstates = topo.nstates,
            nnz = topo.nnz,
            min_row = topo.min_row,
            max_row = topo.max_row,
            working_set_mb = memory.working_set,
            private_mb = memory.private,
        )
        append_csv_row!(csv_path, row)
        println(row)
        flush(stdout)
        KEEP_GRAPHS || (idx % 4 == 0 && GC.gc())
    end

    println("memory_done=", process_memory_mb())
    println("Saved construction stability CSV: ", csv_path)
end

main()
