using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using InteractiveIsing
using SparseArrays
using Random
using Printf

const II = InteractiveIsing
const MANUAL_WEIGHT_RNG = Ref(Random.MersenneTwister(1))

random_manual_weight(; dr, c1 = nothing, c2 = nothing, dc = nothing) = randn(MANUAL_WEIGHT_RNG[])

"""
    make_layer(side, seed; symmetric)

Create a square layer with a random `WeightGenerator`. The `symmetric` keyword
selects the real constructor path:

- `symmetric=false`: old directed fill, one random draw for every ordered edge.
- `symmetric=true`: one random draw for `i < j`, mirrored through lazy vectors.
"""
function make_layer(side::Integer, seed::Integer; symmetric::Bool)
    rng = Random.MersenneTwister(seed)
    MANUAL_WEIGHT_RNG[] = rng
    wg = WeightGenerator(random_manual_weight, 1, rng; symmetric)
    layer = Layer(side, side, StateSet(-1.0, 1.0), wg, Discrete(), Coords(0, 1, 0); periodic = false)
    return layer, wg
end

function symmetry_error(rows, cols, vals, n)
    A = sparse(rows, cols, vals, n, n)
    D = A - transpose(A)
    nz = nonzeros(D)
    maxerr = isempty(nz) ? 0.0 : maximum(abs.(nz))
    return (; A, maxerr, nnzA = nnz(A), nnzD = nnz(D))
end

function time_variant(name, side, seed; symmetric::Bool)
    layer, wg = make_layer(side, seed; symmetric)
    n = length(layer)

    triplet_time = @elapsed rows, cols, vals = II.genLayerConnections(layer, Float64, wg, n)
    sparse_time = @elapsed stats = symmetry_error(rows, cols, vals, n)
    graph_time = @elapsed begin
        graph_layer, _ = make_layer(side, seed; symmetric)
        IsingGraph(graph_layer; precision = Float64)
    end
    total_time = triplet_time + sparse_time

    @printf(
        "%-10s side=%4d triplets=%9.6f sparse=%9.6f direct_total=%9.6f graph=%9.6f nnz=%8d asym_nnz=%8d maxerr=%10.6g rowtype=%s\n",
        name,
        side,
        triplet_time,
        sparse_time,
        total_time,
        graph_time,
        stats.nnzA,
        stats.nnzD,
        stats.maxerr,
        nameof(typeof(rows)),
    )
    return (; name, side, triplet_time, sparse_time, total_time, graph_time, stats)
end

"""
    run_manual_benchmark(; side = 128, repeats = 3)

Compare full adjacency construction for the directed and symmetric-lazy
`WeightGenerator` paths. This is a manual script, not an automated package test.
"""
function run_manual_benchmark(;
    side = parse(Int, get(ENV, "ISING_WG_SIDE", "128")),
    repeats = parse(Int, get(ENV, "ISING_WG_REPEATS", "3")),
)
    println("Random intra-layer WeightGenerator full-adjacency benchmark")
    println("Set ISING_WG_SIDE and ISING_WG_REPEATS to tune the manual run.")
    println()
    for rep in 1:repeats
        println("repeat $rep")
        time_variant("directed", side, rep; symmetric = false)
        time_variant("sym_lazy", side, rep; symmetric = true)
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_manual_benchmark()
end
