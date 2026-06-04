using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "..")))

using IsingLearning
using Random
using SparseArrays
using Statistics

const II = IsingLearning.InteractiveIsing
const StatefulAlgorithms = II.StatefulAlgorithms
const FT = Float32

const DEFAULT_STEPS = 100_000
const DEFAULT_WARMUP_STEPS = 2_000
const DEFAULT_NSPINS = 1_729
const DEFAULT_DEGREE = 48
const DEFAULT_TEMP = 1f0

"""Build one symmetric sparse coupling matrix with no diagonal entries."""
function random_symmetric_adjacency(
    nspins::I,
    degree::J,
    rng::R,
) where {I<:Integer,J<:Integer,R<:Random.AbstractRNG}
    rows = Int32[]
    cols = Int32[]
    vals = FT[]
    sizehint!(rows, 2 * Int(nspins) * Int(degree))
    sizehint!(cols, 2 * Int(nspins) * Int(degree))
    sizehint!(vals, 2 * Int(nspins) * Int(degree))

    # Duplicate random edges are allowed; `sparse` coalesces them.
    @inbounds for col in 1:Int(nspins), _ in 1:Int(degree)
        row = rand(rng, 1:Int(nspins))
        row == col && continue
        weight = FT(0.2) * (2f0 * rand(rng, FT) - 1f0) / sqrt(FT(degree))
        push!(rows, Int32(row))
        push!(cols, Int32(col))
        push!(vals, weight)
        push!(rows, Int32(col))
        push!(cols, Int32(row))
        push!(vals, weight)
    end

    adjacency = sparse(rows, cols, vals, Int(nspins), Int(nspins))
    SparseArrays.dropzeros!(adjacency)
    return SparseMatrixCSC{FT,Int32}(adjacency)
end

"""Create an Ising graph backed by a provided sparse `J`, field `b`, and state."""
function benchmark_graph(
    adjacency::SparseMatrixCSC{T,I},
    field::B,
    initial_state::S,
) where {T<:AbstractFloat,I<:Integer,B<:AbstractVector{T},S<:AbstractVector{T}}
    nspins = length(initial_state)
    layer = II.Layer(nspins, II.StateSet(-one(T), one(T)), II.Discrete(), II.Coords(0, 0, 0); periodic = false)
    graph = II.IsingGraph(
        layer,
        II.Bilinear() + II.MagField(b = _ -> copy(field));
        precision = T,
        adj = copy(adjacency),
        initial_state = copy(initial_state),
    )
    II.temp!(graph, T(DEFAULT_TEMP))
    return graph
end

"""Run a hand-written single-spin Metropolis loop over CSC columns."""
function custom_metropolis_loop!(
    graph::G,
    nsteps::I,
    rng::R,
) where {G,I<:Integer,R<:Random.AbstractRNG}
    state = II.state(graph)
    adjacency = II.adj(graph)
    rows = SparseArrays.rowvals(adjacency)
    colptr = SparseArrays.getcolptr(adjacency)
    nzvals = SparseArrays.nonzeros(adjacency)
    field = only(filter(h -> h isa II.MagField, II.hamiltonians(graph.hamiltonian))).b
    temperature = II.temp(graph)
    accepted = 0
    nspins = length(state)

    @inbounds for _ in 1:Int(nsteps)
        idx = rand(rng, 1:nspins)
        local_field = field[idx]
        for ptr in colptr[idx]:(colptr[idx + 1] - 1)
            local_field += nzvals[ptr] * state[rows[ptr]]
        end

        ΔE = 2 * state[idx] * local_field
        if ΔE <= zero(eltype(state)) || rand(rng, eltype(state)) < exp(-ΔE / temperature)
            state[idx] = -state[idx]
            accepted += 1
        end
    end
    return accepted
end

"""Build a `StatefulAlgorithms` routine that runs the library `Metropolis` stepper."""
function metropolis_process(graph::G, nsteps::I) where {G,I<:Integer}
    dynamics = II.Metropolis()
    algorithm = StatefulAlgorithms.resolve(StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics
        @repeat nsteps dynamics()
    end)
    return StatefulAlgorithms.Process(
        algorithm,
        StatefulAlgorithms.Init(:dynamics; model = graph);
        repeat = 1,
    )
end

"""Time only `run(process); wait(process)`, excluding graph and process construction."""
function process_elapsed_seconds(process::P) where {P}
    return @elapsed begin
        run(process)
        wait(process)
    end
end

"""Build matching graphs for custom and process timing from one Hamiltonian draw."""
function matching_graphs(; seed::I, nspins::J, degree::K) where {I<:Integer,J<:Integer,K<:Integer}
    rng = Random.MersenneTwister(Int(seed))
    adjacency = random_symmetric_adjacency(nspins, degree, rng)
    field = FT(0.05) .* (2f0 .* rand(rng, FT, Int(nspins)) .- 1f0)
    initial_state = [rand(rng, Bool) ? one(FT) : -one(FT) for _ in 1:Int(nspins)]
    return (
        benchmark_graph(adjacency, field, initial_state),
        benchmark_graph(adjacency, field, initial_state),
        length(SparseArrays.nonzeros(adjacency)),
    )
end

"""Run warmup and measured timings for custom Metropolis vs process Metropolis."""
function main()
    nsteps = parse(Int, get(ENV, "ISING_METRO_COMPARE_STEPS", string(DEFAULT_STEPS)))
    warmup_steps = parse(Int, get(ENV, "ISING_METRO_COMPARE_WARMUP_STEPS", string(DEFAULT_WARMUP_STEPS)))
    nspins = parse(Int, get(ENV, "ISING_METRO_COMPARE_NSPINS", string(DEFAULT_NSPINS)))
    degree = parse(Int, get(ENV, "ISING_METRO_COMPARE_DEGREE", string(DEFAULT_DEGREE)))
    seed = parse(Int, get(ENV, "ISING_METRO_COMPARE_SEED", "20260528"))

    # Warm both paths so the measured numbers are not dominated by compilation.
    warm_custom, warm_process, _ = matching_graphs(; seed, nspins, degree)
    custom_metropolis_loop!(warm_custom, warmup_steps, Random.MersenneTwister(seed + 1))
    warm_proc = metropolis_process(warm_process, warmup_steps)
    process_elapsed_seconds(warm_proc)

    custom_graph, process_graph, nnz = matching_graphs(; seed, nspins, degree)
    custom_rng = Random.MersenneTwister(seed + 2)

    custom_accepted = 0
    custom_seconds = @elapsed begin
        custom_accepted = custom_metropolis_loop!(custom_graph, nsteps, custom_rng)
    end

    process = metropolis_process(process_graph, nsteps)
    process_seconds = process_elapsed_seconds(process)

    println("steps,nspins,nnz,temp,custom_seconds,process_seconds,custom_steps_per_second,process_steps_per_second,process_over_custom,custom_acceptance")
    println(join((
        nsteps,
        nspins,
        nnz,
        DEFAULT_TEMP,
        round(custom_seconds; digits = 6),
        round(process_seconds; digits = 6),
        round(nsteps / custom_seconds; digits = 3),
        round(nsteps / process_seconds; digits = 3),
        round(process_seconds / custom_seconds; digits = 3),
        round(custom_accepted / nsteps; digits = 6),
    ), ","))
end

main()
