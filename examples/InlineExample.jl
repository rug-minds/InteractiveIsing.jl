using AbstractMCMC
using BenchmarkTools
using InteractiveIsing
using Random
using SparseArrays

using InteractiveIsing
using InteractiveIsing.Processes
using BenchmarkTools
using LoopVectorization

g = IsingGraph(100,100,
                Discrete(),
                InteractiveIsing.Bilinear(),
                (@WG (;dr) -> dr == 1 ? 1 : 0 NN=1) ) 

comp = g.default_algorithm
p = InlineProcess(comp, Input(comp; state = g), lifetime = 100000)
@benchmark run(p)
# @code_warntype run(p)

# using Profile, PProf
# using Profile.Allocs

# Profile.Allocs.clear()
# Profile.Allocs.@profile run(p)
# Profile.Allocs.fetch()


const BENCHMARK_WG = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

struct PlainIsingModel{A, T} <: AbstractMCMC.AbstractModel
    adj::A
    temperature::T
end

mutable struct PlainIsingState{T}
    spins::Vector{T}
    dims::Tuple{Int, Int}
end

struct PlainMetropolis <: AbstractMCMC.AbstractSampler end

@inline function local_energy_delta(model::PlainIsingModel{A, T}, state::PlainIsingState{T}, idx::Int) where {A, T}
    spins = state.spins
    spin = @inbounds spins[idx]
    total = zero(T)
    rowvals = SparseArrays.rowvals(model.adj)
    weights = SparseArrays.nonzeros(model.adj)
    @turbo for ptr in SparseArrays.nzrange(model.adj, idx)
        total += weights[ptr] * spins[rowvals[ptr]]
    end
    return T(2) * spin * total
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::PlainIsingModel{A, T},
    ::PlainMetropolis,
    state::PlainIsingState{T};
    kwargs...,
) where {A, T}
    idx = rand(rng, eachindex(state.spins))
    delta_e = local_energy_delta(model, state, idx)

    if delta_e <= zero(T) || rand(rng, T) < exp(-delta_e / model.temperature)
        @inbounds state.spins[idx] = -state.spins[idx]
    end

    return state, state
end

const SIDE_LENGTH = 100
const NSTEPS = 100_000
const TEMPERATURE = 2.0f0
const RNG_SEED = 1234

const benchmark_graph = IsingGraph(SIDE_LENGTH, SIDE_LENGTH, Discrete(), BENCHMARK_WG)
const initial_spins = copy(vec(state(benchmark_graph)))
const model = PlainIsingModel(adj(benchmark_graph), TEMPERATURE)
const sampler = PlainMetropolis()
const rng = MersenneTwister(RNG_SEED)
const state = PlainIsingState(copy(initial_spins), (SIDE_LENGTH, SIDE_LENGTH))

function reset_ising!()
    copyto!(state.spins, initial_spins)
    Random.seed!(rng, RNG_SEED)
    return state
end

function run_steps!(nsteps::Int = NSTEPS)
    for _ in 1:nsteps
        AbstractMCMC.step(rng, model, sampler, state)
    end
    return state
end

run_100k!() = run_steps!(NSTEPS)

# Example:
# include("examples/InlineExample.jl")
@benchmark run_100k!()

