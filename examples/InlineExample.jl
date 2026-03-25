using AbstractMCMC
using BenchmarkTools
using InteractiveIsing
using InteractiveIsing.Processes
using Random

# Minimal steady-state comparison:
# - InteractiveIsing: nearest-neighbor Bilinear Hamiltonian, run through InlineProcess
# - AbstractMCMC: minimal hand-written single-spin Metropolis kernel

const SIDE = 100
const NSTEPS = 100_000
const TEMP = 2.0f0
const SEED = 1234
const NN_WG = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1


struct PlainIsingModel{T} <: AbstractMCMC.AbstractModel
    side::Int
    temperature::T
end

mutable struct PlainIsingState{T}
    spins::Matrix{T}
end

struct PlainMetropolis <: AbstractMCMC.AbstractSampler end


@inline function nearest_neighbor_sum(spins, i::Int, j::Int)
    side = size(spins, 1)
    im = i == 1 ? side : i - 1
    ip = i == side ? 1 : i + 1
    jm = j == 1 ? side : j - 1
    jp = j == side ? 1 : j + 1
    @inbounds return spins[im, j] + spins[ip, j] + spins[i, jm] + spins[i, jp]
end

@inline function local_energy_delta(model::PlainIsingModel{T}, state::PlainIsingState{T}, i::Int, j::Int) where {T}
    spins = state.spins
    @inbounds spin = spins[i, j]
    return T(2) * spin * nearest_neighbor_sum(spins, i, j)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::PlainIsingModel{T},
    ::PlainMetropolis,
    state::PlainIsingState{T};
    kwargs...,
) where {T}
    i = rand(rng, 1:model.side)
    j = rand(rng, 1:model.side)
    ΔE = local_energy_delta(model, state, i, j)

    if ΔE <= zero(T) || rand(rng, T) < exp(-ΔE / model.temperature)
        @inbounds state.spins[i, j] = -state.spins[i, j]
    end

    return state, state
end


mutable struct PackageRunner{G,P,V,T}
    graph::G
    process::P
    reference_spins::V
    temperature::T
end

mutable struct AbstractRunner{M,S,R}
    model::M
    sampler::S
    rng::R
    state::PlainIsingState{Float32}
    reference_spins::Matrix{Float32}
end


function make_initial_spins(; side = SIDE, seed = SEED)
    rng = MersenneTwister(seed)
    spins = Vector{Float32}(undef, side * side)
    @inbounds for i in eachindex(spins)
        spins[i] = rand(rng, Bool) ? 1f0 : -1f0
    end
    return spins
end

function make_package_runner(initial_spins; side = SIDE, temperature = TEMP, nsteps = NSTEPS)
    g = IsingGraph(side, side, Discrete(), InteractiveIsing.Bilinear(), NN_WG)
    copyto!(InteractiveIsing.graphstate(g), initial_spins)
    temp!(g, temperature)

    algo = g.default_algorithm
    p = InlineProcess(algo, Input(algo; state = g), lifetime = nsteps)
    return PackageRunner(g, p, copy(initial_spins), temperature)
end

function make_abstract_runner(initial_spins; side = SIDE, temperature = TEMP, seed = SEED)
    model = PlainIsingModel(side, temperature)
    sampler = PlainMetropolis()
    rng = MersenneTwister(seed)
    spins = reshape(copy(initial_spins), side, side)
    return AbstractRunner(model, sampler, rng, PlainIsingState(spins), copy(spins))
end


function reset!(runner::PackageRunner)
    copyto!(InteractiveIsing.graphstate(runner.graph), runner.reference_spins)
    temp!(runner.graph, runner.temperature)
    Processes.reset!(runner.process)
    return runner
end

function reset!(runner::AbstractRunner)
    copyto!(runner.state.spins, runner.reference_spins)
    Random.seed!(runner.rng, SEED)
    return runner
end


function run_abstractmcmc!(runner::AbstractRunner, nsteps::Int = NSTEPS)
    sample = nothing
    state = runner.state
    @inbounds for _ in 1:nsteps
        sample, state = AbstractMCMC.step(runner.rng, runner.model, runner.sampler, state)
    end
    return sample
end


function warmup!(pkg::PackageRunner, plain::AbstractRunner)
    reset!(pkg)
    run(pkg.process)
    pkg.reference_spins .= InteractiveIsing.graphstate(pkg.graph)

    reset!(plain)
    run_abstractmcmc!(plain, 1)
    plain.reference_spins .= reshape(pkg.reference_spins, size(plain.state.spins))

    return nothing
end

function compare_ising_speed(; side = SIDE, temperature = TEMP, nsteps = NSTEPS)
    initial_spins = make_initial_spins(; side)
    pkg = make_package_runner(initial_spins; side, temperature, nsteps)
    plain = make_abstract_runner(initial_spins; side, temperature)

    warmup!(pkg, plain)

    process = pkg.process
    interactiveising = @benchmark run($process) setup = (reset!($pkg))
    abstractmcmc = @benchmark run_abstractmcmc!($plain, $nsteps) setup = (reset!($plain))

    display(interactiveising)
    display(abstractmcmc)

    return (; interactiveising, abstractmcmc, pkg, plain)
end


# Usage:
# include("examples/InlineExample.jl")
compare_ising_speed(nsteps = 1000000)
