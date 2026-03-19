using AbstractMCMC
using BenchmarkTools
using InteractiveIsing
using Random

struct PlainIsingModel{T} <: AbstractMCMC.AbstractModel
    side_length::Int
    coupling::T
    temperature::T
    initial_spins::Matrix{T}
end

mutable struct PlainIsingState{T}
    spins::Matrix{T}
end

struct PlainMetropolis <: AbstractMCMC.AbstractSampler end

nearest_neighbor_weight(; dr) = dr == 1 ? 1f0 : 0f0
const BENCHMARK_ISING_WG = @WG nearest_neighbor_weight NN = 1

@inline function periodic_neighbor_sum(spins, i::Int, j::Int)
    side_length = size(spins, 1)
    im = i == 1 ? side_length : i - 1
    ip = i == side_length ? 1 : i + 1
    jm = j == 1 ? side_length : j - 1
    jp = j == side_length ? 1 : j + 1
    @inbounds return spins[im, j] + spins[ip, j] + spins[i, jm] + spins[i, jp]
end

@inline function local_energy_delta(model::PlainIsingModel{T}, state::PlainIsingState{T}, i::Int, j::Int) where T
    spins = state.spins
    @inbounds spin = spins[i, j]
    return T(2) * model.coupling * spin * periodic_neighbor_sum(spins, i, j)
end

function AbstractMCMC.step(rng::AbstractRNG, model::PlainIsingModel{T}, sampler::PlainMetropolis; kwargs...) where T
    state = PlainIsingState(copy(model.initial_spins))
    return AbstractMCMC.step(rng, model, sampler, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::PlainIsingModel{T},
    ::PlainMetropolis,
    state::PlainIsingState{T};
    kwargs...,
) where T
    i = rand(rng, 1:model.side_length)
    j = rand(rng, 1:model.side_length)
    delta_e = local_energy_delta(model, state, i, j)

    if delta_e <= zero(T) || rand(rng, T) < exp(-delta_e / model.temperature)
        @inbounds state.spins[i, j] = -state.spins[i, j]
    end

    return state, state
end

function make_initial_spins(rng::AbstractRNG, side_length::Int, ::Type{T} = Float32) where T
    spins = Matrix{T}(undef, side_length, side_length)
    @inbounds for j in 1:side_length, i in 1:side_length
        spins[i, j] = rand(rng, Bool) ? one(T) : -one(T)
    end
    return spins
end

function make_interactiveising_graph(side_length::Int, temperature::T, spins::AbstractMatrix{T}) where T
    g = IsingGraph(side_length, side_length, BENCHMARK_ISING_WG, Discrete(), Ising(); precision = T)
    copyto!(state(g), vec(spins))
    temp!(g, temperature)
    return g
end

function make_interactiveising_kernel(side_length::Int, temperature::T, spins::AbstractMatrix{T}) where T
    g = make_interactiveising_graph(side_length, temperature, spins)
    hamiltonian = init!(g.hamiltonian, g)
    proposer = InteractiveIsing.get_proposer(g)
    rng = MersenneTwister(1234)
    return (; state = g, hamiltonian, proposer, rng)
end

function make_abstractmcmc_state(side_length::Int, temperature::T, spins::AbstractMatrix{T}) where T
    model = PlainIsingModel(side_length, one(T), temperature, copy(spins))
    sampler = PlainMetropolis()
    sampler_state = PlainIsingState(copy(spins))
    rng = MersenneTwister(1234)
    return model, sampler, rng, sampler_state
end

function run_interactiveising_steps!(kernel, nsteps::Int)
    (; state, hamiltonian, proposer, rng) = kernel
    last_output = nothing
    Ttype = eltype(state)
    for _ in 1:nsteps
        proposal = rand(rng, proposer)
        delta_e = InteractiveIsing.calculate(InteractiveIsing.ΔH(), hamiltonian, state, proposal)
        t = temp(state)
        if delta_e <= zero(Ttype) || rand(rng, Ttype) < exp(-delta_e / t)
            proposal = InteractiveIsing.accept(proposer, proposal)
        end
        InteractiveIsing.update!(Metropolis(), hamiltonian, (; state, hamiltonian, proposer, rng, proposal, ΔE = delta_e, T = t))
        last_output = proposal
    end
    return last_output
end

function run_abstractmcmc_steps!(
    rng::AbstractRNG,
    model::PlainIsingModel,
    sampler::PlainMetropolis,
    sampler_state::PlainIsingState,
    nsteps::Int,
)
    sample = nothing
    state = sampler_state
    for _ in 1:nsteps
        sample, state = AbstractMCMC.step(rng, model, sampler, state)
    end
    return sample
end

function benchmark_ising_metropolis(; side_length = 100, temperature = 2.0f0, nsteps = 10_000)
    seed_rng = MersenneTwister(20260319)
    initial_spins = make_initial_spins(seed_rng, side_length, Float32)

    interactive_setup = @benchmarkable make_interactiveising_kernel($side_length, $temperature, $initial_spins)
    abstractmcmc_setup = @benchmarkable make_abstractmcmc_state($side_length, $temperature, $initial_spins)

    interactive_steps = @benchmarkable begin
        kernel = make_interactiveising_kernel($side_length, $temperature, $initial_spins)
        run_interactiveising_steps!(kernel, $nsteps)
    end evals = 1

    abstractmcmc_steps = @benchmarkable begin
        model, sampler, rng, sampler_state = make_abstractmcmc_state($side_length, $temperature, $initial_spins)
        run_abstractmcmc_steps!(rng, model, sampler, sampler_state, $nsteps)
    end evals = 1

    return (
        interactive_setup = run(interactive_setup),
        abstractmcmc_setup = run(abstractmcmc_setup),
        interactive_steps = run(interactive_steps),
        abstractmcmc_steps = run(abstractmcmc_steps),
    )
end

function print_benchmark_summary(results)
    println("InteractiveIsing setup:")
    display(results.interactive_setup)
    println()
    println("AbstractMCMC setup:")
    display(results.abstractmcmc_setup)
    println()
    println("InteractiveIsing $((BenchmarkTools.params(results.interactive_steps)).evals)-eval step batch:")
    display(results.interactive_steps)
    println()
    println("AbstractMCMC $((BenchmarkTools.params(results.abstractmcmc_steps)).evals)-eval step batch:")
    display(results.abstractmcmc_steps)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = benchmark_ising_metropolis()
    print_benchmark_summary(results)
end
