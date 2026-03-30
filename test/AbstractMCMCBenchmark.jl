using AbstractMCMC
using BenchmarkTools
using InteractiveIsing
using Random
using UUIDs: uuid1

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

function make_inline_process(func; repeats::Int, inputs = tuple(), overrides = tuple(), threaded = false)
    if !(func isa Processes.LoopAlgorithm)
        func = Processes.SimpleAlgo(func)
    end

    inputs = inputs isa Tuple ? inputs : (inputs,)
    overrides = overrides isa Tuple ? overrides : (overrides,)

    empty_context = Processes.ProcessContext(func)
    reg = Processes.getregistry(empty_context)

    named_inputs = Processes.to_named(reg, filter(x -> x isa Processes.Input, inputs)...)
    named_overrides = Processes.to_named(reg, filter(x -> x isa Processes.Override, overrides)...)

    lifetime = Processes.Repeat(repeats)
    td = Processes.TaskData(func; lifetime, inputs = named_inputs, overrides = named_overrides)
    context = Processes.init_context(td)

    return Processes.InlineProcess{typeof(td), typeof(context), threaded}(
        uuid1(),
        td,
        context,
        UInt(1),
        repeats,
        nothing,
        nothing,
    )
end

function make_interactiveising_process(side_length::Int, temperature::T, spins::AbstractMatrix{T}, nsteps::Int) where T
    g = make_interactiveising_graph(side_length, temperature, spins)
    algo = g.default_algorithm
    return make_inline_process(algo; repeats = nsteps, inputs = Processes.Input(algo, state = g))
end

function make_warmed_interactiveising_process(side_length::Int, temperature::T, spins::AbstractMatrix{T}, nsteps::Int) where T
    g = make_interactiveising_graph(side_length, temperature, spins)
    algo = g.default_algorithm

    warm_process = make_inline_process(algo; repeats = 1, inputs = Processes.Input(algo, state = g))
    run(warm_process)

    copyto!(state(g), vec(spins))
    temp!(g, temperature)

    process = make_inline_process(algo; repeats = nsteps, inputs = Processes.Input(algo, state = g))
    return (; process, graph = g)
end

function make_interactiveising_runner(side_length::Int, temperature::T, spins::AbstractMatrix{T}, nsteps::Int) where T
    g = make_interactiveising_graph(side_length, temperature, spins)
    algo = g.default_algorithm
    process = make_inline_process(algo; repeats = nsteps, inputs = Processes.Input(algo, state = g))
    return (; process, graph = g)
end

function reset_interactiveising_runner!(runner, temperature, spins)
    copyto!(state(runner.graph), vec(spins))
    temp!(runner.graph, temperature)
    Processes.reset!(runner.process)
    return runner
end

function make_abstractmcmc_state(side_length::Int, temperature::T, spins::AbstractMatrix{T}) where T
    model = PlainIsingModel(side_length, one(T), temperature, copy(spins))
    sampler = PlainMetropolis()
    sampler_state = PlainIsingState(copy(spins))
    rng = MersenneTwister(1234)
    return model, sampler, rng, sampler_state
end

function make_abstractmcmc_runner(side_length::Int, temperature::T, spins::AbstractMatrix{T}) where T
    model, sampler, rng, sampler_state = make_abstractmcmc_state(side_length, temperature, spins)
    return (; model, sampler, rng, sampler_state)
end

function make_warmed_abstractmcmc_runner(side_length::Int, temperature::T, spins::AbstractMatrix{T}, nsteps::Int) where T
    runner = make_abstractmcmc_runner(side_length, temperature, spins)
    run_abstractmcmc_steps!(runner.rng, runner.model, runner.sampler, runner.sampler_state, 1)

    runner = make_abstractmcmc_runner(side_length, temperature, spins)
    return runner
end

function reset_abstractmcmc_runner!(runner, spins)
    copyto!(runner.sampler_state.spins, spins)
    Random.seed!(runner.rng, 1234)
    return runner
end

function run_interactiveising_steps!(process::Processes.InlineProcess)
    return run(process)
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
    interactive_runner = make_warmed_interactiveising_process(side_length, temperature, initial_spins, nsteps)
    abstractmcmc_runner = make_warmed_abstractmcmc_runner(side_length, temperature, initial_spins, nsteps)

    reset_interactiveising_runner!(interactive_runner, temperature, initial_spins)
    run(interactive_runner.process)
    reset_interactiveising_runner!(interactive_runner, temperature, initial_spins)

    reset_abstractmcmc_runner!(abstractmcmc_runner, initial_spins)
    run_abstractmcmc_steps!(
        abstractmcmc_runner.rng,
        abstractmcmc_runner.model,
        abstractmcmc_runner.sampler,
        abstractmcmc_runner.sampler_state,
        1,
    )
    reset_abstractmcmc_runner!(abstractmcmc_runner, initial_spins)

    interactive_steps = @benchmark begin
        run($interactive_runner.process)
        nothing
    end setup = (
        reset_interactiveising_runner!($interactive_runner, $temperature, $initial_spins)
    ) evals = 1

    abstractmcmc_steps = @benchmark begin
        run_abstractmcmc_steps!(
            $abstractmcmc_runner.rng,
            $abstractmcmc_runner.model,
            $abstractmcmc_runner.sampler,
            $abstractmcmc_runner.sampler_state,
            $nsteps,
        )
        nothing
    end setup = (
        reset_abstractmcmc_runner!($abstractmcmc_runner, $initial_spins)
    ) evals = 1

    display(interactive_steps)
    display(abstractmcmc_steps)

    reset_interactiveising_runner!(interactive_runner, temperature, initial_spins)
    run_interactiveising_steps!(interactive_runner.process)

    reset_abstractmcmc_runner!(abstractmcmc_runner, initial_spins)
    run_abstractmcmc_steps!(
        abstractmcmc_runner.rng,
        abstractmcmc_runner.model,
        abstractmcmc_runner.sampler,
        abstractmcmc_runner.sampler_state,
        nsteps,
    )

    return (
        interactive_steps,
        abstractmcmc_steps,
        interactive_graph = interactive_runner.graph,
        interactive_process = interactive_runner.process,
        abstract_model = abstractmcmc_runner.model,
        abstract_sampler = abstractmcmc_runner.sampler,
        abstract_rng = abstractmcmc_runner.rng,
        abstract_state = abstractmcmc_runner.sampler_state,
    )
end

run_benchmark(; kwargs...) = benchmark_ising_metropolis(; kwargs...)

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
