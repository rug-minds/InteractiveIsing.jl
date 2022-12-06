# module Benchmarking
# include("oldbenchmarkfuncs.jl")
using BenchmarkTools
using Random

macro includetextfile(pathsymbs...)
    path = joinpath(@__DIR__, string.(pathsymbs)...)
    esc(Meta.parse(read(open(path), String)))
end

randclamp = Float32.(rand(500^2).* 2 .- 1)

function clampE(idx)
    return randclamp[idx]
end

export benchmarkIsing

function benchmarkIsing(sim, bfunc, iterations = 80000000)
    function clampE(idx)
        return randclamp[idx]
    end
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    g = graph(sim)
    
    isRunning(sim,false)
    pauseSim(sim)
    state(g) .= initRandomState(g)
    # bfunc(sim,iterations, rng)
    bfunc(sim, rng)
end

function updateSim1(sim::IsingSim, rng)
    g = sim.layers[1]
    ghtype = htype(g)
    g_iterator = ising_it(g,ghtype)
    gstate = g.state
    gadj = g.adj
    params = sim.params
    println("Starting")
    updateGraph1(sim, params, g, gstate, gadj, ghtype, rng, g_iterator)
end

function updateGraph1(sim::IsingSim, params, g, gstate, gadj, ghtype, rng, g_iterator)
        
    ti = time()
    for _ in 1:10^8
        updateMonteCarloIsingD1(params, g, gstate, gadj, rng, g_iterator, ghtype)
        params.updates += 1
    end
    tf = time()

    println(tf - ti)

end

function updateMonteCarloIsingD1(params, g, gstate, gadj, rng, g_iterator, ghtype)
    beta = 1/(params.Temp)
    
    idx = rand(rng, g_iterator)
    
    Estate = @inbounds gstate[idx]*getEFactor(g, gstate, gadj, idx, ghtype)

    minEdiff = 2*Estate

    if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -1
    end
    
end

function updateSim2(sim::IsingSim, rng)
    g = sim.layers[1]
    ghtype = htype(g)
    g_iterator = ising_it(g,ghtype)
    gstate = g.state
    gadj = g.adj
    params = sim.params
    obs = sim.obs

    updateGraph2(sim, obs, params, g, gstate, gadj, ghtype, rng, g_iterator)
end

function updateGraph2(sim::IsingSim, obs, params, g, gstate, gadj, ghtype, rng, g_iterator)
    
    ti = time()
    for _ in 1:10^8
        updateMonteCarloIsingD2(obs, g, gstate, gadj, rng, g_iterator, ghtype)
        params.updates += 1
    end
    tf = time()

    println(tf - ti)

end

function updateMonteCarloIsingD2(obs, g, gstate, gadj, rng, g_iterator, ghtype)

    beta = 1/(obs.Temp[])
    
    idx = rand(rng, g_iterator)
    
    Estate = @inbounds gstate[idx]*getEFactor(g, gstate, gadj, idx, ghtype)

    minEdiff = 2*Estate

    if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -1
    end
    
end