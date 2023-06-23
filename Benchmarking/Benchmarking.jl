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


function Benchmark(sim::IsingSim, process = processes(sim)[1], gidx = 1; g = gs(sim)[gidx], updateFunc, energyFunc)
    ghtype = htype(g)
    Random.seed!(1234)
    rng = MersenneTwister(1234)
    g_iterator = ising_it(g,ghtype)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)

    quitSim(sim)
    runTimedFunctions(sim)[] = false

    state(g) .= initRandomState(g)
    params.updates = 0

    println("Starting Benchmark")
    ti = time()
    mainLoop(sim, process, params, gidx, g, gstate, gadj, loopTemp, rng, g_iterator, updateFunc, energyFunc)
    tf = time()

    display(gToImg(g))

    return tf - ti
end

let iterations = 10^7
    global function monteCarloBenchmark(sim, g, params, lTemp, gstate, gadj, rng, g_iterator, ghtype, energyFunc)
        beta::Float32 = 1/(lTemp[])
        
        idx::Int32 = rand(rng, g_iterator)
        
        Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)

        minEdiff::Float32 = 2*Estate

        if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
        if params.updates > iterations
            messages(sim)[1] = :Quit;
        else
            params.updates += 1
        end

    end
end

function simtest(sim, time = 5)
    quitSim(sim)
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    g = graph(sim)
    state(g) .= initRandomState(g)
    createProcess(sim; rng)
    sleep(time)

    quitSim(sim)

end

function simtestList(sim, time = 5)
    quitSim(sim)
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    g = graph(sim)
    state(g) .= initRandomState(g)
    createProcessList(sim; rng)
    sleep(time)

    quitSim(sim)

end

function simtestNew(sim, time = 5)
    quitSim(sim)
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    g = graph(sim)
    state(g) .= initRandomState(g)
    # createProcessNew(sim; rng)
    updateGraphList(sim; rng)
    sleep(time)

    quitSim(sim)

end

function testfun(fstate, fadj, fadjlist, fstype, repeats = 10^8)
    # fg = graph(sim)
    # fstate = state(fg)
    # fadj = adj(fg)
    # fstype = stype(fg)
    # fadjlist = adjlist(fg)

    ti1 = time()
    testloop1(fstate, fadj, fstype)
    tf1 = time()
    println(tf1 - ti1)

    ti2 = time()
    testloop2(fstate, fadjlist, fstype, repeats)
    tf2 = time()
    println(tf2 - ti2)

    return

end

const repeats = 10^8

function testloop1(state, adj::Vector{Vector{Tuple{Int32,Float32}}} , stype)
    # for _ in 1:repeats
    @rtime getEFactor(state, adj[1], stype) repeats
    # end
end

function testloop2(state, adj::AdjList , stype, repeats)
    for _ in 1:repeats
        getEFactor(state, adj[1], stype)
    end
end