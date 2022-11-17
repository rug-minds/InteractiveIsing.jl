# module Benchmarking
include("oldbenchmarkfuncs.jl")
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

    g = sim.g

    pauseSim(sim)
    g.state = initRandomIsing!(g)
    bfunc(sim,iterations, rng)
end

# export benchmarkFuncGenerated
function benchmarkFuncGenerated(sim, iterations, rng)
    g = sim.g
    TIs = sim.TIs
    htype = g.htype

    g_iterator = ising_it(g, htype)
    
    getFac(g,idx) = getEFactor(g,idx, htype)


    # Defining argumentless functions here seems faster.
    
    # Offset large function into files for clearity
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingD
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingC

    isingUpdate = typeof(g) == IsingGraph{Int8} ? updateMonteCarloIsingD : updateMonteCarloIsingC
   
    sim.updates = 0

    t_start = time()

    for _ in 1:(iterations)
        isingUpdate()
        # sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end

# export benchmarkFuncGenerated2
function benchmarkFuncGenerated2(sim, iterations, rng)
    g = sim.g
    TIs = sim.TIs
    htype = g.htype

    g_iterator = ising_it(g, htype)

    # Defining argumentless functions here seems faster.
    
    # Offset large function into files for clearity
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingD2
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingC

    isingUpdate = typeof(g) == IsingGraph{Int8} ? updateMonteCarloIsingD : updateMonteCarloIsingC
   
    sim.updates = 0

    t_start = time()

    for _ in 1:(iterations)
        isingUpdate()
        sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end


function benchmarkFuncGenerated3(sim, iterations, rng)
    g = sim.g
    TIs = sim.TIs
    htype = g.htype

    g_iterator = ising_it(g, htype)
    
    getE(g,idx) = getEFactor(g,idx, htype)


    # Defining argumentless functions here seems faster.
    
    # Offset large function into files for clearity
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingD3
    @includetextfile ".." src textfiles MonteCarlo updateMonteCarloIsingC3

    isingUpdate = typeof(g) == IsingGraph{Int8} ? updateMonteCarloIsingD : updateMonteCarloIsingC
   
    sim.updates = 0

    t_start = time()

    for _ in 1:(iterations)
        isingUpdate()
        # sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end

# end