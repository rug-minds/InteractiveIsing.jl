
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export updateGraph
"""
function updateSim(sim::IsingSim, gidx, energyFunc = getEFactor)::Nothing
    g = gs(sim)[gidx]
    ghtype = htype(g)
    rng = MersenneTwister()
    g_iterator = ising_it(g,ghtype)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)

    updateFunc = continuous(g) ? updateMonteCarloIsingC : updateMonteCarloIsingD

    updateGraph(sim, gidx, params, loopTemp, g, gstate, gadj, ghtype, rng, g_iterator, updateFunc, energyFunc)
end

export updateSim

function updateGraph(sim::IsingSim, gidx, params::IsingParams, lTemp, g, gstate, gadj, ghtype, rng, g_iterator, updateFunc, energyFunc)::Nothing
    
    isRunning(sim,true)
    
    while shouldRun(sim)
        updateFunc(lTemp, g, gstate, gadj, rng, g_iterator, ghtype, energyFunc)
        params.updates += 1
        GC.safepoint()
    end

    isRunning(sim,false)
    while !shouldRun(sim)
        yield()

        if shouldQuit(sim)
            return
        end
    end

    updateSim(sim, gidx, energyFunc)
    return 
end

function updateMonteCarloIsingD(lTemp, g, gstate, gadj, rng, g_iterator, ghtype, energyFunc)

    beta = 1/(lTemp[])
    
    idx = rand(rng, g_iterator)
    
    Estate = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)

    minEdiff = 2*Estate

    if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -1
    end
    
end

function updateMonteCarloIsingC(lTemp, g, gstate, gadj, rng, g_iterator, ghtype)
    # @inline function deltE(efac,newstate,oldstate)::Float32
    #     return efac*(newstate-oldstate)
    # end

    @inline function sampleCState()::Float32
        Float32(2*(rand(rng)-.5))
    end

    beta = 1/(lTemp[])

    idx = rand(rng, g_iterator)
     
    oldstate = @inbounds gstate[idx]

    efactor = energyFunc(g, gstate, gadj, idx, ghtype)

    newstate = sampleCState()
    
    # Ediff = deltE(efactor,newstate,oldstate)
    ediff = Ediff(g, ghtype, idx, efactor, oldstate, newstate)
    if (ediff < 0 || rand(rng) < exp(-beta*ediff))
        @inbounds g.state[idx] = newstate 
    end
end

