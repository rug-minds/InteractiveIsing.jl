
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export updateGraph
"""
function updateGraph(sim::IsingSim, layer)
    g = sim.layers[layer]
    s_TIs = TIs(sim)
    htype = g.htype
    
    rng = MersenneTwister()
    g_iterator = ising_it(g,htype)
    s_shouldRun = shouldRun(sim)
    
    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    @includetextfile Sim Loop updateMonteCarloIsingD
    @includetextfile Sim Loop updateMonteCarloIsingC

    isingUpdate = typeof(g) == IsingGraph{Int8} ? 
            updateMonteCarloIsingD : updateMonteCarloIsingC

    isRunning(sim,true)

    while s_shouldRun[]
        isingUpdate()
        sim.updates += 1
        
        GC.safepoint()
    end

    isRunning(sim,false)
    while !s_shouldRun[]
        yield()
    end
    updateGraph(sim, layer)

end
export updateGraph

