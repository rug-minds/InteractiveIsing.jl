
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets shouldRun to false
Then it waits until isRunning is set to false after which shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export updateGraph
"""
function updateGraph(sim::IsingSim)
    g = sim.g
    TIs = sim.TIs
    htype = g.htype
    
    rng = MersenneTwister()
    g_iterator = ising_it(g,htype)
    shouldRun = sim.shouldRun
    
    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    @includetextfile MonteCarlo updateMonteCarloIsingD
    @includetextfile MonteCarlo updateMonteCarloIsingC

    isingUpdate = typeof(g) == IsingGraph{Int8} ? 
            updateMonteCarloIsingD : updateMonteCarloIsingC

    sim.isRunning = true

    while shouldRun[]
        isingUpdate()
        sim.updates += 1
        
        GC.safepoint()
    end

    sim.isRunning = false
    while !shouldRun[]
        yield()
    end
    updateGraph(sim)

end
export updateGraph

