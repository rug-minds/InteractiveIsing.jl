
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(sim::IsingSim, ; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)::Nothing
    process = nothing
    for (idx, p) in enumerate(processes(sim))
        if status(p) == :Terminated
            process = p
            status(p, :Starting)
            break
        end

        if idx == length(processes(sim))
            println("No available process")
            return
        end
    end

    Threads.@spawn updateGraph(sim, process; gidx, updateFunc, energyFunc)
    return
end
createProcess(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)::Nothing = 
    for _ in 1:num; createProcess(sim; gidx, updateFunc, energyFunc) end
export createProcess

function updateGraph(sim::IsingSim, process; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)
    g = gs(sim)[gidx]
    ghtype = htype(g)
    rng = MersenneTwister()
    g_iterator = ising_it(g,ghtype)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)

    mainLoop(sim, process, gidx, params, loopTemp, g, gstate, gadj, ghtype, rng, g_iterator, updateFunc, energyFunc)
end

export updateGraph

function mainLoop(sim::IsingSim, process, gidx, params::IsingParams, 
    lTemp, g, gstate, gadj, ghtype, rng, g_iterator, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)::Nothing

    status(process, :Running)

    while message(process) == :Nothing
        updateFunc(lTemp, g, gstate, gadj, rng, g_iterator, ghtype, energyFunc)
        params.updates += 1
        GC.safepoint()
    end

    status(process, :Paused)

    while message(process) != :Nothing
        
        if message(process) == :Execute
            func(process)(sim, gidx, energyFunc)
        elseif message(process) == :Quit
            status(process, :Terminated)
            message(process, :Nothing)
            return
        else
            sleep(0.5)
        end
            
        yield()
        GC.safepoint()
    end

    updateGraph(sim, process; gidx, energyFunc)
    return 
end

function updateMonteCarloIsing(lTemp, g, gstate::Vector{Int32}, gadj, rng, g_iterator, ghtype, energyFunc)

    beta = 1/(lTemp[])
    
    idx = rand(rng, g_iterator)
    
    Estate = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)

    minEdiff = 2*Estate

    if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -1
    end
    
end

function updateMonteCarloIsing(lTemp, g, gstate::Vector{Float32}, gadj, rng, g_iterator, ghtype)
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

let times = Ref([])
    global function upDebug(lTemp, g, gstate::Vector{Int32}, gadj, rng, g_iterator, ghtype, energyFunc)

        beta = 1/(lTemp[])
        
        idx = rand(rng, g_iterator)
        
        ti = time()
        Estate = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)
        tf = time()

        push!(times[], tf-ti)
        if length(times[]) == 1000000
            println(sum(times[])/length(times[]))
            times[] = []
        end

        minEdiff = 2*Estate

        if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
    end
end
export upDebug