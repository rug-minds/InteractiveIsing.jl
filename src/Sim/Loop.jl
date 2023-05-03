
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(sim::IsingSim, ; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())::Nothing
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

    errormonitor(Threads.@spawn updateGraph(sim, process; gidx, updateFunc, energyFunc, rng))
    return
end

createProcess(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())::Nothing = 
    for _ in 1:num; createProcess(sim; gidx, updateFunc, energyFunc) end
export createProcess

function updateGraph(sim::IsingSim, process = processes(sim)[1]; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())
    g = gs(sim)[gidx]
    ghtype = htype(g)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)

    try
        mainLoop(sim, process, params, gidx, g, gstate, gadj, loopTemp, rng, updateFunc, energyFunc)
    catch 
        status(process, :Terminated)
        message(process, :Nothing)
        rethrow()
    end
    
end

export updateGraph

function mainLoop(sim::IsingSim, process, params, gidx, g, gstate, gadj, lTemp, rng, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor; ghtype = htype(g))::Nothing

    status(process, :Running)

    while message(process) == :Nothing
        updateFunc(sim, g, params, lTemp, gstate, gadj, rng, ghtype, energyFunc)
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
            sleep(0.01)
        end
            
        yield()
        GC.safepoint()
    end

    updateGraph(sim, process; gidx, energyFunc)
    return 
end
export mainLoop

function updateMonteCarloIsing(sim, g, params, lTemp, gstate::Vector{Int8}, gadj, rng, ghtype, energyFunc)

    beta::Float32 = 1/(lTemp[])
    
    idx::Int32 = rand(rng, ising_it(g, ghtype))
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -1
    end
    
    params.updates +=1

end

function updateMonteCarloIsing(sim, g, params, lTemp, gstate::Vector{Float32}, gadj, rng, ghtype, energyFunc)
    # @inline function deltE(efac,newstate,oldstate)::Float32
    #     return efac*(newstate-oldstate)
    # end

    @inline function sampleCState()::Float32
        Float32(2*(rand(rng)-.5))
    end

    beta::Float32 = 1/(lTemp[])

    idx::Int32 = rand(rng, ising_it(g))
     
    oldstate::Float32 = @inbounds gstate[idx]

    efactor::Float32 = energyFunc(g, gstate, gadj, idx, ghtype)

    newstate::Float32 = sampleCState()
    
    # Ediff = deltE(efactor,newstate,oldstate)
    ediff::Float32 = Ediff(g, ghtype, idx, efactor, oldstate, newstate)
    if (ediff < 0 || rand(rng) < exp(-beta*ediff))
        @inbounds g.state[idx] = newstate 
    end

    params.updates +=1
end

export updateMonteCarloIsing

let times = Ref([])
    global function upDebug(lTemp, g, gstate::Vector, gadj, rng, g_iterator, ghtype, energyFunc, params)

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