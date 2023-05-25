
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(sim::IsingSim, ; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister(), threaded = true)::Nothing
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
    if threaded 
        errormonitor(Threads.@spawn updateGraph(sim, process; gidx, updateFunc, energyFunc, rng))
    else
        updateGraph(sim, process; gidx, updateFunc, energyFunc, rng)
    end

    return
end

createProcess(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())::Nothing = 
    for _ in 1:num; createProcess(sim; gidx, updateFunc, energyFunc) end
export createProcess

function updateGraph(sim::IsingSim, process = processes(sim)[1]; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())
    g = gs(sim)[gidx]
    gstype = stype(g)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)
    iterator = ising_it(g,gstype)

    try
        mainLoop(process, params, gidx, g, gstate, gadj, loopTemp, iterator, rng, updateFunc, energyFunc; gstype)
    catch 
        status(process, :Terminated)
        message(process, :Nothing)
        rethrow()
    end
    
end

export updateGraph

function mainLoop(process, params, gidx, g, gstate, gadj::Vector{Vector{Conn}}, lTemp, iterator, rng, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor; gstype::ST = stype(g))::Nothing where {ST}

    status(process, :Running)

    while message(process) == :Nothing
        updateFunc(g, params, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
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
            sleep(0.1)
        end
            
        yield()
        GC.safepoint()
    end

    updateGraph(sim(g), process; gidx, energyFunc)
    return 
end
export mainLoop

@inline function updateMonteCarloIsing(g::IsingGraph, params, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}
    idx = rand(rng, iterator)
    updateMonteCarloIsing(idx, g, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
    params.updates +=1
end

@inline @generated function updateMonteCarloIsing(idx::Integer, g::IsingGraph, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc, statetype::MixedState) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end



@inline function updateMonteCarloIsing(idx::Integer, g, lTemp, gstate::Vector{Int8}, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end

@inline function updateMonteCarloIsing(idx::Integer, g, lTemp, gstate::Vector{Float32}, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}

    @inline function sampleCState()
        2f0*(rand(rng, Float32)- .5f0)
    end

    beta = 1f0/(lTemp[])
     
    oldstate = @inbounds gstate[idx]

    efactor = energyFunc(g, gstate, gadj, idx, gstype)

    newstate = sampleCState()

    # ediff = efactor*(newstate-oldstate)

    ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
    if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
        @inbounds g.state[idx] = newstate 
    end

end

export updateMonteCarloIsing

let times = Ref([])
    global function upDebug(lTemp, g, params, gstate::Vector, gadj, rng, g_iterator, gstype, energyFunc)

        beta = 1/(lTemp[])
        
        idx = rand(rng, g_iterator)
        
        ti = time()
        Estate = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, gstype)
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