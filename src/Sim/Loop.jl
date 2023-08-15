
"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(sim::IsingSim; gidx = 1, threaded = true, kwargs...)::Nothing
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
        errormonitor(Threads.@spawn updateGraph(sim, process; gidx, kwargs...))
    else
        updateGraph(sim, process; gidx, updateFunc, energyFunc, rng, kwargs...)
    end

    return
end

# createProcess(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, gadj = rng = MersenneTwister(), kwargs...)::Nothing = 
#     for _ in 1:num; createProcess(sim; gidx, updateFunc, energyFunc, kwargs...) end

export createProcess

function updateGraph(sim::IsingSim, process = processes(sim)[1]; gidx = 1, kwargs...)
    g = gs(sim)[gidx]
    gstype = stype(g)
    gstate = state(g)
    params = sim.params
    lTemp = Temp(sim)
    iterator = ising_it(g,gstype)
    gadj = haskey(kwargs, :gadj) ? kwargs[:gadj] : adj(g)
    updateFunc = haskey(kwargs, :updateFunc) ? kwargs[:updateFunc] : updateMonteCarloIsing
    energyFunc = haskey(kwargs, :energyFunc) ? kwargs[:energyFunc] : getEFactor
    rng = haskey(kwargs, :rng) ? kwargs[:rng] : MersenneTwister()

    try
        mainLoop(process, gidx, g, gstate, gadj, lTemp, iterator, rng, updateFunc, energyFunc, gstype; kwargs...)
    catch 
        status(process, :Terminated)
        atomic_message(process, :Nothing)
        rethrow()
    end
    
end

export updateGraph

function mainLoop(process, gidx, g, gstate, gadj, lTemp, iterator, rng, updateFunc, energyFunc, gstype::ST; kwargs...)::Nothing where {ST <: SType}

    status(process, :Running)

    while run(process)
        updateFunc(g, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
        inc(process)
        GC.safepoint()
    end
    
    status(process, :Paused)


    # Atomic set it back to run
    if atomic_message(process) == :Quit
        signal!(process, true, :Nothing)
        status(process, :Terminated)
        return
    end

    if message(process) == :Pause
        status(process, :Paused)
        while message(process) == :Pause
            yield()
            sleep(0.1)
            GC.safepoint()
        end
    end

    # Consume message and mark process to run
    signal!(process, true, :Nothing)

    updateGraph(sim(g), process; gidx, energyFunc, kwargs...)
    return 
end
export mainLoop

@inline function updateMonteCarloIsing(g::IsingGraph, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}
    idx = rand(rng, iterator)
    updateMonteCarloIsing(idx, g, lTemp, gstate, gadj, rng, gstype, energyFunc)
end


@inline function updateMonteCarloIsing(idx::Integer, g, lTemp, gstate::Vector{Int8}, gadj, rng, gstype::ST, energyFunc) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end


@inline function updateMonteCarloIsing(idx::Integer, g, lTemp, gstate::Vector{Float32}, gadj, rng, gstype::ST, energyFunc) where {ST <: SType}

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

# @inline function updateMonteCarloIsingSparse(g, lTemp, gstate, sp_adj::SparseMatrixCSC, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}

#     @inline function sampleCState()
#         2f0*(rand(rng, Float32)- .5f0)
#     end

#     idx = rand(iterator)

#     beta = 1f0/(lTemp[])
     
#     oldstate = @inbounds gstate[idx]

#     efactor = energyFunc(g, gstate, sp_adj, idx, gstype)

#     newstate = sampleCState()

#     # ediff = efactor*(newstate-oldstate)

#     ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
#     if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
#         @inbounds g.state[idx] = newstate 
#     end

# end

# export updateMonteCarloIsingSparse

# updateFunc(g, params, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
let times = Ref([])
    global function upDebug(g, params, lTemp, gstate::Vector, gadj, iterator, rng, gstype, energyFunc)

        beta = 1/(lTemp[])
        
        idx = rand(rng, iterator)
        
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

@inline function getEFactor(g, state, sparse::SparseMatrixCSC, idx, stype::SType)
    efac = 0f0 
    # @inbounds @fastmath @simd for idx in nzrange(sparse, idx)
    @turbo check_empty = true for idx in nzrange(sparse, idx)
    # for idx in nzrange(sparse, idx)
        efac += -state[sparse.rowval[idx]] * sparse.nzval[idx]
    end
    return efac
end

