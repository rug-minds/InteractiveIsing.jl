
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
    gadj = adj(g)::Vector{Vector{Tuple{Int32,Float32}}}
    params = sim.params
    lTemp = Temp(sim)
    iterator = ising_it(g,gstype)

    try
        mainLoop(process, params, gidx, g, gstate, gadj, lTemp, iterator, rng, updateFunc, energyFunc, gstype)
    catch 
        status(process, :Terminated)
        message(process, :Nothing)
        rethrow()
    end
    
end

export updateGraph

function mainLoop(process, params, gidx, g, gstate, gadj::AT, lTemp, iterator, rng, updateFunc, energyFunc, gstype::ST)::Nothing where {AT <: Vector{Vector{Tuple{Int32,Float32}}}, ST <: SType}

    status(process, :Running)

    while run(process)
        updateFunc(g, params, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
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

    updateGraph(sim(g), process; gidx, energyFunc)
    return 
end
export mainLoop

@inline function updateMonteCarloIsing(g::IsingGraph, params, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}
    idx = rand(rng, iterator)
    updateMonteCarloIsing(idx, g, lTemp, gstate, rng, gstype, energyFunc)
    params.updates += 1
end

# @inline @generated function updateMonteCarloIsing(idx::Integer, g::IsingGraph, lTemp, gstate, iterator, rng, gstype::ST, energyFunc, statetype::MixedState) where {ST <: SType}

#     beta::Float32 = 1f0/(lTemp[])
    
#     Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, adj(g), idx, gstype)

#     minEdiff::Float32 = 2*Estate

#     if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
#         @inbounds g.state[idx] *= -Int8(1)
#     end
# end



@inline function updateMonteCarloIsing(idx::Integer, g, lTemp, gstate::Vector{Int8}, rng, gstype::ST, energyFunc) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, adj(g), idx, gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end

@inline function updateMonteCarloIsing(idx::Int32, g, lTemp, gstate::Vector{Float32}, rng, gstype::ST, energyFunc) where {ST <: SType}

    @inline function sampleCState()
        2f0*(rand(rng, Float32)- .5f0)
    end

    beta = 1f0/(lTemp[])
     
    oldstate = @inbounds gstate[idx]

    efactor = energyFunc(g, gstate, adj(g), idx, gstype)

    newstate = sampleCState()

    # ediff = efactor*(newstate-oldstate)

    ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
    if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
        @inbounds g.state[idx] = newstate 
    end

end

export updateMonteCarloIsing

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


"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcessList(sim::IsingSim, ; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister(), threaded = true)::Nothing
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
        errormonitor(Threads.@spawn updateGraphList($sim, process; gidx, updateFunc, energyFunc, rng))
    else
        updateGraphList(sim, process; gidx, updateFunc, energyFunc, rng)
    end

    return
end
export createProcessList

createProcessList(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())::Nothing = 
    for _ in 1:num; createProcessList(sim; gidx, updateFunc, energyFunc) end
export createProcess

function updateGraphList(sim::IsingSim, process = processes(sim)[1]; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())
    g = gs(sim)[gidx]
    gstype = stype(g)
    gstate = state(g)
    gadj = adjlist(g)
    params = sim.params
    loopTemp = Temp(sim)
    iterator = ising_it(g,gstype)

    updateFunc = updateMonteCarloIsingList
    energyFunc = getEFactor

    try
        mainLoopList(process, params, gidx, g, gstate, gadj, loopTemp, iterator, rng, gstype, updateFunc, energyFunc)
    catch 
        status(process, :Terminated)
        message(process, :Nothing)
        rethrow()
    end

    return
end

export updateGraphList

function mainLoopList(process, params, gidx, g, gstate, gadj::AL, lTemp, iterator, rng, gstype, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)::Nothing where {AL <: AdjList{Connections}}

    status(process, :Running)

    while message(process) == :Nothing
        # updateFunc(g, params, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)

        idx = rand(rng, iterator)
        connections = gadj[idx]

        @inline function sampleCState()
            2f0*(rand(rng, Float32)- .5f0)
        end
    
        beta = 1f0/(lTemp[])
         
        oldstate = @inbounds gstate[idx]
        
        efactor = getEFactor(gstate, connections)
    
        newstate = sampleCState()
    
        ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
        if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
            @inbounds g.state[idx] = newstate 
        end

        params.updates += 1
        GC.safepoint()

    end

    status(process, :Terminated)
    message(process, :Nothing)
    return 
end
export mainLoopList

@inline function updateMonteCarloIsingList(g::IsingGraph, params, lTemp, gstate, gadj::AL, iterator, rng, gstype, energyFunc) where {AL <: AdjList{Connections}}
    idx = rand(rng, iterator)
    connections = gadj[idx]
    updateMonteCarloIsingList(idx, g, lTemp, gstate, connections, gstype, rng, energyFunc)
    params.updates += 1
end

@inline function updateMonteCarloIsingList(idx::Integer, g, lTemp, gstate::Vector{Int8}, connections::Connections, rng, gstype::ST, energyFunc) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(gstate, gadj[idx], gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end

@inline function updateMonteCarloIsingList(idx::Integer, g, lTemp, gstate::Vector{Float32}, connections::C, gstype, rng, energyFunc) where {C <: Connections}

    @inline function sampleCState()
        2f0*(rand(rng, Float32)- .5f0)
    end

    beta = 1f0/(lTemp[])
     
    oldstate = @inbounds gstate[idx]
    
    efactor = energyFunc(gstate, connections)

    newstate = sampleCState()

    # ediff = efactor*(newstate-oldstate)

    ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
    if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
        @inbounds g.state[idx] = newstate 
    end

end

@inline function getEFactor(gstate, connections::C) where C <: Connections
    efactor = 0f0
    weights = connections.weights
    idxs = connections.idxs
    @inbounds @simd for conn_idx in eachindex(connections)
    # @turbo for conn_idx in eachindex(weights)
        idx = idxs[conn_idx]
        weight = weights[conn_idx]
        efactor += -weight*gstate[idx]
    end
    return efactor
end


export updateMonteCarloIsingList
using LoopVectorization
# @inline function getEFactorList(gstate, connections, gstype)
#     efactor = 0f0
#     weights = connections.weights
#     idxs = connections.idxs
#     @inbounds @simd for conn_idx in eachindex(connections)
#         idx = idxs[conn_idx]
#         weight = weights[conn_idx]
#         efactor += -weight*gstate[idx]
#     end
#     return efactor
# end

@inline function getEFactorList(gstate, connections, gstype)::Float32
    efactor = 0f0
    weights = connections.weights
    idxs = connections.idxs
    # @inbounds @simd for conn_idx in eachindex(connections)
    @turbo for conn_idx in eachindex(weights)
        idx = idxs[conn_idx]
        weight = weights[conn_idx]
        efactor += -weight*gstate[idx]
    end
    return efactor
end

@inline function getEFactor(gstate, connections, gstype)
    efactor = 0f0
    weights = connections.weights
    idxs = connections.idxs
    @inbounds @simd for conn_idx in eachindex(connections)
    # @turbo for conn_idx in eachindex(weights)
        idx = idxs[conn_idx]
        weight = weights[conn_idx]
        efactor += -weight*gstate[idx]
    end
    return efactor
end


# @inline function getEFactorList(g, gstate, gadj, idx, gstype)
#     connections = gadj[idx]
#     efactor = vreduce(+, -connections.weights.*(@view gstate[connections.idxs]))
#     # @inbounds @simd for conn_idx in eachindex(connections)
#     return efactor
# end
function getfac(state,tuple)
    return 
end

# Make a view of the Int32's of a vector of Tuple{Int32,Float32}
function idxs(v::Vector{Tuple{Int32,Float32}})
    return first.(@view v[1:end])
end
export idxs

@inline function getidxweight(connections::Vector{Tuple{Int32,Float32}}, idx)::Tuple{Int32,Float32}
    return first(connections[idx]), last(connections[idx])
end

# TODO: Rewrite getefactor like this
@inline function getEFactor(gstate, connections::Vector{Tuple{Int32,Float32}}, gstype)
    efactor = 0f0
    @inline @simd for conn_idx in eachindex(connections)
        idx, weight = getidxweight(connections, conn_idx)
        efactor += -weight*gstate[idx]
    end
    return efactor
end

export getEFactorTurbo

export getEFactorList

function createProcessNew(sim::IsingSim, ; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister(), threaded = true)::Nothing
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
        errormonitor(Threads.@spawn updateGraphNew(sim, process; gidx, updateFunc, energyFunc, rng))
    else
        updateGraphNew(sim, process; gidx, updateFunc, energyFunc, rng)
    end

    return
end
export createProcessNew

createProcessNew(sim::IsingSim, num; gidx = 1, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor, rng = MersenneTwister())::Nothing = 
    for _ in 1:num; createProcessNew(sim; gidx, updateFunc, energyFunc) end
export createProcess

function updateGraphNew(sim::IsingSim, process = processes(sim)[1]; gidx = 1, updateFunc = nothing, energyFunc = getEFactor, rng = MersenneTwister())
    g = gs(sim)[gidx]
    gstype = stype(g)
    gstate = state(g)
    gadj = adj(g)
    params = sim.params
    loopTemp = Temp(sim)
    iterator = ising_it(g,gstype)

    updateFunc = updateMonteCarloIsingNew

    try
        mainLoopNew(process, params, gidx, g, gstate, gadj, loopTemp, iterator, rng, updateFunc, energyFunc; gstype)
    catch 
        status(process, :Terminated)
        message(process, :Nothing)
        rethrow()
    end
    
end

@Base.propagate_inbounds function mainLoopNew(process, params, gidx, g, gstate, gadj::Vector{Vector{Conn}}, lTemp, iterator, rng, updateFunc = updateMonteCarloIsing, energyFunc = getEFactor; gstype::ST = stype(g))::Nothing where {ST}

    status(process, :Running)

    while message(process) == :Nothing
        updateFunc(g, params, lTemp, gstate, gadj, iterator, rng, gstype, energyFunc)
        # GC.safepoint()
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

@inline function updateMonteCarloIsingNew(g::IsingGraph, params, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc) where {ST <: SType}
    idx = rand(rng, iterator)
    updateMonteCarloIsingNew(idx, g, lTemp, gstate, gadj, rng, gstype, energyFunc)
    # updates(sim(g), updates(sim(g)) + 1)
    # updates(params, updates(params) + 1)
    params.updates += 1
end

@inline @generated function updateMonteCarloIsingNew(idx::Integer, g::IsingGraph, lTemp, gstate, gadj, iterator, rng, gstype::ST, energyFunc, statetype::MixedState) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(gstate, gadj[idx], gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end



@inline function updateMonteCarloIsingNew(idx::Integer, g, lTemp, gstate::Vector{Int8}, gadj, rng, gstype::ST, energyFunc) where {ST <: SType}

    beta::Float32 = 1f0/(lTemp[])
    
    Estate::Float32 = @inbounds gstate[idx]*energyFunc(gstate, gadj[idx], gstype)

    minEdiff::Float32 = 2*Estate

    if (Estate >= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds g.state[idx] *= -Int8(1)
    end
end

@inline function updateMonteCarloIsingNew(idx::Integer, g, lTemp, gstate::Vector{Float32}, gadj, rng, gstype::ST, energyFunc) where {ST <: SType}

    @inline function sampleCState()
        2f0*(rand(rng, Float32)- .5f0)
    end

    beta = 1f0/(lTemp[])
     
    oldstate = gstate[idx]

    efactor = energyFunc(gstate, gadj[idx], gstype)

    newstate = sampleCState()

    # ediff = efactor*(newstate-oldstate)

    ediff = Ediff(g, gstype, idx, efactor, oldstate, newstate)
    if (ediff < 0f0 || rand(rng, Float32) < exp(-beta*ediff))
        g.state[idx] = newstate 
    end

end