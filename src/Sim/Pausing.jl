# Move the locks to the graph function

# Pauses sim and waits until paused
function pauseSim(sim; block = false, ignore_lock = false, print = true)
    if print
        println("Pausing sim")
    end
    running = isrunning.(processes(sim))
    idxs = [1:length(processes(sim));][running]
    pause.((@view processes(sim)[idxs]))
    isPaused(sim)[] = true
    return
end

export pauseSim

function unpauseSim(sim; block = false, ignore_lock = false, print = true)
    if print
        println("Unpausing sim")
    end
    # Find running processes
    for g in gs(sim)
        unpause(g)
    end
    isPaused(sim)[] = false
    return
end
export unpauseSim

"""
Pause sim and lock pausing until it's unlocked
This garantuees that the sim cannot be unpaused by a user
"""
function lockPause(sim; block = true)
    lock(processes(sim))
    close.(values(timers(sim)))
    pauseSim(sim, ignore_lock = true, print = false; block)
end
lockPause(::Nothing; kwargs...) = nothing
export lockPause

"""
Unlock and unpause the sim
"""
function unlockPause(sim; block = true)
    unpauseSim(sim, ignore_lock = true, print = false; block)
    start.(values(timers(sim)))
    unlock(processes(sim))
end
unlockPause(::Nothing; kwargs...) = nothing
export unlockPause

"""
Quit sim and block until all processes are terminated
"""
function quit(sim::IsingSim)
    quit.(processes(sim))
    deleteat!(processes(sim), 1:length(processes(sim)))
end

function pause(sim::IsingSim)
    pause.(processes(sim))
end

function quit(g::IsingGraph)
    quit.(processes(g))
    gidx = get_gidx(g)
    deleteat!(processes(sim(g)), processes(sim(g)).graphidx .== gidx)
end

function pause(g::IsingGraph)
    pause.(processes(g))
end

unpause(g::IsingGraph) = restart(g)

# For broadcasting
pause(::Nothing) = nothing
quit(::Nothing) = nothing
unpause(::Nothing) = nothing


export quitSim, quit, pause, unpause

# Todo: make this skip prep if not needed.
"""
Restart processes of the graph with given kwargs
Starts up all processes that are paused
"""
function restart(g; kwargs...)
    _processes = processes(g)
    
    for process in _processes
        # Is process being used? Otherwise nothing has to be started
        _isused = isused(process)
        pause(process)
        if _isused
            task = process -> errormonitor(Threads.@spawn mainLoop(g, process; kwargs...))
            @atomic process.run = true
            runtaskOLD(process, task, g)
        end
    end
    return
end 
export restart

"""
Keep the keywords and recompile the processes
"""
function refresh(g; kwargs...)
    _processes = processes(g)
    for process in _processes
        wasrunning = isrunning(process)
        # Is process being used? Otherwise nothing has to be started
        _isused = isused(process)
        pause(process)
        if _isused
            task = process -> errormonitor(Threads.@spawn mainLoop(g, process; kwargs...))
            @atomic process.run = true
            runtaskOLD(process, task, g, run = wasrunning)
        end
    end
    return
end

"""
Reset the keywords to the standard values, and restart the processes

"""
function reset(g; kwargs...)
    _processes = processes(g)
    for process in _processes
        # Is process being used? Otherwise nothing has to be started
        _isused = isused(process)
        quit(process)
        if _isused
            task = process -> errormonitor(Threads.@spawn mainLoop(g, process; kwargs...))
            runtask(process, task, g)
        end
    end
    return
end

function togglePause(g)
    if any(isrunning.(processes(sim(g))))
        pause.(processes(sim(g)))
    else
        for process in processes(sim(g))
            if ispaused(process)
                args = fetch(process)
                task = process -> Threads.@spawn mainLoop(g, process; args...)
                errormonitor(runtask(process, task, g))
            end
        end
    end    
end


export togglePause

