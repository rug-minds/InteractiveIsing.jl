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
Pause sim and lock pausing untill it's unlocked
This garantuees that the sim cannot be unpaused by a user
"""
function lockPause(sim; block = true)
    lock(processes(sim))
    close.(values(timers(sim)))
    pauseSim(sim, ignore_lock = true, print = false; block)
end
export lockPause
"""
Unlock and unpause the sim
"""
function unlockPause(sim; block = true)
    unpauseSim(sim, ignore_lock = true, print = false; block)
    start.(values(timers(sim)))
    unlock(processes(sim))
end
export unlockPause

"""
Quit sim and block until all processes are terminated
"""
function quit(sim::IsingSim)
    quit.(processes(sim))
end

function pause(sim::IsingSim)
    pause.(processes(sim))
end

function quit(g) 
    quit.(processes(g))
end

function pause(g)
    pause.(processes(g))
end

unpause(g::IsingGraph) = restart(g)

export quitSim, quit, pause, unpause

"""
Restart Sim with same number of processes
"""
function restart(g; kwargs...)
    _processes = processes(g)
    for process in _processes
        # Do this all at once, not in the loop
        # TODO: Fix this behavior
        _isrunning = ispaused(process) || isrunning(process)
        pause(process)
        # Get args from return value of process
        oldkwargs = retval(process)
        if _isrunning
            # Use the old kwargs
            newkwargs = mergekwargs(oldkwargs, kwargs)
            task = process -> errormonitor(Threads.@spawn mainLoop(g, process, oldkwargs; newkwargs...))
            runtask(process, task, g)
        end
    end
    return
end 
export restart

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

