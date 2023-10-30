# Pauses sim and waits until paused
# TODO: Might crash when refresh sim is run too many times
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
    wasrunning = ispaused.(processes(sim))
    idxs = [1:length(processes(sim));][wasrunning]
    for process in (@view processes(sim)[idxs])
        kwargs = retval(process)
        createProcess(g, process; kwargs...)
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
    pauseSim(sim, ignore_lock = true, print = false; block)
end
export lockPause
"""
Unlock and unpause the sim
"""
function unlockPause(sim; block = true)
    unpauseSim(sim, ignore_lock = true, print = false; block)
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
        paused_or_running = ispaused(process) || isrunning(process)
        pause(process)
        ###
        args = retval(process)
        if paused_or_running
            newkwargs = mergekwargs(args, kwargs)
            task = process -> errormonitor(Threads.@spawn updateGraph(g, process; newkwargs...))
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
                task = process -> Threads.@spawn updateGraph(g, process; args...)
                errormonitor(runtask(process, task, g))
            end
        end
    end    
end


export togglePause
