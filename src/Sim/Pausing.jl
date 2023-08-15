# Pauses sim and waits until paused
# TODO: Might crash when refresh sim is run too many times
function pauseSim(sim; block = false, ignore_lock = false, print = true)
    if print
        println("Pausing sim")
    end

    running = status(sim) .== :Running
    idxs = [1:length(processes(sim));][running]
    for process in (@view processes(sim)[idxs])
        signal!(process, false, :Pause; ignore_lock)
    end

    if block
        # Wait for all to terminate
        while any(x -> x == :Running, status(sim))
            yield()
        end
    end

    isPaused(sim)[] = true

    return
end
export pauseSim

function unpauseSim(sim; block = false, ignore_lock = false, print = true)
    if print
        println("Unpausing sim")
    end
    # Find running processes
    running = status(sim) .== :Paused
    idxs = [1:length(processes(sim));][running]
    for process in (@view processes(sim)[idxs])
        signal!(process, true, :Nothing; ignore_lock)
    end

    if block
        # Wait for all to terminate
        while any(x -> x == :Paused, status(sim))
            # sleep(.5)
            yield()
        end
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
function quitSim(sim)
    # Find running processes
    running = status(sim) .== :Running
    idxs = [1:length(processes(sim));][running]
    for process in (@view processes(sim)[idxs])
        signal!(process, false, :Quit)
    end

    # Wait for all to terminate
    while any(x -> x == :Running, status(sim))
        # sleep(.5)
        yield()
    end

    return
end
export quitSim

function refreshSim(sim)
    # Find running processes
    running = status(sim) .== :Running
    idxs = [1:length(processes(sim));][running]
    for process in (@view processes(sim)[idxs])
        run!(process, false)
    end
    return
end
export refreshSim

"""
Restart Sim with same number of processes
"""
function restartSim(sim, gidx = 1; kwargs...)
    processNum = count(stat -> stat == :Running, status(sim)) 
    processNum = processNum < 1 ? 1 : processNum
    quitSim(sim)
    # createProcess(sim, processNum; gidx, updateFunc, energyFunc)
    createProcess(sim; gidx, kwargs...)
    return
end 
export restartSim

function togglePauseSim(sim)
    # If any are paused, unpause else pause
    if any(x -> x == :Paused, status(sim))
        unpauseSim(sim)
    else
        pauseSim(sim)
    end
end
export togglePauseSim

function resetStatus!(sim)
    status(sim) .= :Terminated
    messages(sim) .= :Nothing
end

export resetStatus!