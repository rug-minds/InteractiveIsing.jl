# Pauses sim and waits until paused
function pauseSim(sim; print = true)
    if print
        println("Pausing sim")
    end

    # If no processes are running
    if !any(x -> x == :Running, status(sim))
        return true
    end

    # Set all running to paused
    messages(sim)[ [status == :Running for status in status(sim)] ] .= :Pause

    # While any are running, yield
    while any(x -> x == :Running, status(sim))
        # sleep(.5)
        yield()
    end

    return true
end
export pauseSim

function unpauseSim(sim; print = true)
    if print
        println("Unpausing sim")
    end
    
    # If none are paused return
    if !any(x -> x == :Paused, status(sim))
        return true
    end

    # Remove all messages
    messages(sim)[ [status == :Paused for status in status(sim)] ] .= :Nothing

    while any(x -> x == :Paused, status(sim))
        # sleep(.5)
        yield()
    end

    return true
end
export unpauseSim

function quitSim(sim)

    messages(sim) .= :Quit

    while !all(x -> x == :Terminated, status(sim))
        yield()
    end

    messages(sim) .= :Nothing

    return true
end
export quitSim

function refreshSim(sim)
    pauseSim(sim, print = false)
    unpauseSim(sim, print = false)
end
export refreshSim

"""
Restart the sim with a single process and possible a given energy factor function
Has some errors if the process terminated
"""
function restartSim(sim, gidx = 1; updateFunc = updateMonteCarloIsing, energyFunc = getEFactor)
    processNum = count(stat -> stat == :Running, status(sim)) 
    processNum = processNum < 1 ? 1 : processNum
    println(processNum)
    quitSim(sim)
    createProcess(sim, processNum; gidx, updateFunc, energyFunc)
    println("Restarted Sim")
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