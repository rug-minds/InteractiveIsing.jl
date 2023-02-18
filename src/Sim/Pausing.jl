# Pauses sim and waits until paused
function pauseSim(sim)
    println("Pausing sim")

    # If no processes are running
    if !any(x -> x == :Running, status(sim))
        return true
    end

    # Set all running to paused
    messages(sim)[ [status == :Running for status in status(sim)] ] .= :Pause

    # While any are running, yield
    while any(x -> x == :Running, status(sim))
        yield()
    end

    return true
end
export pauseSim

function unpauseSim(sim)
    println("Unpausing sim")
    
    # If none are paused return
    if !any(x -> x == :Paused, status(sim))
        return true
    end

    # Remove all messages
    messages(sim)[ [status == :Paused for status in status(sim)] ] .= :Nothing

    while any(x -> x == :Paused, status(sim))
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

    return true
end
export quitSim

function refreshSim(sim)
    pauseSim(sim)
    unpauseSim(sim)
end
export refreshSim

function restartSim(sim, gidx, energyFunc = getEFactor)
    quitSim(sim)
    createProcess(sim; gidx, energyFunc)
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

