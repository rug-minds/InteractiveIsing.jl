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
# function Processes.quit(sim::IsingSim)
    # quit.(processes(sim))
    # deleteat!(processes(sim), 1:length(processes(sim)))
# end

# function Processes.pause(sim::IsingSim)
#     pause.(processes(sim))
# end

function Processes.quit(g::IsingGraph)
    quit.(processes(g))
    deleteat!(processes(g), 1:length(processes(g)))
end

function Processes.pause(g::IsingGraph)
    pause.(processes(g))
end

Processes.unpause(g::IsingGraph) = restart(g)

# For broadcasting
Processes.pause(::Nothing) = nothing
Processes.quit(::Nothing) = nothing
Processes.unpause(::Nothing) = nothing


export quitSim, quit, pause, unpause

# TODO: Make this work with the new process system KWARGS
function Processes.restart(g::IsingGraph; kwargs...)
    _processes = processes(g)
    for process in _processes
        # Is process being used? Otherwise nothing has to be started
        restart(process)
    end
end
export restart

"""
Keep the keywords and recompile the processes
"""
function Processes.refresh(g::IsingGraph; kwargs...)
    _processes = processes(g)
    for process in _processes
        if isrunning(process)
            refresh(process)
        end
    end
    return
end
export refresh

# """
# Reset the keywords to the standard values, and restart the processes

# """
# function Processes.reset(g::IsingGraph; kwargs...)
#     _processes = processes(g)
#     for process in _processes
#         # Is process being used? Otherwise nothing has to be started
#         _isused = isused(process)
#         quit(process)
#         if _isused
#             task = process -> errormonitor(Threads.@spawn mainLoop(g, process; kwargs...))
#             runtaskOLD(process, task, g)
#         end
#     end
#     return
# end

function togglePause(g::IsingGraph)
    if any(isrunning.(processes(g)))
        pause.(processes(g))
    else
        for process in processes(g)
            if Processes.ispaused(process)
                start(process)
            end
        end
    end    
end


export togglePause

