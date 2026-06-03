# Move the locks to the graph functio

function quit(g::AbstractSpinGraph)
    @warn "quit is deprecated, use close instead"
    close(g)
end
function Processes.close(g::AbstractSpinGraph)
    for process in reverse(processes(g))
        try 
            close(process)
        catch
            if isdone(process)
                deleteat!(processes(g), findfirst(isequal(process), processes(g)))
            end
        end
    end
    deleteat!(processes(g), 1:length(processes(g)))
end

function Processes.run(g::AbstractSpinGraph)
    run.(processes(g))
end

function Processes.pause(g::AbstractSpinGraph)
    pause.(processes(g))
end

unpause(g::AbstractSpinGraph) = run.(processes(g))

# For broadcasting
Processes.pause(::Nothing) = nothing
Processes.quit(::Nothing) = nothing
unpause(::Nothing) = nothing


export quitSim, quit, pause, unpause

# TODO: Make this work with the new process system KWARGS
function Processes.restart(g::AbstractSpinGraph; kwargs...)
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
function Processes.reinit(g::AbstractSpinGraph; kwargs...)
    _processes = processes(g)
    for process in _processes
        if isrunning(process)
            reinit(process)
        end
    end
    return
end

export reinit

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

function togglepause(g::AbstractSpinGraph)
    if any(isrunning.(processes(g)))
        pause.(processes(g))
    else
        for process in processes(g)
            if Processes.ispaused(process)
                run(process)
            end
        end
    end    
end


export togglepause
