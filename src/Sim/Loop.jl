"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(g::IsingGraph, process = nothing, looptype = mainLoop; threaded = true, kwargs...)
    _sim = sim(g)
    if isnothing(process)
        for p in processes(_sim)
            if isidle(p)
                process = p
                break
            end
        end
        if isnothing(process)
            println("No available processes")
            return
        end
    else # if specific process is chosen check if it is available
        if !isidle(process)
            println("Process is not idle")
            return
        end
    end
    
    if threaded
        task = process -> errormonitor(Threads.@spawn looptype(g, process; kwargs...))
        runtask(process, task, g)
    else
        looptype(g, process; kwargs...)
    end

    return
end

createProcesses(g::IsingGraph, num; kwargs...) = 
    for _ in 1:num; createProcess(g; kwargs...) end

export createProcess, createProcesses

function mainLoop(g::IsingGraph, process = processes(sim(g))[1], oldkwargs = pairs((;));
        algorithm = g.default_algorithm,
        kwargs...)

    args = prepare(algorithm, g; kwargs...)

    return mainLoop(process, algorithm, args; kwargs...)

end

export mainLoop
# g, gstate, gadj, iterator, rng, updateFunc, dEFunc, gstype::ST
function mainLoop(process, @specialize(algorithm::Function), @specialize(args); kwargs...)

    while run(process)
        algorithm(args)
        inc(process)
        GC.safepoint()
    end

    return kwargs
end
export mainLoop

function mainLoopIterated(g::IsingGraph, process = processes(sim(g))[1], oldkwargs = pairs((;));
        algorithm = g.default_algorithm,
        iterations = 10000,
        kwargs...)
    # Keywords reserved for the main loop itself
    reserved_keywords = (:algorithm, iterations)    

    args = prepare(algorithm, g; kwargs...)
    
    return mainLoopIterated(process, algorithm, args, iterations; kwargs...)

end

export mainLoop
# g, gstate, gadj, iterator, rng, updateFunc, dEFunc, gstype::ST
function mainLoopIterated(process, @specialize(algorithm::Function), @specialize(args), iterations; kwargs...)

    while process.updates < iterations
        algorithm(args)
        inc(process)
        GC.safepoint()
    end

    return kwargs
end


