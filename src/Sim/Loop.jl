"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and s_shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets s_shouldRun to false
Then it waits until isRunning is set to false after which s_shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export mainLoop
"""

function createProcess(g::IsingGraph, process = nothing, looptype = mainLoop; run = true, threaded = true, kwargs...)
    _sim = sim(g)
    if isnothing(process)
        # process = get_free_process(processes(_sim))
        process = Process()
        push!(processes(_sim), (process, 1))
    end
    
    if threaded
        task = process -> errormonitor(Threads.@spawn looptype(g, process; kwargs...))
        runtask(process, task, g; run)
    else
        looptype(g, process; kwargs...)
    end
    return
end

createProcesses(g::IsingGraph, num; kwargs...) = 
    for _ in 1:num; createProcess(g; kwargs...) end

export createProcess, createProcesses

function mainLoop(g::IsingGraph,
        process = get_free_process(processes(sim(g)));
        algorithm = g.default_algorithm,
        kwargs...)

    @assert !isnothing(process) "No free processes available"

    process.algorithm = algorithm
    
    algo_args = prepare(algorithm, g; kwargs...)
   
    masked_args = choose_args(process, algo_args; kwargs...)

    return _mainLoop(process, algorithm, masked_args; kwargs...)
end

using InteractiveUtils
export mainLoop
# g, gstate, gadj, iterator, rng, updateFunc, dEFunc, gstype::ST
function _mainLoop(process, @specialize(algorithm), @specialize(algo_args); kwargs...)
    println("Running on thread: ", Threads.threadid())
    while run(process)
        @inline algorithm(algo_args)
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

    while process.loopidx < iterations + 1
        @inline algorithm(args)
        inc(process)
        GC.safepoint()
    end

    return kwargs
end

function createProcessNew(g::IsingGraph, process = nothing, looptype = mainLoop; run = true, threaded = true, kwargs...)
    _sim = sim(g)
    if isnothing(process)
        process = Process()
        push!(processes(_sim), (process, 1))
    end

    # algo_args = prepare(g.default_algorithm, g; g)


    ct(process, g.default_algorithm; prepare, g)
    
    # mainLoopNew(g, process; kwargs...)

    return
end

function ct(process, @specialize(func); prepare = (a,b) -> (), @specialize(kwargs...))
    println("CT")
    @atomic process.run = true
    args = (;proc = process, kwargs...)
    algo_args = prepareNEW(args, kwargs)
    # masked_args = choose_argsNEW(process, algo_args; kwargs...)

    createtaskNEW(process, func, algo_args, process.taskfunc.runtime)
end


function createtaskNEW(p::Process, @specialize(func), @specialize(args), runtime::RT) where RT <: Runtime
    # task = nothing
    
    println("HERE")
    # algo_args = prepare(func, g; kwargs...)

    # masked_args = choose_argsNEW(p, algo_args; kwargs...)
        
    # p.task = @task indefiniteloop(p, func, masked_args)
    p.task = @task loopchoice(p, func, args, p.taskfunc.runtime)
    p.task.sticky = false
    Threads._spawn_set_thrpool(p.task, :default)
    schedule(p.task)

    return kwargs
end

function loopchoice(@specialize(p), @specialize(func), @specialize(args), runtime)
    println("Choosing loop")
    if runtime isa Indefinite
        return indefiniteloop(p, func, args)
    else
        return repeatloop(p, func, args, runtime)
    end
end

function indefiniteloop(@specialize(p), @specialize(func), @specialize(args))
    println("In indefiniteloop")
    println("Running on thread $(Threads.threadid())")
    while run(p) 
        @inline func(args)
        inc(p) 
        GC.safepoint() 
    end
end
export createProcessNew, mainLoopNew
