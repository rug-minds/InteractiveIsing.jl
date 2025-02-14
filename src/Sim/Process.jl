function createProcess(g::IsingGraph, func = nothing; run = true, threaded = true, lifetime = nothing, args...)
    if isnothing(func)
        func = get(args, :algorithm, g.default_algorithm)
    end
    process = Process(func; lifetime, g, args...)
    
    ps = processes(g)
    push!(ps, process)
    start(process)

    return
end

export createProcess, createProcesses

hasprocess(g::IsingGraph) = !isempty(processes(g))
doneprocesses(g::IsingGraph) = findall(p -> isdone(p), processes(g))
norunningprocesses(g::IsingGraph) = length(doneprocesses(g)) == length(processes(g))
lastprocess(g::IsingGraph) = processes(g)[end]
cleanprocesses(g::IsingGraph) = deleteat!(processes(g), doneprocesses(g))

"""
Wait until all processes are done.
"""
Base.wait(g::IsingGraph) = wait.(processes(g))
"""
Fetch last output
"""
Base.fetch(g::IsingGraph) = fetch(process(g))
export wait, fetch