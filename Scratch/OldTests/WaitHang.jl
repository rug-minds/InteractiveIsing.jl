mutable struct Process
    task::Union{Nothing, Task}
    updates::Int
    @atomic run::Bool
end

function quit(p::Process)
    @atomic p.run = false
    if !isnothing(p.task)
        if istaskstarted(p.task)
            wait(p.task)
        end
    end
    p.updates = 0
    return p
end

function fib(n)
    if n <= 1 return 1 end
    return fib(n - 1) + fib(n - 2)
end

function threadedloop(p)
    n = 0
    while p.run
        n = fib(n)
        p.updates += 1
        GC.safepoint()
    end
    return n
end

Process() = Process(nothing, 0, true)

@inline function createtask(p, func)
    println("Running task")
    @atomic p.run = true
    println("Task assigned")
    p.task = Threads.@spawn func(p)
    println("Returning task")
    return p.task
end
processes = [Process() for i in 1:7]
println("Making processes")
begin 
    for i in 1:7
        println(i)
        createtask(processes[i], threadedloop)
    end
end
quit.(processes)
