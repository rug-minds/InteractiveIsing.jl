# Probably add a ref to the graph it's working on
import Base: Threads.SpinLock, lock, unlock
mutable struct Process
    task::Union{Nothing, Task}
    updates::Int   
    # To make sure other processes don't interfere
    lock::SpinLock 
    @atomic run::Bool
    objectref::Any
    retval::Any
    errorlog::Any
    algorithm::Any #Ref to the algorithm being run
end

# Process() = Process(nothing, 0, Threads.SpinLock(), (true, :Nothing))
Process() = Process(nothing, 0, Threads.SpinLock(), false, nothing, nothing, nothing, nothing)
@setterGetter Process lock run

function Base.show(io::IO, p::Process)
    statestring = ""
    if ispaused(p)
        statestring = "Paused"
    elseif isrunning(p)
        statestring = "Running"
    elseif isdone(p)
        statestring = "Finished"
    elseif neverstarted(p)
        statestring = "Unstarted"
    end

    print(io, "$statestring Process")

    return nothing
end
"""
Chooses the right combination of arguments to use for a process
    1. Algorithm prepares arguments
    2. If the process was previously paused, also use earlier specified kwargs
    3. If the user gives new args through kwargs, also use those
"""
function choose_args(p::Process, prepared_args, specified_args = nothing)
    masked_args = prepared_args
    if ispaused(p) # If paused, use the old kwargs
        #TODO: Is the if here neccesary?
        # Shouldn't it be decided by wether there is a retval
        masked_args = replacekwargs(prepared_args, p.retval)
    end
    if !isnothing(specified_args)
        masked_args = replacekwargs(prepared_args, specified_args)
    end

    return masked_args
end


status(p::Process) = isrunning(p) ? :Running : :Quit
message(p::Process) = run(p) ? :Run : :Quit

isstarted(p::Process) = !isnothing(p.task) && istaskstarted(p.task)

isrunning(p::Process) = !isnothing(p.task) && !istaskdone(p.task)

ispaused(p::Process) = !isnothing(p.task) && istaskdone(p.task)

isdone(p::Process) = isnothing(p.task)

isidle(p::Process) = isdone(p.task) || ispaused(p)

neverstarted(p::Process) = isnothing(p.task) && isnothing(p.retval)

"""
Can be used for a new process
"""
isfree(p::Process) = neverstarted(p) || isdone(p)
"""
Is currently used for running,
    can be paused
"""
isused(p::Process) = isrunning(p) || ispaused(p)



function quit(p::Process)
    @atomic p.run = false
    p.retval = fetch(p)
    p.updates = 0
    p.task = nothing
    return true
end

function pause(p::Process)
    @atomic p.run = false
    try 
        p.retval = fetch(p)
    catch e
        p.errorlog = e
    end
    return true
end
"""
Takes a runnable task and then runs it with the process
p is the process
"""
function runtask(p::Process, task::Task, objectref = nothing; run = true)
    p.objectref = objectref
    @atomic p.run = run
    task.sticky = false
    schedule(task)
    p.task = task
    p.retval = nothing
    return task
end

"""
Give a function and then creates a task that is run by the process
The function needs to take an object as a reference
"""
function runtask(p, taskf::Function, objectref = nothing; run = true)
    p.objectref = objectref
    @atomic p.run = run
    t = taskf(p)
    p.task = t
    p.retval = nothing
    return t
end



(p::Process)(taskf::Function, objectref = nothing) = runtask(p, taskf, objectref)
(p::Process)(task::Task, objectref = nothing) = runtask(p, task, objectref)

@inline lock(p::Process) = lock(p.lock)
@inline lock(f, p::Process) = lock(f, p.lock)
@inline unlock(p::Process) =  unlock(p.lock)

reset!(p::Process) = (p.updates = 0; p.retval = nothing; p.task = nothing; @atomic p.run = false; p.objectref = nothing)

run(p::Process) = p.run
run(p::Process, val) = @atomic p.run = val

@inline Base.wait(p::Process) = if !isnothing(p.task) wait(p.task) else nothing end
@inline Base.fetch(p::Process) = if !isnothing(p.task) fetch(p.task) else nothing end

@inline inc(p::Process) = p.updates += 1

"""
Access run atomically
"""
@inline function atomic_run(p::Process; ignore_lock = false)
    !ignore_lock && lock(p)
    # ret = (@atomic p.signal)[1]
    ret = @atomic p.run
    !ignore_lock && unlock(p)
    return ret
end

@inline function atomic_run!(p::Process, val; ignore_lock = false)
    !ignore_lock && lock(p)
    # ret = @atomic p.signal = (val, p.signal[2])
    ret = @atomic p.run = val
    !ignore_lock && unlock(p)
    return ret
end

@inline running(p::Process) = p.status == :Running

# Base.put!(p::Process, val) = put!(p.refresh, val)
# Base.take!(p::Process) = take!(p.refresh)
# Base.isready(p::Process) = isready(p.refresh)
# Base.isopen(p::Process) = isopen(p.refresh)
# Base.close(p::Process) = close(p.refresh)
# @inline Base.isempty(p::Process) = isempty(p.refresh)


mutable struct Processes <: AbstractVector{Process}
    procs::Vector{Process}
end

lock(p::Processes) = lock.(p.procs)
unlock(p::Processes) = unlock.(p.procs)

Base.size(p::Processes) = (length(p.procs),)
Processes(num::Integer) = Processes([Process() for _ in 1:num])

Base.getindex(processes::Processes, num) = processes.procs[num]
Base.setindex!(processes::Processes, val, idx) = setindex(processes.procs[num], val, idx)

Base.length(processes::Processes) = length(processes.procs)

Base.iterate(processes::Processes, s = 1) = iterate(processes.procs, s)

function get_free_process(procs::Union{Processes, Vector{Process}})
    for p in procs
        if isfree(p)
            return p
        end
    end
    return nothing
end

"""
For iterator
Still used?
"""
struct ProcessStats <: AbstractVector{Symbol}
    processes::Processes
    type::Symbol
end

Base.size(ps::ProcessStats) = (length(ps.processes),)

messages(procs::Processes) = ProcessStats(procs, :message)
messages(sim) = messages(sim.processes)
status(procs::Processes) = ProcessStats(procs, :status)
status(sim) = status(sim.processes)
export messages
export status

Base.iterate(ps::ProcessStats, state = 1) = state > length(ps.processes.procs) ? nothing : (getfield(ps.processes.procs[state], ps.type), state + 1)

function Base.getindex(ps::ProcessStats, num::Integer)
    if ps.type == :message
        return message(ps.processes[num])
    else 
        return status(ps.processes[num])
    end
end
function Base.setindex!(ps::ProcessStats, val, idx)
    if ps.type == :message
        return run(ps.processes[idx], val)
    else 
        error("Cannot set status")
    end
end
Base.getindex(ps::ProcessStats, num::Vector) = getindex.(Ref(ps), num)

