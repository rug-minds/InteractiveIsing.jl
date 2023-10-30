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
end

# Process() = Process(nothing, 0, Threads.SpinLock(), (true, :Nothing))
Process() = Process(nothing, 0, Threads.SpinLock(), false, nothing, nothing)
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


status(p::Process) = isrunning(p) ? :Running : :Quit
message(p::Process) = run(p) ? :Run : :Quit

isstarted(p::Process) = !isnothing(p.task) && istaskstarted(p.task)

isrunning(p::Process) = !isnothing(p.task) && !istaskdone(p.task)

ispaused(p::Process) = !isnothing(p.task) && istaskdone(p.task)

isdone(p::Process) = isnothing(p.task) && !isnothing(p.retval)

isidle(p::Process) = isnothing(p.task) || ispaused(p)

neverstarted(p::Process) = isnothing(p.task) && isnothing(p.retval)

function quit(p::Process)
    @atomic p.run = false
    p.retval = fetch(p)
    p.updates = 0
    p.task = nothing
    return true
end

function pause(p::Process)
    @atomic p.run = false
    p.retval = fetch(p)
    return true
end

function runtask(p, taskf::Function, objectref = nothing)
    p.objectref = objectref
    @atomic p.run = true
    t = taskf(p)
    p.task = t
    p.retval = nothing
    return t
end

function runtask(p, task::Task, objectref = nothing)
    p.objectref = objectref
    @atomic p.run = true
    task.sticky = false
    schedule(task)
    p.task = task
    p.retval = nothing
    return task
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

# @inline run(p::Process) = p.signal[1]
# @inline message(p::Process) = p.signal[2]

# @inline atomic_message(p::Process, val) = @atomic p.signal = (p.signal[1], val)

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
# @inline function atomic_message(p::Process; ignore_lock = false)
#     !ignore_lock && lock(p)
#     ret = (@atomic p.signal)[2]
#     !ignore_lock && unlock(p)
#     return ret
# end
@inline function atomic_run!(p::Process, val; ignore_lock = false)
    !ignore_lock && lock(p)
    # ret = @atomic p.signal = (val, p.signal[2])
    ret = @atomic p.run = val
    !ignore_lock && unlock(p)
    return ret
end
# @inline function message!(p::Process, val; ignore_lock = false)
#     !ignore_lock && lock(p)
#     ret = @atomic p.signal = (p.signal[1], val)
#     !ignore_lock && unlock(p)
#     return ret
# end
# @inline function signal!(p::Process, bool, symb; ignore_lock = false)
#     !ignore_lock && lock(p)
#     # ret = @atomic p.signal = (bool, symb)
#     ret = @atomic p.signal = bool
#     !ignore_lock && unlock(p)
#     return ret
# end

@inline running(p::Process) = p.status == :Running

# Base.put!(p::Process, val) = put!(p.refresh, val)
# Base.take!(p::Process) = take!(p.refresh)
# Base.isready(p::Process) = isready(p.refresh)
# Base.isopen(p::Process) = isopen(p.refresh)
# Base.close(p::Process) = close(p.refresh)
@inline Base.isempty(p::Process) = isempty(p.refresh)


mutable struct Processes <: AbstractVector{Process}
    procs::Vector{Process}
end

lock(p::Processes) = lock.(p.procs)
unlock(p::Processes) = unlock.(p.procs)

Base.size(p::Processes) = (length(p.procs),)

# Base.put!(p::Processes, idxs) = for idx in idxs; put!(p.procs[idx], true); end

Processes(num::Integer) = Processes([Process() for _ in 1:num])

getindex(processes::Processes, num) = processes.procs[num]

export getindex

setindex!(processes::Processes, val, idx) = setindex(processes.procs[num], val, idx)

length(processes::Processes) = length(processes.procs)
export length

iterate(processes::Processes, s = 1) = iterate(processes.procs, s)

struct ProcessStats <: AbstractVector{Symbol}
    processes::Processes
    type::Symbol
end

size(ps::ProcessStats) = (length(ps.processes),)

messages(procs::Processes) = ProcessStats(procs, :message)
messages(sim) = messages(sim.processes)
status(procs::Processes) = ProcessStats(procs, :status)
status(sim) = status(sim.processes)
export messages
export status

iterate(ps::ProcessStats, state = 1) = state > length(ps.processes.procs) ? nothing : (getfield(ps.processes.procs[state], ps.type), state + 1)

function getindex(ps::ProcessStats, num::Integer)
    if ps.type == :message
        return message(ps.processes[num])
    else 
        return status(ps.processes[num])
    end
end
function setindex!(ps::ProcessStats, val, idx)
    if ps.type == :message
        return run(ps.processes[idx], val)
    else 
        error("Cannot set status")
    end
end
getindex(ps::ProcessStats, num::Vector) = getindex.(Ref(ps), num)
# setindex!(ps::ProcessStats, val, idx) = setfield!(ps.processes.procs[idx], ps.type, val)

export iterate


