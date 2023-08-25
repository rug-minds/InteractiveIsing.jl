# Probably add a ref to the graph it's working on
import Base: Threads.SpinLock, lock, unlock
mutable struct Process{F <: Function}
    status::Symbol
    func::F       
    p_updates::Int   
    # To make sure other processes don't interfere
    lock::SpinLock 
    @atomic signal::Tuple{Bool,Symbol}
end

Process() = Process(:Terminated, _ -> Sleep(0.5), 0, Threads.SpinLock(), (true, :Nothing))
@setterGetter Process lock


@inline inc(p::Process) = p.p_updates += 1
@inline reset!(p::Process) = p.p_updates = 0

@inline run(p::Process) = p.signal[1]
@inline message(p::Process) = p.signal[2]

@inline atomic_message(p::Process, val) = @atomic p.signal = (p.signal[1], val)

@inline function atomic_run(p::Process; ignore_lock = false)
    !ignore_lock && lock(p)
    ret = (@atomic p.signal)[1]
    !ignore_lock && unlock(p)
    return ret
end
@inline function atomic_message(p::Process; ignore_lock = false)
    !ignore_lock && lock(p)
    ret = (@atomic p.signal)[2]
    !ignore_lock && unlock(p)
    return ret
end
@inline function run!(p::Process, val; ignore_lock = false)
    !ignore_lock && lock(p)
    ret = @atomic p.signal = (val, p.signal[2])
    !ignore_lock && unlock(p)
    return ret
end
@inline function message!(p::Process, val; ignore_lock = false)
    !ignore_lock && lock(p)
    ret = @atomic p.signal = (p.signal[1], val)
    !ignore_lock && unlock(p)
    return ret
end
@inline function signal!(p::Process, bool, symb; ignore_lock = false)
    !ignore_lock && lock(p)
    ret = @atomic p.signal = (bool, symb)
    !ignore_lock && unlock(p)
    return ret
end

@inline running(p::Process) = p.status == :Running

@inline lock(p::Process) = lock(p.lock)
@inline unlock(p::Process) =  unlock(p.lock)

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

getindex(ps::ProcessStats, num::Integer) = getfield(ps.processes.procs[num], ps.type)
getindex(ps::ProcessStats, num::Vector) = getfield.(ps.processes.procs[num], ps.type)
setindex!(ps::ProcessStats, val, idx) = setfield!(ps.processes.procs[idx], ps.type, val)

export iterate


