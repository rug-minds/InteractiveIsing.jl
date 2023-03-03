mutable struct Process
    status::Symbol
    message::Symbol
    func::Function
end

@setterGetter Process

mutable struct Processes
    procs::Vector{Process}
end

Processes(num) = Processes([Process(:Terminated, :Nothing, _ -> Sleep(0.5)) for _ in 1:num])

import Base: getindex, setindex!, length, iterate

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


