mutable struct Process
    status::Ref{Symbol}
    message::Ref{Symbol}
    func::Ref
end

@inline status(process) = process.status[]
@inline status(process, val) = process.status[] = val

@inline message(process) = process.message[]
@inline message(process, val) = process.message[] = val

mutable struct Processes
    status::Vector{Ref{Symbol}}
    messages::Vector{Ref{Symbol}}
    funcs::Vector{Ref}
    processes::Vector{Process}

    function Processes(num)
        p = new(
            repeat([Ref(:Terminated)], num), 
            repeat([Ref(:Nothing)], num), 
            repeat([Ref(_ -> ())], num),
        )

        p.processes = [Process(p.status[i], p.messages[i], p.funcs[i]) for i in eachindex(p.status) ]

        return p
    end
end

function Base.getindex(processes::Processes, num)
    return processes.processes[num]
end

Base.length(processes::Processes) = length(processes.processes)

Base.iterate(processes::Processes) = iterate(processes.processes)

@setterGetter Processes


    





