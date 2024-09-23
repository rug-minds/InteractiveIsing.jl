# Probably add a ref to the graph it's working on
import Base: Threads.SpinLock, lock, unlock
abstract type Runtime end
struct Indefinite <: Runtime end
struct Repeat{Num} <: Runtime 
    function Repeat{Num}() where Num 
        @assert Num isa Real "Repeats must be an integer" 
        new{Num}()
    end
end

repeats(r::Repeat{N}) where N = N
struct TaskFunc
    func::Any
    args::Any
    kwargs::Any
    runtime::Runtime
end

TaskFunc(func::Function) = TaskFunc(func, (), (), Indefinite())
# TaskFunc(n::Nothing, args::Any, rt::Any = nothing) = nothing

mutable struct Process
    id::UUID
    taskfunc::Union{Nothing,TaskFunc}
    task::Union{Nothing, Task}
    loopidx::Int   
    # To make sure other processes don't interfere
    lock::SpinLock 
    @atomic run::Bool
    @atomic paused::Bool
    objectref::Any
    retval::Any
    errorlog::Any
    algorithm::Any #Ref to the algorithm being run
end

# List of processes in use
const processlist = Dict{UUID, WeakRef}()
register_process!(p) = let id = uuid1(); processlist[id] = WeakRef(p); id end

function Base.finalizer(p::Process)
    quit(p)
    delete!(processlist, p.id)
end


# Process() = Process(nothing, 0, Threads.SpinLock(), (true, :Nothing))
function Process(func = nothing, repeats::Int = 0, args...) 
    rt = repeats == 0 ? Indefinite() : Repeat{repeats}()
    return Process(func, rt, args...)
end

function Process(func = nothing, rt::Runtime = Indefinite(), args...)
    p = Process(uuid1(), TaskFunc(func, tuple(args...), (), rt), nothing, 1, Threads.SpinLock(), false, false, nothing, nothing, nothing, nothing)
    register_process!(p)
    return p
end
@setterGetter Process lock run

function Base.show(io::IO, p::Process)
    if !isnothing(p.task) && p.task._isexception
        print(io, "Error in process")
        return display(p.task)
    end
    statestring = ""
    if ispaused(p)
        statestring = "Paused"
    elseif isrunning(p)
        statestring = "Running"
    elseif isdone(p)
        statestring = "Finished"
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

function choose_argsNEW(p::Process, prepared_args, specified_args = nothing)
    masked_args = prepared_args
    if ispaused(p) # If paused, use the old kwargs
        #TODO: Is the if here neccesary?
        # Shouldn't it be decided by wether there is a retval
        masked_args = replacekwargs(prepared_args, fetch(p))
    end
    if !isnothing(specified_args)
        masked_args = replacekwargs(prepared_args, specified_args)
    end

    return masked_args
end


status(p::Process) = isrunning(p) ? :Running : :Quit
message(p::Process) = run(p) ? :Run : :Quit

isstarted(p::Process) = !isnothing(p.task) && istaskstarted(p.task)

isrunning(p::Process) = isstarted(p) && !istaskdone(p.task)

ispaused(p::Process) = !isnothing(p.task) && p.paused

isdone(p::Process) = !isnothing(p.task) && istaskdone(p.task)

isidle(p::Process) = isdone(p.task) || ispaused(p)

"""
Can be used for a new process
"""
isfree(p::Process) = !isrunning(p) && !ispaused(p)
"""
Is currently used for running,
    can be paused
"""
isused(p::Process) = isrunning(p) || ispaused(p)

export start, restart, quit, pause
function start(p::Process, sticky = false)
    @assert isfree(p) "Process is already in use"
    @assert !isnothing(p.taskfunc) "No task to run"

    createtask!(p)
    runtask!(p, sticky)
    return true
end   

function quit(p::Process)
    @atomic p.run = false
    @atomic p.paused = false
    @sync p.task
    p.loopidx = 1
    return true
end

function pause(p::Process)
    @atomic p.run = false
    @atomic p.paused = true
    @sync p.task
    try 
        p.retval = fetch(p)
    catch e
        p.errorlog = e
    end
    return true
end

function unpause(p::Process)
    # @atomic p.run = true
    # @atomic p.paused = false
    # runtask!(p)
    # return true
    start(p)
end

function refresh(p::Process)
    @assert !isnothing(p.taskfunc) "No task to run"
    pause(p)
    unpause(p)
    return true
end

function restart(p::Process, sticky = false)
    @assert !isnothing(p.taskfunc) "No task to run"
    quit(p)
    createtask!(p)
    runtask!(p, sticky)
    return true
end

# function createtask(p::Process, @specialize(func), @specialize(args), runtime::RT; kwargs...) where RT <: Runtime
#     # task = nothing
    
#     println("HERE")
#     # task = Threads.@spawn begin 
#         (;g) = args
#         algo_args = prepare(func, g; kwargs...)

#         masked_args = choose_args(p, algo_args; kwargs...)
        
#         @inline indefiniteloop(p, func, masked_args)
#         return kwargs
#     # end
#     # println(args)
#     # if runtime isa Indefinite
#     #     task = Threads.@spawn begin 
#     #         (;g) = args
#     #         algo_args = prepare(func, g; kwargs...)

#     #         masked_args = choose_argsNEW(p, algo_args; kwargs...)
#     #         @inline indefiniteloop(p, func, masked_args)
#     #         return kwargs
#     #     end
#     # elseif runtime isa Repeat
#     #     task = @task @inline begin 
#     #         repeatloop(p, func, args, repeats(runtime))
#     #         return kwargs
#     #     end
#     # end
#     # return task
# end
# createtask!(p::Process, func, args, runtime) = p.task = createtask(p, func, args, runtime)

# function indefiniteloop(@specialize(p), @specialize(func), @specialize(args))
#     println("In indefiniteloop")
#     println("Running on thread $(Threads.threadid())")
#     while run(p) 
#         @inline func(args)
#         inc(p) 
#         GC.safepoint() 
#     end
# end

function repeatloop(p, func, args, repeats)
    for _ in loopidx(p):repeats
        if !run(p)
            break
        end
        @inline func(p, args)
        inc(p)
        GC.safepoint()
    end
end



"""
Create a task from the taskfunction
"""
createtask(p::Process) = createtask(p, p.taskfunc.func, p.taskfunc.args, p.taskfunc.runtime)

"""
Create a task and assign it to the process
"""
createtask!(p::Process) = p.task = createtask(p)


function newprocess(@specialize(func), @specialize(args::Tuple) = () , runtime::RT = Indefinite()) where RT <: Runtime
    newp = Process(func, runtime, args...)
    register_process!(newp)
    # createtask!(newp, func, args, runtime)
    # createtask!(newp, func, args, runtime)
    # runtask!(newp)
    start(newp)
    return newp
end

newprocess(func, repeats::Int = 0, args...) = let rt = repeats == 0 ? Indefinite() : Repeat{repeats}(); newprocess(func, args, rt); end

export newprocess

# function maketask!(@specialize(g), @specialize(func), @specialize(p), @specialize(args) = (;) , runtime::RT = Indefinite()) where RT <: Runtime
#     # p.taskfunc = TaskFunc(func, args, runtime)
#     # @atomic p.run = true
#     println("NOW")
#     createtaskNEW(g, p, func, args, runtime)
#     # start(p)
#     return p
# end

# function createtaskNEW(g, p::Process, @specialize(func), @specialize(args), runtime::RT; kwargs...) where RT <: Runtime
#     # task = nothing
    
#     println("HERE")
#     # task = Threads.@spawn begin 
#         algo_args = prepare(func, g; kwargs...)

#         masked_args = choose_argsNEW(p, algo_args; kwargs...)
        
#         indefiniteloop(p, func, masked_args)
#         return kwargs
#     # end
#     # println(args)
#     # if runtime isa Indefinite
#     #     task = Threads.@spawn begin 
#     #         (;g) = args
#     #         algo_args = prepare(func, g; kwargs...)

#     #         masked_args = choose_argsNEW(p, algo_args; kwargs...)
#     #         @inline indefiniteloop(p, func, masked_args)
#     #         return kwargs
#     #     end
#     # elseif runtime isa Repeat
#     #     task = @task @inline begin 
#     #         repeatloop(p, func, args, repeats(runtime))
#     #         return kwargs
#     #     end
#     # end
#     # return task
# end

"""
Takes the task prepared in p and then runs it
p is the process
"""
function runtask!(p::Process, sticky = false)
    @assert !isnothing(p.task) "No task to run, create a task first"
    @assert !isrunning(p) "Task is already running"
    @assert !isdone(p) "Task is already done"

    @atomic p.run = true
    @atomic p.paused = false

    p.task.sticky = sticky
    Threads._spawn_set_thrpool(p.task, :default)
    schedule(p.task)
    return true
end

function runtask(p::Process, task::Task, objectref = nothing; run = true, sticky = false)
    # Maybe this is not needed
    p.objectref = objectref
    
    @atomic p.run = run

    task.sticky = sticky
    # p.taskref = task
    p.task = task
    p.retval = nothing
    schedule(p.task)
    return p.task
end

# TODO: Does this make sense?
"""
Give a function and then creates a task that is run by the process
The function needs to take an object as a reference
"""
function runtask(p, taskf::Function, repeats = 0; objectref = nothing, run = true)
    p.objectref = objectref
    @atomic p.run = run
    @atomic p.paused = !run
    # t.taskfunc = TaskFunc(taskf, Repeat{repeats})
    # t = Task(t.taskfunc.func(p, objectref))
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

reset!(p::Process) = (p.loopidx = 1; p.retval = nothing; p.task = nothing; @atomic p.run = false; p.objectref = nothing)

run(p::Process) = p.run
run(p::Process, val) = @atomic p.run = val

@inline Base.wait(p::Process) = if !isnothing(p.task) wait(p.task) else nothing end
@inline Base.fetch(p::Process) = if !isnothing(p.task) fetch(p.task) else nothing end

@inline inc(p::Process) = p.loopidx += 1

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



