"""
Struct to define the runtime of a process
Is a struct so that dispatch can be used to choose the appropriate loop during compile time
"""
abstract type Runtime end
struct Indefinite <: Runtime end
struct Repeat{Num} <: Runtime 
    function Repeat{Num}() where Num 
        @assert Num isa Real "Repeats must be an integer" 
        new{Num}()
    end
end

repeats(r::Repeat{N}) where N = N

"""
Struct with all information to create the function within a process
"""
struct TaskFunc
    func::Any
    prepare::Any # Appropriate function to prepare the arguments
    cleanup::Any
    args::Union{NamedTuple, Base.Pairs}
    prepared_args::Union{NamedTuple, Base.Pairs}
    overrides::Any # Given as kwargs
    runtime::Runtime
    timeout::Float64 # Timeout in seconds
end

TaskFunc(func;prepare = (func, args) -> args, cleanup = (func, args) -> nothing, overrides::NamedTuple = (;), runtime = Indefinite(), args...) = 
    TaskFunc(func, prepare, cleanup, args, (;), overrides, runtime, 1.0)

getfunc(p::Process) = p.taskfunc.func
getprepare(p::Process) = p.taskfunc.prepare
getcleanup(p::Process) = p.taskfunc.cleanup
args(p::Process) = p.taskfunc.args
overrides(p::Process) = p.taskfunc.overrides
taskruntime(p::Process) = p.taskfunc.runtime
timeout(p::Process) = p.taskfunc.timeout

# struct TaskFuncs{N}
#     funcs::NTuple{N,Any}
#     prepares::NTuple{N,Any}
#     cleanups::NTuple{N,Any}
#     intervals::NTuple{N,Val}
#     args::Any
#     overrides::Any # Given as kwargs
#     runtime::Runtime
#     timeout::Float64
# end

# TaskFuncs(funcs::Tuple; prepares = fill((func, args) -> args, length(funcs)), cleanups = fill((func, args) -> nothing, length(funcs)), 
#     intervals = fill(Val{1}(), length(funcs)), args = (), kwargs = (), runtime = Indefinite(), timeout = 1.0) = 
#     TaskFuncs(funcs, prepares, cleanups, intervals, args, kwargs, runtime, timeout)

createtask!(p::Process) = createtask!(p, p.taskfunc.func; runtime = taskruntime(p), prepare = p.taskfunc.prepare, overrides = overrides(p), args(p)...)

function createtask!(process, @specialize(func); runtime = Indefinite(), prepare = (func, args) -> (;args), cleanup = (func, args) -> nothing, overrides = (;), skip_prepare = false, args...)   
    timeouttime = get(overrides, :timeout, 1.0)


    # If prepare is skipped, then the prepared arguments are already stored in the process
    prepared_args = nothing
    if skip_prepare
        prepared_args = process.taskfunc.prepared_args
    else
        # Prepare always has access to process and runtime
        prepared_args = prepare(func, (;proc = process, runtime, args...))
    end

    # Again add process and runtime if user didn't specify it in the prepare function
    algo_args = (;proc = process, runtime, prepared_args...)

    # Create new taskfunc
    process.taskfunc = TaskFunc(func, prepare, cleanup, args, prepared_args, overrides, runtime, timeouttime)

    # Add the overrides
    # They are not stored in the args of the taskfunc but separately
    # They are mostly for debugging or testing, so that the user can pass in the arguments to the function
    # These overrides should be removed at a restart, but not a refresh or pause
    task_args = (;algo_args..., overrides...)
    
    # Make the task
    process.task = @task @inline processloop(process, func, task_args, runtime)
end
export createtask!

# function createtasks!(p, funcs, intervals = ((Val{1}() for _ in 1:length(funcs))...,); prepare = fill((func, args) -> args, length(funcs)), 
#     cleanups = fill((func, args) -> nothing, length(funcs)), args = (), kwargs = (), runtime = Indefinite(), timeout = 1.0)

#     p.taskfunc = TaskFuncs(funcs, prepare, cleanups, intervals, args, kwargs, runtime, timeout)
#     p.task = @task @inline processloop(p, p.taskfunc.funcs, p.taskfunc.args, p.taskfunc.intervals)
# end
