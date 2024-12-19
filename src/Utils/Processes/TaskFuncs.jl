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
    overrides::Any # Given as kwargs
    runtime::Runtime
    timeout::Float64 # Timeout in seconds
end

TaskFunc(func::Function; prepare = (func, args) -> args, cleanup = (func, args) -> nothing, overrides::NamedTuple = (;), args...) = 
    TaskFunc(func, prepare, cleanup, args, overrides, Indefinite(), 1.0)

getfunc(p::Process) = p.taskfunc.func
getprepare(p::Process) = p.taskfunc.prepare
getcleanup(p::Process) = p.taskfunc.cleanup
args(p::Process) = p.taskfunc.args
overrides(p::Process) = p.taskfunc.overrides
runtime(p::Process) = p.taskfunc.runtime
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

createtask!(p::Process) = createtask!(p, p.taskfunc.func; runtime = runtime(p), prepare = p.taskfunc.prepare, overrides = overrides(p), args(p)...)

function createtask!(process, @specialize(func); runtime = Indefinite(), prepare = (func, oldargs, newargs) -> (newargs), cleanup = (func, args) -> nothing, overrides = (;), args...)   
    timeouttime = get(overrides, :timeout, 1.0)

    # Get the runtime or set it to indefinite
    
    # Add the process to the arguments
    newargs = (;proc = process, runtime, args...)
    # Get the old arguments
    oldargs = process.taskfunc.args

    args = (;oldargs..., newargs...)

    # Prepare the arguments for the algorithm
    algo_args = prepare(func, args)

    # Again add process and runtime if user didn't specify it in the prepare function
    algo_args = (;proc = process, runtime, algo_args...)

    # Create new taskfunc
    process.taskfunc = TaskFunc(func, prepare, cleanup, algo_args, overrides, runtime, timeouttime)

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
