"""
An observable that can be polled for updates.
This might be useful to track values that can be updated in locations unaware of the observable.
Can be equipped with a PTimer for regular polling.
Includes a lock to prevent race conditions during bidirectional synchronization.
"""
struct PolledObservable{O,F<:Function, P<:Union{PTimer, Nothing}} <: AbstractObservable{O}
    obs::Observable{O}
    pollingfunc::F
    timer::P
    lock::ReentrantLock
end

Observables.observe(po::PolledObservable) = po.obs

function PolledObservable(val::T, func::Function; interval = nothing) where {T}
    if !isnothing(interval)
        timer = PTimer((timer) -> poll!(val), 0., interval = interval)
    else
        timer = nothing
    end
    return PolledObservable(Observable(val), func, timer, ReentrantLock())
end

hastimer(po::PolledObservable) = !isnothing(po.timer)
togglepause(po::PolledObservable) = begin
    if hastimer(po)
        return false
    end
    if ispaused(po.timer)
        start(po.timer)
    else
        close(po.timer)
    end
    return true
end

Base.close(po::PolledObservable) = begin
    if hastimer(po)
        close(po.timer)
        return true
    end
    return false
end

Processes.start(po::PolledObservable) = begin
    if hastimer(po)
        start(po.timer)
        return true
    end
    return false
end


# notify(po::PolledObservable) = notify(po.obs)
# setindex!(po::PolledObservable, idx, val) = setindex!(po.obs, idx, val)
# getindex(po::PolledObservable, idx) = getindex(po.obs, idx)
Base.getproperty(po::PolledObservable, name::Symbol) = begin
    if name === :val
        return po.obs.val            # Custom property
    else
        return getfield(po, name)
    end
end

# listeners(po::PolledObservable) = listeners(po.obs)
Base.eltype(po::PolledObservable) = eltype(po.obs)

function poll!(obs::PolledObservable)
    # Try to acquire the lock, skip this poll cycle if busy
    if trylock(obs.lock)
        checkedval = obs.pollingfunc(obs)
        # println("Checked val: ", checkedval)
        if checkedval != obs[]
            # println("Updating obs[] to: ", checkedval)
            obs[] = checkedval
        end
        unlock(obs.lock)
    end
    # If lock is busy, just skip this poll - we'll catch up on the next one
end

# Override setindex! to use the lock
function Base.setindex!(po::PolledObservable, value)
    lock(po.lock)
    po.obs[] = value
    unlock(po.lock)
end

Base.getindex(po::PolledObservable) = po.obs[]

