"""
An observable that can be polled for updates.
This might be useful to track values that can be updated in locations unaware of the observable.
Can be equipped with a PTimer for regular polling.
"""
struct PolledObservable{O,F<:Function, P<:Union{PTimer, Nothing}} <: AbstractObservable{O}
    obs::Observable{O}
    pollingfunc::F
    timer::P
end

Observables.observe(po::PolledObservable) = po.obs

function PolledObservable(val::T, func::Function; interval = nothing) where {T}
    if !isnothing(interval)
        timer = PTimer((timer) -> poll!(val), 0., interval = interval)
    else
        timer = nothing
    end
    return PolledObservable(Observable(val), func, timer)
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
    checkedval = obs.pollingfunc(obs)
    if checkedval != obs[]
        obs[] = checkedval
    end
end

