# Pausable Timer
export PTimer, ispaused, wait, close, start, callback
"""
A pausable timer that can be started, stopped, and restarted.
"""
mutable struct PTimer
    callback::Function
    timer::Union{Timer, Nothing}
    closed_timer::Union{Timer, Nothing}
    delay::Float64
    interval::Float64
end

function PTimer(callback::Function, delay::Real; interval::Real = 0.)
    delay = convert(Float64, delay)
    interval = convert(Float64, interval)
    return PTimer(callback, Timer(callback, delay; interval), nothing, delay, interval)
end

function EmptyPTimer()
    PTimer(identity, nothing, nothing, 0., 0.)
end

getinterval(pt::PTimer) = pt.interval
getdelay(pt::PTimer) = pt.delay
Processes.ispaused(pt::PTimer) = isnothing(pt.timer)
function Base.wait(pt::PTimer)
    timer = isnothing(pt.timer) ? pt.closed_timer : pt.timer
    isnothing(timer) && return nothing
    try
        wait(timer)
    catch err
        err isa EOFError || rethrow()
    end
    return nothing
end

function Base.close(pt::PTimer)
    isnothing(pt.timer) && return nothing
    timer = pt.timer
    close(timer)
    pt.timer = nothing
    pt.closed_timer = timer
    return nothing
end

function Processes.start(pt::PTimer)
    close(pt)
    pt.closed_timer = nothing
    pt.timer = Timer(pt.callback, pt.delay, interval = pt.interval)
    return pt
end
callback(func, pt::PTimer) = pt.callback = func
