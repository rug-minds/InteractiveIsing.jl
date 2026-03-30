# Pausable Timer
export PTimer, ispaused, wait, close, start, callback
"""
A pausable timer that can be started, stopped, and restarted.
"""
mutable struct PTimer
    callback::Function
    timer::Union{Timer, Nothing}
    delay::Float64
    interval::Float64
end

function PTimer(callback::Function, delay::Real; interval::Real = 0.)
    delay = convert(Float64, delay)
    interval = convert(Float64, interval)
    return PTimer(callback, Timer(callback, delay; interval), delay, interval)
end

function EmptyPTimer()
    PTimer(identity, nothing, 0.,0.)
end

getinterval(pt::PTimer) = pt.interval
getdelay(pt::PTimer) = pt.delay
Processes.ispaused(pt::PTimer) = isnothing(pt.timer)
Base.wait(pt::PTimer) = if !isnothing(pt.timer); wait(pt.timer); end
Base.close(pt::PTimer) = if !isnothing(pt.timer) ; close(pt.timer); pt.timer = nothing end
Processes.start(pt::PTimer) = begin if !isnothing(pt.timer); close(pt.timer) end ; pt.timer = Timer(pt.callback, pt.delay, interval = pt.interval); end
callback(func, pt::PTimer) = pt.callback = func
