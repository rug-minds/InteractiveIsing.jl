# Pausable Timer
mutable struct PTimer
    callback::Function
    timer::Union{Timer, Nothing}
    delay::Float64
    interval::Float64
end

function PTimer(callback::Function, delay::Float64; interval::Float64 = 0)
    return PTimer(callback, Timer(callback, delay; interval), delay, interval)
end

ispaused(pt::PTimer) = isnothing(pt.timer)
Base.wait(pt::PTimer) = if !isnothing(pt.timer); wait(pt.timer); end
Base.close(pt::PTimer) = if !isnothing(pt.timer) ; close(pt.timer); pt.timer = nothing end
start(pt::PTimer) = if isnothing(pt.timer); pt.timer = Timer(pt.callback, pt.delay, interval = pt.interval); end
callback(func, pt::PTimer) = pt.callback = func