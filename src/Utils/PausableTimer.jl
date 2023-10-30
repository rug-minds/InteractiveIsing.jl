# Pausable Timer
struct PTimer{T <: Function}
    callback::T
    timer::Union{Timer, Nothing}
    delay::Float64
    interval::Float64
end

function PTimer(callback::Function, delay::Float64; interval::Float64 = 0)
    return PTimer(callback, Timer(callback, delay; interval), delay, interval)
end

ispaused(pt::PTimer) = isnothing(pt.timer)
Base.close(pt::PTimer) = begin close(pt.timer); pt.timer = nothing end
start(pt::PTimer) = if isnothing(pt.timer); pt.timer = Timer(pt.callback, pt.delay, interval = pt.interval); end
