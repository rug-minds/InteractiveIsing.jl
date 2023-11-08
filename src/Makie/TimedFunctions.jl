struct TimedFunctions <: Function
    calls::Vector{Function}
end

const timedFunctions = TimedFunctions(Vector{Function}())