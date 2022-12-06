mutable struct IsingParams
    updates::Int64

    shouldRun::Bool
    isRunning::Bool

    started::Bool
end

IsingParams() = IsingParams(0,true, true, false)