const timedFunctions = Dict{String, Function}()

function timerFuncs(sim)
    for func in values(timedFunctions)
        func(sim)
    end
end