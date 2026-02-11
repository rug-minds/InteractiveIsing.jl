function gatherSusceptibility(g, maxsteps = 100)
    avg = AvgData(convergence = 1e-3)
    avg_sq = AvgData(x -> x^2, convergence = 1e-3)
    step = 0
    while !(converged(avg) && converged(avg_sq)) && step <= maxsteps
        M = sum(state(g))
        push!(avg, M)
        push!(avg_sq, M)
        step += 1

    end
    return Temp(g)[]*(avg_sq[] - (avg[])^2), avg, avg_sq
end
export gatherSusceptibility


struct Susceptibility end

function Processes.init(::Susceptibility, args)
    magnetizations = []
    processsizehint!(args, magnetizations)
    return (;magnetizations)
end

function Susceptibility(args)
    (;M, magnetizations) = args
    push!(magnetizations, M)
end
