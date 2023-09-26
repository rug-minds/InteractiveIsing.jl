function gatherSusceptibility(g, maxsteps = 100)
    avg = AvgData(convergence = 1e-3)
    avg_sq = AvgData(x -> x^2, convergence = 1e-3)
    step = 0
    while !(converged(avg) && converged(avg_sq)) && step <= maxsteps
        M = sum(state(g))
        push!(avg, M)
        push!(avg_sq, M)
        step += 1
        sleep(1/1000)
        # println("Step $step")
        # println(step > maxsteps)
        # println(converged(avg))
        # println(converged(avg_sq))
    end
    return Temp(sim(g))[]*(avg_sq[] - (avg[])^2), avg, avg_sq
end
export gatherSusceptibility
