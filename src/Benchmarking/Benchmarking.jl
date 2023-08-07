let iterations = 10^7
    global function monteCarloBenchmark(sim, g, params, lTemp, gstate, gadj, rng, g_iterator, ghtype, energyFunc)
        beta::Float32 = 1/(lTemp[])
        
        idx::Int32 = rand(rng, g_iterator)
        
        Estate::Float32 = @inbounds gstate[idx]*energyFunc(g, gstate, gadj, idx, ghtype)

        minEdiff::Float32 = 2*Estate

        if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
        if params.updates > iterations
            quitSim(sim)
        else
            params.updates += 1
        end

    end
end

function simtest(sim, time = 5)
    quitSim(sim)
    Random.seed!(1234)
    rng = MersenneTwister(1234)

    g = graph(sim)
    state(g) .= initRandomState(g)
    createProcess(sim; rng)
    sleep(time)

    quitSim(sim)

end