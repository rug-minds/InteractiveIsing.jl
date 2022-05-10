
""" Updates the spin ising"""
function updateMonteCarlo(state::Vector{Int8},cd::Dict{Int32,Vector{Int32}}, T, J)
    beta = T>0 ? 1/T : Inf
    
    oldstate = copy(state)
  
    # Why does this only work correctly if it is threaded?
    it::UnitRange{Int32} = 1:length(oldstate)
    Threads.@threads for idx in it
    # for idx in it
        
        if rand() > 0.5
            continue
        end
        
        Estate = 0
        
        for jdx in cd[idx]
            Estate += -J*oldstate[jdx]*oldstate[jdx]
        end
        
        if (Estate >= 0 || rand() < exp(2*beta*Estate))
                state[idx] *= -1
        end
    end

end

function updateMonteCarloQMLOLD(julia_display, g::IsingGraph, T, J)
    beta = T>0 ? 1/T : Inf
    newState = copy(g.state)

    it::UnitRange{Int32} = 1:g.size
    # Threads.@threads for idx in it
    for idx in it
        # if rand() > 0.5
        #     continue
        # end
        
        Estate = 0

        for jdx in g.adj[idx]
            Estate += -J*g.state[idx]*g.state[jdx]
        end

        
        
        if (Estate >= 0 || rand() < exp(2*beta*Estate))
                newState[idx] *= -1
        end
    end

    g = IsingGraph(newState, g.adj)
    dispIsing(julia_display,g)
end

function updateMonteCarloQML!(g::IsingGraph, T, J)

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    Estate = 0

    for jdx in g.adj[idx]
        Estate += -J*g.state[idx]*g.state[jdx]
    end


    if (Estate >= 0 || rand() < exp(2*beta*Estate))
            g.state[idx] *= -1
    end
    
end

# OLD
function updateMonteCarloQMLObs!(g, T, J)

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g[]))

    Estate = 0

    for jdx in g[].adj[idx]
        Estate += -J*g[].state[idx]*g[].state[jdx]
    end


    if (Estate >= 0 || rand() < exp(2*beta*Estate))
            g[].state[idx] *= -1
    end
    
end

