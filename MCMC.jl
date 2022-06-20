""" Updates the spin ising"""
# function updateMonteCarloQML!(g::IsingGraph, T)

#     beta = T>0 ? 1/T : Inf

#     idx = rand(ising_it(g))
#     Estate = 0

#     for jdx in g.adj[idx]
#         @inbounds Estate += -g.state[idx]*g.state[jdx]
#     end


#     if (Estate >= 0 || rand() < exp(2*beta*Estate))
#         @inbounds g.state[idx] *= -1
#     end
    
# end

function updateMonteCarloQML!(g::IsingGraph, T)

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    Estate = getH(g,idx)

    if (Estate >= 0 || rand() < exp(2*beta*Estate))
        @inbounds g.state[idx] *= -1
    end
    
end

function getH(g,idx)::Float32
    
    Estate = 0.
    if !g.weighted
        for conn in g.adj[idx]
            @inbounds Estate += -g.state[idx]*g.state[connIdx(conn)]
        end
    else
        for conn in g.adj[idx]
            @inbounds Estate += -connW(conn)*g.state[idx]*g.state[connIdx(conn)]
        end
    end
        

    return Estate
end
