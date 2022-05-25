""" Updates the spin ising"""
function updateMonteCarloQML!(g::IsingGraph, T, J)

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    Estate = 0

    for jdx in g.adj[idx]
        @inbounds Estate += -J*g.state[idx]*g.state[jdx]
    end


    if (Estate >= 0 || rand() < exp(2*beta*Estate))
        @inbounds g.state[idx] *= -1
    end
    
end

