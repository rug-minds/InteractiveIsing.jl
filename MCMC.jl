""" Updates the spin ising"""
__precompile__()
module IsingMetropolis

# include("IsingGraphs.jl")
using ..IsingGraphs

export updateMonteCarloIsing!

function updateMonteCarloIsing!(g::IsingGraph, T)

    @inline function deltE(Estate)
        return 2*Estate
    end

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    Estate = getH(g,idx)

    if (Estate >= 0 || rand() < exp(beta*deltE(Estate)))
        @inbounds g.state[idx] *= -1
    end
    
end

function updateMonteCarloIsing!(g::CIsingGraph, T)

    # No self energy
    # @inline function deltE(Estate,newstate,oldstate)
    #     return Estate*(newstate/oldstate-1)
    # end

    @inline function deltE(Estate,newstate,oldstate)
        ratio = newstate/oldstate
        return Estate*(ratio-1) - oldstate^2*(ratio)+newstate^2
    end

    @inline function sampleCState()
        2*(rand()-.5)
    end

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    oldstate = g.state[idx]
    Estate = getH(g,oldstate,idx)

    newstate = sampleCState()
    

    if (Estate >= 0 || rand() < exp(-beta*deltE(Estate,newstate,oldstate)))
        @inbounds g.state[idx] = newstate
    end
    
end


end


