""" Updates the spin ising"""
__precompile__()

module IsingMetropolis
push!(LOAD_PATH, pwd())

# include("IsingGraphs.jl")
using IsingGraphs

export updateMonteCarloIsing!, deltED, deltEC

# Discrete


# Continuous


function updateMonteCarloIsing!(g::IsingGraph{Int8}, T; getE::Function = HFunc)

    @inline function deltE(Estate)
        return -2*Estate
    end

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    Estate = g.state[idx]*getE(g,idx)

    if (Estate >= 0 || rand() < exp(-beta*deltE(Estate)))
        @inbounds g.state[idx] *= -1
    end
    
end

function updateMonteCarloIsing!(g::IsingGraph{Float32}, T; getE::Function = HFunc)

    @inline function deltE(efac,newstate,oldstate)
        return efac*(newstate-oldstate)
    end

    @inline function sampleCState()
        Float32(2*(rand()-.5))
    end

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))
     
    oldstate = g.state[idx]

    efactor = getE(g,idx, oldstate)

    newstate = sampleCState()
    
    Ediff = deltE(efactor,newstate,oldstate)
    if (Ediff < 0 || rand() < exp(-beta*Ediff))
        @inbounds g.state[idx] = newstate 
    end
    
end


end

"""
OLD STUFF

function updateMonteCarloIsingOLD!(g::CIsingGraph, T)

    # No self energy
    @inline function deltE(Estate,newstate,oldstate)
        return Estate*(newstate/oldstate-1)
    end

    @inline function deltESelf(Estate,newstate,oldstate)
        ratio = newstate/oldstate
        return Estate*(ratio-1) - oldstate^2*(ratio)+newstate^2
    end

    @inline function sampleCState()
        2*(rand()-.5)
    end

    if g.selfE
        deltEFunc = deltESelf
    else
        deltEFunc = deltE
    end

    beta = T>0 ? 1/T : Inf

    idx = rand(ising_it(g))

    oldstate = g.state[idx]
    Estate = getH(g,oldstate,idx)

    newstate = sampleCState()
    

    if (Estate >= 0 || rand() < exp(-beta*deltEFunc(Estate,newstate,oldstate)))
        @inbounds g.state[idx] = newstate
    end
    
end
"""

