export MetropolisGB

struct MetropolisGB <: MCAlgorithm end
requires(::Type{MetropolisGB}) = Δi_H()


function Total_H(gstate, gadj, B)
    totalsum = zero(eltype(gstate))
    for stateidx in eachindex(gstate)
        partsum = zero(eltype(gstate))

        @turbo for ptr in nzrange(gadj, stateidx)
            j = gadj.rowval[ptr]
            wij = gadj.nzval[ptr]
            partsum += wij * gstate[j]
        end
        totalsum += -gstate[stateidx] * (partsum + B[])
    end
    return totalsum
end

function GlobalBH(i, gstate, newstate, oldstate, gadj, B)
    cumsum = zero(eltype(gstate))
    @turbo for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        wij = gadj.nzval[ptr]
        cumsum += wij * gstate[j] 
    end
    return (oldstate-newstate) * (cumsum + B[])
end

function Processes.prepare(::MetropolisGB, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    gparams = g.params
    iterator = ising_it(g)
    # rng = MersenneTwister()
    rng = Random.GLOBAL_RNG
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)
    M = Ref(sum(g.state))
    lmeta = LayerMetaData(g[1])
    B = Ref(0f0)
    total_E = Ref(Total_H(gstate, gadj, B))
    return (;g, gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M, B, total_E)
end

@inline function (::MetropolisGB)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, lmeta, iterator, total_E, rng, M, B) = args
    i = rand(rng, iterator)
    MetropolisGB(i, g, gstate, gadj, M, B, total_E, rng, lmeta)
end

@inline function MetropolisGB(i, g, gstate::Vector{T}, gadj, M, B, total_E, rng, lmeta) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))   

    ΔE = @inline GlobalBH(i, gstate, newstate, oldstate, gadj, B)

    efac = exp(-β*ΔE)
    randnum = rand(rng, T)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[i] = newstate 
        M[] += (newstate - oldstate)
        total_E[] += ΔE - B[]
    end
    
    return nothing
end


# @inline function updateMetropolis(g, gstate::Vector{T}, gadj, gparams, ΔH)
#     β = 1f0/(temp(g))

#     oldstate = @inbounds gstate[idx]
    
#     ΔE = ΔH(g, oldstate, 1, gstate, gadj, idx, gstype, Discrete)

#     if (ΔE <= 0f0 || rand(rng, Float32) < exp(-β*ΔE))
#         @inbounds gstate[idx] *= -Int8(1)
#     end
#     return nothing
# end





