export MetropolisGB

struct MetropolisGB <: MCAlgorithm end
requires(::Type{MetropolisGB}) = Δi_H()


function Total_H(s, wij, B)
    totalsum = zero(eltype(s))
    for stateidx in eachindex(s)
        partsum = zero(eltype(s))

        @turbo for ptr in nzrange(wij, stateidx)
            j = wij.rowval[ptr]
            wij = wij.nzval[ptr]
            partsum += wij * s[j]
        end
        totalsum += -s[stateidx] * (partsum + B[])
    end
    return totalsum
end

function GlobalBH(i, s, newstate, oldstate, wij, B)
    cumsum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, i)
        j = wij.rowval[ptr]
        wij = wij.nzval[ptr]
        cumsum += wij * s[j] 
    end
    return (oldstate-newstate) * (cumsum + B[])
end

function Processes.prepare(::MetropolisGB, @specialize(args))
    (;g) = args
    s = g.state
    wij = g.adj
    gparams = g.params
    iterator = ising_it(g)
    # rng = MersenneTwister()
    rng = Random.GLOBAL_RNG
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)
    M = Ref(sum(g.state))
    lmeta = LayerMetaData(g[1])
    B = Ref(0f0)
    total_E = Ref(Total_H(s, wij, B))
    return (;s, wij, gparams, iterator, ΔH, lmeta, rng, M, B, total_E)
end

@inline function (::MetropolisGB)(@specialize(args))
    #Define vars
    (;g, s, wij, lmeta, iterator, total_E, rng, M, B) = args
    i = rand(rng, iterator)
    MetropolisGB(i, g, s, wij, M, B, total_E, rng, lmeta)
end

@inline function MetropolisGB(i, g, s::Vector{T}, wij, M, B, total_E, rng, lmeta) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds s[i]
    
    newstate = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))   

    ΔE = @inline GlobalBH(i, s, newstate, oldstate, wij, B)

    efac = exp(-β*ΔE)
    randnum = rand(rng, T)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds s[i] = newstate 
        M[] += (newstate - oldstate)
        total_E[] += ΔE - (oldstate-newstate)*B[]
    end
    
    return nothing
end


# @inline function updateMetropolis(g, s::Vector{T}, wij, gparams, ΔH)
#     β = 1f0/(temp(g))

#     oldstate = @inbounds s[idx]
    
#     ΔE = ΔH(g, oldstate, 1, s, wij, idx, gstype, Discrete)

#     if (ΔE <= 0f0 || rand(rng, Float32) < exp(-β*ΔE))
#         @inbounds s[idx] *= -Int8(1)
#     end
#     return nothing
# end





