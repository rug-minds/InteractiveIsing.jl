export Metropolis

struct Metropolis <: MCAlgorithm end
struct MetropolisNew <: MCAlgorithm end
requires(::Type{Metropolis}) = Δi_H()


function reserved_symbols(::Type{Metropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function example_ising(i, gstate, newstate, oldstate, gadj, gparams, lt)
    cumsum = zero(eltype(gstate))
    @turbo for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        wij = gadj.nzval[ptr]
        cumsum += wij * gstate[j]
    end
    return (oldstate-newstate) * cumsum
end

function Processes.prepare(::Metropolis, @specialize(args))
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
    return (;gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M)
end

@inline function (::Metropolis)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M) = args
    i = rand(rng, iterator)
    Metropolis(i, g, gstate, gadj, gparams, M, ΔH, rng, lmeta)
end

@inline function Metropolis(i, g, gstate::Vector{T}, gadj, gparams, M, ΔH, rng, lmeta) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))   

    ΔE = @inline ΔH(i, gstate, newstate, oldstate, gadj, gparams, lmeta)

    efac = exp(-β*ΔE)
    randnum = rand(rng, T)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[i] = newstate 
        M[] += (newstate - oldstate)
    end
    
    return nothing
end

function Processes.prepare(::MetropolisNew, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    gparams = g.params
    iterator = ising_it(g)
    hamiltonian = g.hamiltonian
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    lmeta = LayerMetaData(g[1])
    return (;g, gstate, gadj, gparams, iterator, hamiltonian, lmeta, rng, M)
end

@inline function (::MetropolisNew)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, hamiltonian, lmeta, rng, M) = args
    j = rand(rng, iterator)
    MetropolisNew(args, j, g, gstate, gadj, gparams, hamiltonian, M, rng, lmeta)
end

@inline function MetropolisNew(args, j, g, gstate::Vector{T}, gadj, gparams, hamiltonian, M, rng, lmeta) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[j]
    
    newstate = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))   

    ΔE = @inline dh(hamiltonian, (;args..., newstate); j)

    efac = exp(-β*ΔE)
    randnum = rand(rng, T)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[j] = newstate 
        M[] += (newstate - oldstate)
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





