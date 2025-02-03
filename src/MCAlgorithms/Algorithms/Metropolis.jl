export Metropolis

struct Metropolis <: MCAlgorithm end
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
    # (;g) = args
    # return (;g,
    #         gstate = g.state,
    #         gadj = g.adj,
    #         gparams = g.params,
    #         iterator = ising_it(g, g.stype),
    #         rng = MersenneTwister(),
    #         lt = g[1],
    #         ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian),
    #         )
    (;g) = args
    gstate = g.state
    # println("Got state")
    gadj = g.adj
    # println("Got adj")
    # gparams = g.params
    # println("Got params")
    iterator = ising_it(g, g.stype)
    # println("Got iterator")
    rng = MersenneTwister()
    # println("Got rng")
    lt = g[1]
    # println("Got lt")
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)
    # ΔH = example_ising

    # return (;g, gstate, gadj, gparams, iterator, rng, lt, ΔH)
    return (;g, gstate, gadj, iterator, rng, ΔH, lt)
end

@inline function Metropolis(@specialize(args))
    #Define vars
    # (;g, gstate, gadj, gparams, iterator, ΔH, lt, rng) = args
    (;g, gstate, gadj, iterator, ΔH, lt, rng) = args
    i = rand(rng, iterator)
    # Metropolis(i, g, gstate, gadj, gparams, ΔH, lt)
    Metropolis(i, g, gstate, gadj, 1, ΔH, lt)
end

@inline function Metropolis(i, g, gstate::Vector{T}, gadj, gparams, ΔH, lt) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = @inline sampleState(statetype(lt), oldstate, rng, stateset(lt))   

    ΔE = @inline ΔH(i, gstate, newstate, oldstate, gadj, gparams, lt)

    efac = exp(-β*ΔE)
    randnum = rand(rng, Float32)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[i] = newstate 
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





