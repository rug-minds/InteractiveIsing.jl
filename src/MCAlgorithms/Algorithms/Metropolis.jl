export Metropolis
struct Metropolis<: MCAlgorithm end
struct deltaH end

function Processes.prepare(::Metropolis, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    # params = g.params
    iterator = ising_it(g)
    hamiltonian = init!(g.hamiltonian, g)
    deltafunc = deltaH(hamiltonian)
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    Δs_j = Ref(zero(eltype(g.state)))

    lmeta = LayerMetaData(g[1])
    return (;gstate, gadj, iterator, hamiltonian, deltafunc, lmeta, rng, M, Δs_j)
end

@inline function (::Metropolis)(@specialize(args))
    #Define vars
    (;iterator, rng) = args
    j = rand(rng, iterator)
    Metropolis((;args..., j))
end

@inline function Metropolis(args::As) where As
    (;g, gstate, j, deltafunc, M, rng, lmeta) = args
    T = eltype(g)
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[j]
    newstate = SparseVal((@inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta)))::eltype(gstate), Int32(length(gstate)), j)::SparseVal{eltype(gstate), Int32}

    ΔE = @inline deltafunc((;args..., newstate); j)
    
    efac = exp(-β*ΔE)
    if (ΔE <= zero(T) || rand(rng, T) < efac)
        @inbounds gstate[j] = newstate 
        @hasarg if M isa Ref
            M[] += (newstate - oldstate)
        end
        @hasarg if Δs_j isa Ref
            Δs_j[] = newstate - oldstate
        end
    else
        @hasarg if Δs_j isa Ref
            Δs_j[] = 0
        end
    end

    @inline update!(args.hamiltonian, args)
    return nothing
end
