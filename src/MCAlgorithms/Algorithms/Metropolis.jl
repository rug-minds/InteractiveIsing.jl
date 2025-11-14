export Metropolis
struct Metropolis<: MCAlgorithm end
# struct deltaH end

function Processes.prepare(::Metropolis, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    self = g.self
    iterator = ising_it(g)
    hamiltonian = init!(g.hamiltonian, g)
    # deltafunc = deltaH(hamiltonian)
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    Δs_j = Ref(zero(eltype(g.state)))
    # newstate = SparseVal(eltype(gstate)(0), Int32(0), Int32(length(gstate)))
    
    drule = DeltaRule(:s, j = 0 => eltype(gstate)(0)) # Specify which spin will be flipped

    lmeta = LayerMetaData(g[1])
    return (;g ,gstate, gadj, iterator, hamiltonian, lmeta, rng, M, Δs_j, self, drule)
    # args = (;gstate, gadj, iterator, hamiltonian, deltafunc, lmeta, rng, newstate)
end


@inline function (::Metropolis)(args::As) where As
    #Define vars
    (;iterator, rng) = args
    j = rand(rng, iterator)
    @inline Metropolis((;args..., j))
end

@inline function Metropolis(args::As) where As
    (;g, gstate, gadj, self, j, rng, lmeta, hamiltonian, drule) = args
    Ttype = eltype(g)
    β = one(Ttype)/(temp(g))

    oldstate = @inbounds gstate[j]

    # newstate[j] = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))
    drule[j] = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))
    
    # ΔE = @inline deltafunc((;args..., newstate), (;j))

    ΔE = ΔH(hamiltonian, (;args..., self = self, s = gstate, w = gadj, hamiltonian...), drule)
    
    efac = exp(-β*ΔE)
    if (ΔE <= zero(Ttype) || rand(rng, Ttype) < efac)
        @inbounds gstate[j] = drule[]
        @hasarg if M isa Ref
            M[] += (drule[] - oldstate)
        end
        @hasarg if Δs_j isa Ref
            Δs_j[] = drule[] - oldstate
        end
    else
        @hasarg if Δs_j isa Ref
            Δs_j[] = 0
        end
    end

    @inline update!(args.hamiltonian, args)
    return nothing
end
