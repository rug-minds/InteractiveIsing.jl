export Metropolis
struct Metropolis<: MCAlgorithm end
# struct deltaH end

function Processes.prepare(::Metropolis, @specialize(args))
    (;g) = args
    s = g.state
    wij = g.adj
    self = g.self
    iterator = ising_it(g)
    hamiltonian = init!(g.hamiltonian, g)
    # deltafunc = deltaH(hamiltonian)
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    Δs_j = Ref(zero(eltype(g.state)))
    # newstate = SparseVal(eltype(s)(0), Int32(0), Int32(length(s)))
    
    drule = DeltaRule(:s, j = 0 => eltype(s)(0)) # Specify which spin will be flipped

    lmeta = LayerMetaData(g[1])
    return (;g ,s, wij, iterator, hamiltonian, lmeta, rng, M, Δs_j, self, drule)
    # args = (;s, wij, iterator, hamiltonian, deltafunc, lmeta, rng, newstate)
end


@inline function (::Metropolis)(args::As) where As
    #Define vars
    (;iterator, rng) = args
    j = rand(rng, iterator)
    @inline Metropolis((;args..., j))
end

@inline function Metropolis(args::As) where As
    (;g, s, wij, self, j, rng, lmeta, hamiltonian, drule) = args
    Ttype = eltype(g)
    β = one(Ttype)/(temp(g))

    oldstate = @inbounds s[j]

    # newstate[j] = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))
    drule[j] = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))
    
    # ΔE = @inline deltafunc((;args..., newstate), (;j))

    ΔE = ΔH(hamiltonian, (;args..., self = self, s = s, w = wij, hamiltonian...), drule)
    
    efac = exp(-β*ΔE)
    if (ΔE <= zero(Ttype) || rand(rng, Ttype) < efac)
        @inbounds s[j] = drule[]
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

    @inline update!(Metropolis(), args.hamiltonian, args)
    return nothing
end
