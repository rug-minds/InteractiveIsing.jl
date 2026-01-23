export Metropolis
struct Metropolis <: MCAlgorithm 
    track_M::Bool
    track_sj::Bool
end

Metropolis() = Metropolis(false, false)

const MetropolisStandard = Metropolis(false, false)
const MetropolisTracked = Metropolis(true, true)


# @inline function (::Metropolis)(context::As) where As
@ProcessAlgorithm function Metropolis(iterator::I, rng, context::C) where {I, C}
    #Define vars
    # (;iterator, rng) = context
    j = rand(rng, iterator)
    @inline Metropolis_step(context, j)
end


function Processes.prepare(::Metropolis, context::A) where A
    (;g) = context

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

    layer = g.layers[1]
    return (;g ,s, wij, iterator, hamiltonian, layer, rng, M, Δs_j, self, drule)
end



function Metropolis_step(context::C, j) where C
    (;g, s, wij, self, rng, layer, hamiltonian, drule) = context

    Ttype = eltype(g)
    β = one(Ttype)/(temp(g))

    oldstate = @inbounds s[j]

    # newstate[j] = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))
    drule[j] = @inline sampleState(statetype(layer), oldstate, rng, stateset(layer))
    
    # ΔE = @inline deltafunc((;context..., newstate), (;j))

    ΔE = @inline ΔH(hamiltonian, (;self = self, s = s, w = wij, hamiltonian...), drule)
    
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

    @inline update!(Metropolis(), context.hamiltonian, context)
    return nothing
end
