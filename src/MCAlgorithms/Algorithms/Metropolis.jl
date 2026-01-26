export Metropolis
struct Metropolis <: MCAlgorithm 
    track_M::Bool
    track_sj::Bool
end

Metropolis() = Metropolis(false, false)

const MetropolisStandard = Metropolis(false, false)
const MetropolisTracked = Metropolis(true, true)


# @inline function (::Metropolis)(context::As) where As
@ProcessAlgorithm function Metropolis(rng, context::C) where {C}
    @inline Metropolis_step(context)
end


function Processes.prepare(::Metropolis, context::Cont) where Cont
    (;isinggraph, state, hamiltonian) = context

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, isinggraph)
    proposer = get_proposer(isinggraph)
    proposal = FlipProposal{:s, :j, statetype(proposer)}(0, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)
    T = eltype(isinggraph)(1)
    println("Has T")
    return (;hamiltonian, proposer, rng, proposal, T)
end



@inline function Metropolis_step(context::C) where C
    (;isinggraph, state, adj, self, rng, hamiltonian, proposer, proposal, T) = context

    proposal = @inline rand(rng, proposer)::FlipProposal{:s, :j, statetype(proposer)}
    # proposal = FlipProposal{:s, :j, statetype(proposer)}(1, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)

    Ttype = eltype(isinggraph)
    # β = one(Ttype)/(temp(isinggraph))
    β = one(Ttype)/T
    
    # ΔE = @inline deltafunc((;context..., newstate), (;j))

    ΔE = @inline ΔH(hamiltonian, (;self = self, s = state, w = adj, hamiltonian...), proposal)

    if (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-β*ΔE))
        proposal = @inline accept(proposal)
        
        @inbounds state[at_idx(proposal)] = to_val(proposal)
    #     @hasarg if M isa Ref
    #         M[] += (drule[] - oldstate)
    #     end
    #     @hasarg if Δs_j isa Ref
    #         Δs_j[] = drule[] - oldstate
    #     end
    # else
    #     @hasarg if Δs_j isa Ref
    #         Δs_j[] = 0
    #     end
    end

    @inline update!(Metropolis(), context.hamiltonian, context)
    # return (;proposal)
end

function Processes.prepare(::Metropolis, context::Cont) where Cont
    (;isinggraph, state, hamiltonian) = context

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, isinggraph)
    proposer = get_proposer(isinggraph)
    proposal = FlipProposal{:s, :j, statetype(proposer)}(0, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)
    # T = eltype(isinggraph)(1)
    return (;hamiltonian, proposer, rng, proposal)
end



@inline function Metropolis_step(context::C) where C
    (;isinggraph, state, adj, self, rng, hamiltonian, proposer, proposal) = context

    proposal = @inline rand(rng, proposer)::FlipProposal{:s, :j, statetype(proposer)}
    # proposal = FlipProposal{:s, :j, statetype(proposer)}(1, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)

    Ttype = eltype(isinggraph)
    β = one(Ttype)/(temp(isinggraph))
    # β = one(Ttype)/T
    
    # ΔE = @inline deltafunc((;context..., newstate), (;j))

    ΔE = @inline ΔH(hamiltonian, (;self = self, s = state, w = adj, hamiltonian...), proposal)

    if (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-β*ΔE))
        proposal = @inline accept(proposal)
        
        @inbounds state[at_idx(proposal)] = to_val(proposal)
    #     @hasarg if M isa Ref
    #         M[] += (drule[] - oldstate)
    #     end
    #     @hasarg if Δs_j isa Ref
    #         Δs_j[] = drule[] - oldstate
    #     end
    # else
    #     @hasarg if Δs_j isa Ref
    #         Δs_j[] = 0
    #     end
    end

    @inline update!(Metropolis(), context.hamiltonian, context)
    # return (;proposal)
end
