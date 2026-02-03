export Metropolis
struct Metropolis <: MCAlgorithm 
    track_M::Bool
    track_sj::Bool
end

Metropolis(;tracked = false) = Metropolis(tracked, tracked)

function IsingMetropolis(;tracked = false)
    destr = DestructureInput()
    metro = Metropolis(tracked = tracked)
    SimpleAlgo(tuple(metro), destr, Share(destr, metro))
end

MetropolisTracked() = Metropolis(tracked = true)

# @inline function (::Metropolis)(context::As) where As
function Processes.step!(::Metropolis, context::C) where {C}
    @inline Metropolis_step(context)
end

function Processes.prepare(::Metropolis, context::Cont) where Cont
    # @show typeof(context)

    (;structure) = context
    state = InteractiveIsing.state(structure)
    hamiltonian = structure.hamiltonian
    adj = InteractiveIsing.adj(structure)
    self = structure.self
    isinggraph = structure


    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, isinggraph)
    proposer = get_proposer(isinggraph)
    proposal = FlipProposal{:s, :j, statetype(proposer)}(0, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)
    M = sum(state)
    return (;hamiltonian, proposer, rng, proposal, M, isinggraph, state, adj, self)
end

# function Processes.prepare(::Metropolis, context::Cont) where Cont
#     (;isinggraph) = context

#     rng = Random.MersenneTwister()

#     hamiltonian = isinggraph.hamiltonian
#     state = InteractiveIsing.state(isinggraph)
#     adj = InteractiveIsing.adj(isinggraph)
#     self = isinggraph.self

#     hamiltonian = init!(hamiltonian, isinggraph)
#     proposer = get_proposer(isinggraph)
#     proposal = FlipProposal{:s, :j, statetype(proposer)}(0, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)
#     M = sum(state)
#     return (;rng, isinggraph, state, adj, self, hamiltonian, proposer, proposal, M)
# end



@inline function Metropolis_step(context::C) where C
    (;rng, isinggraph, state, adj, self, hamiltonian, proposer, proposal, M) = context

    proposal = @inline rand(rng, proposer)::FlipProposal{:s, :j, statetype(proposer)}
    # proposal = FlipProposal{:s, :j, statetype(proposer)}(1, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)

    Ttype = eltype(isinggraph)
    β = one(Ttype)/(@inline temp(isinggraph))
    # β = one(Ttype)/T
    
    # ΔE = @inline deltafunc((;context..., newstate), (;j))

    ΔE = @inline ΔH(hamiltonian, (;self = self, s = state, w = adj, hamiltonian...), proposal)

    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-β*ΔE)))
        proposal = @inline accept(state, proposal)
        M += @inline delta(proposal)
    end

    context = @inline inject(context, (;proposal, M))
    
    @inline update!(Metropolis(), hamiltonian, context)

    return (;proposal, M)
end
