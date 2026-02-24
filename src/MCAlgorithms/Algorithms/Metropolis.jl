export Metropolis
struct Metropolis <: MCAlgorithm 
    track_M::Bool
    track_sj::Bool
end

Metropolis(;tracked = false) = Metropolis(tracked, tracked)

function IsingMetropolis(;tracked = false)
    destr = DestructureInput()
    metro = Metropolis(tracked = tracked)
    package(SimpleAlgo(metro, destr, Route(destr => metro, :isinggraph => :structure)))
end

MetropolisTracked() = Metropolis(tracked = true)

function Processes.init(::Metropolis, context::Cont) where Cont
    # @show typeof(context)
    (;structure) = context

    state = InteractiveIsing.state(structure)
    hamiltonian = structure.hamiltonian
    adj = InteractiveIsing.adj(structure)
    self = structure.self


    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, structure)
    proposer = get_proposer(structure)
    proposal = FlipProposal{:s, :j, statetype(proposer)}(0, zero(statetype(proposer)), zero(statetype(proposer)), 1, false)
    M = sum(state)
    ΔE = zero(eltype(state))

    returnargs = (;hamiltonian, proposer, rng, proposal, M, ΔE, isinggraph = structure, state, adj, self)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
function Processes.step!(::Metropolis, context::C) where {C}
    (;rng, isinggraph, state, adj, self, hamiltonian, proposer, proposal, M) = context

    proposal = @inline rand(rng, proposer)::FlipProposal{:s, :j, statetype(proposer)}

    Ttype = eltype(isinggraph)
    β = one(Ttype)/(@inline temp(isinggraph))    

    ΔE = @inline calculate(ΔH(), hamiltonian, (;self = self, s = state, w = adj, M, hamiltonian...), proposal)
    
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-β*ΔE)))
        proposal = @inline accept(state, proposal)
        M += @inline delta(proposal)
    end

    context = @inline inject(context, (;proposal, M))
    
    @inline update!(Metropolis(), hamiltonian, context)

    return (;proposal, M, ΔE)
end

# function Processes.init(::Metropolis, context::Cont) where Cont
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



# @inline function Metropolis_step(context::C) where C
    
# end
