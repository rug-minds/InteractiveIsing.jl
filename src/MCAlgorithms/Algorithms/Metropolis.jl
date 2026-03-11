export Metropolis
struct Metropolis <: MCAlgorithm end


@inline function IsingMetropolis()
    return Unique(Metropolis())
    # destr = DestructureInput()
    # metro = Metropolis(tracked = tracked)
    # package(SimpleAlgo(metro, destr, Route(destr => metro, :isinggraph => :structure)))
end

@inline MetropolisTracked() = Metropolis()

@inline function Processes.init(::Metropolis, context::Cont) where Cont
    # @show typeof(context)
    (;state) = context

    hamiltonian = state.hamiltonian

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, state)
    proposer = get_proposer(state)
    proposal = @inline rand(rng, proposer)

    ΔE = zero(eltype(state))

    returnargs = (;state, hamiltonian, proposer, rng, proposal, ΔE)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
@inline function Processes.step!(metro::Metropolis, context::C) where {C}
    (;rng, state, hamiltonian, proposer, proposal) = context

    proposal = @inline rand(rng, proposer)
    Ttype = eltype(state)

    β = one(Ttype)/(@inline temp(state))

    ΔE = @inline calculate(ΔH(), hamiltonian, state, proposal)
    
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-β*ΔE)))
        proposal = @inline accept(proposer, proposal)
    end

    injected = @inline inject(context, (;proposal))
    @inline update!(metro, hamiltonian, injected)
    return (;proposal, ΔE)
end
