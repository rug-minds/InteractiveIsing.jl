export Metropolis
struct Metropolis <: IsingMCAlgorithm end


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
    T = temp(state)

    returnargs = (;state, hamiltonian, proposer, rng, proposal, ΔE, T)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
@inline function Processes.step!(metro::Metropolis, context::C) where {C}
    (;rng, state, hamiltonian, proposer, proposal) = context

    proposal = @inline rand(rng, proposer)
    Ttype = eltype(state)

   
    ΔE = @inline calculate(ΔH(), hamiltonian, state, proposal)
    T = temp(state)
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-ΔE/T)))
        proposal = @inline accept(proposer, proposal)
    end

    @inline update!(metro, hamiltonian, state, proposal)
    return (;proposal, ΔE, T)
end
