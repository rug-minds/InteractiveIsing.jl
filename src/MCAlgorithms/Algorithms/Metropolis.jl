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
    (;model) = context

    hamiltonian = model.hamiltonian

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    proposer = get_proposer(model)
    proposal = @inline rand(rng, proposer)

    ΔE = zero(eltype(model))
    T = temp(model)

    returnargs = (;model, hamiltonian, proposer, rng, proposal, ΔE, T)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
@inline function Processes.step!(metro::Metropolis, context::C) where {C}
    (;rng, model, hamiltonian, proposer, proposal) = context

    proposal = @inline rand(rng, proposer)
    Ttype = eltype(model)

   
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal)
    T = temp(model)
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-ΔE/T)))
        proposal = @inline accept(proposer, proposal)
    end

    @inline update!(metro, hamiltonian, model, proposal)
    return (;proposal, ΔE, T)
end
