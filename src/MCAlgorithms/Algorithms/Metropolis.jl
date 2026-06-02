export Metropolis
struct Metropolis <: IsingMCAlgorithm end


@inline function IsingMetropolis()
    return Unique(Metropolis())
    # destr = DestructureInput()
    # metro = Metropolis(tracked = tracked)
    # package(CompositeAlgorithm(metro, destr, Route(destr => metro, :isinggraph => :structure)))
end

@inline MetropolisTracked() = Metropolis()

@inline function Processes.init(::Metropolis, context::Cont) where {Cont}
    # @show typeof(context)
    (;model) = context

    hamiltonian = model.hamiltonian

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    proposer = get_proposer(model)

    returnargs = (;model, hamiltonian, proposer, rng)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
@inline function Processes.step!(metro::Metropolis, context::C) where {C}
    (;rng, model, hamiltonian, proposer) = context

    proposal = @inline rand(rng, proposer)
    Ttype = eltype(model)
    T = Ttype(temp(model))

   
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal)
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-ΔE/T)))
        proposal = @inline accept(proposer, proposal)
    end

    @inline update!(metro, hamiltonian, model, proposal)
    return (;proposal, ΔE, T)
end
